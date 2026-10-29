"""Bounded cross-request batching for Cross-Encoder inference."""

import os
import time
from collections import deque
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from threading import Condition, Thread

import numpy as np
import torch

from backend.api.services.cross_encoder import (
    _inference_lock,
    get_cross_encoder,
    resolve_batch_size,
    resolve_device,
)


class BatchQueueFullError(RuntimeError):
    pass


class BatchTimeoutError(RuntimeError):
    pass


@dataclass
class BatchJob:
    query: str
    candidates: list[dict]
    top_k: int
    future: Future
    submitted_at: float


class DynamicBatchReranker:
    """Single-worker, bounded, cross-request batching scheduler."""

    def __init__(
        self,
        max_requests: int = 4,
        max_pairs: int = 512,
        max_wait_ms: float = 5.0,
        max_queue_size: int = 128,
        timeout_seconds: float = 10.0,
    ):
        if min(max_requests, max_pairs, max_queue_size) < 1:
            raise ValueError("Batch limits must be positive.")

        if max_wait_ms < 0 or timeout_seconds <= 0:
            raise ValueError("Invalid timing configuration.")

        self.max_requests = max_requests
        self.max_pairs = max_pairs
        self.max_wait_seconds = max_wait_ms / 1000.0
        self.max_queue_size = max_queue_size
        self.timeout_seconds = timeout_seconds

        self._queue = deque()
        self._condition = Condition()
        self._closed = False

        self._worker = Thread(
            target=self._worker_loop,
            name="dynamic-cross-encoder-worker",
            daemon=True,
        )
        self._worker.start()

    def submit(
        self,
        query: str,
        candidates: list[dict],
        top_k: int,
    ):
        if not candidates:
            return [], {
                "batch_queue_wait_ms": 0.0,
                "batch_inference_ms": 0.0,
                "batch_request_count": 0,
                "batch_pair_count": 0,
            }

        if len(candidates) > self.max_pairs:
            raise ValueError(
                "Request candidate count exceeds max_pairs."
            )

        future = Future()

        job = BatchJob(
            query=query,
            candidates=candidates,
            top_k=top_k,
            future=future,
            submitted_at=time.perf_counter(),
        )

        with self._condition:
            if self._closed:
                raise RuntimeError("Batch scheduler is closed.")

            if len(self._queue) >= self.max_queue_size:
                raise BatchQueueFullError(
                    "Dynamic batching queue is full."
                )

            self._queue.append(job)
            self._condition.notify()

        try:
            return future.result(timeout=self.timeout_seconds)
        except FutureTimeoutError as exc:
            future.cancel()
            raise BatchTimeoutError(
                "Dynamic batching request timed out."
            ) from exc

    def _worker_loop(self):
        while True:
            with self._condition:
                while not self._queue and not self._closed:
                    self._condition.wait()

                if self._closed and not self._queue:
                    return

                first = self._queue.popleft()

                if first.future.cancelled():
                    continue

                batch = [first]
                pair_count = len(first.candidates)

                deadline = (
                    first.submitted_at + self.max_wait_seconds
                )

                while len(batch) < self.max_requests:
                    while self._queue and self._queue[0].future.cancelled():
                        self._queue.popleft()

                    if self._queue:
                        next_job = self._queue[0]
                        next_pairs = len(next_job.candidates)

                        if pair_count + next_pairs > self.max_pairs:
                            break

                        next_job = self._queue.popleft()

                        if next_job.future.cancelled():
                            continue

                        batch.append(next_job)
                        pair_count += next_pairs
                        continue

                    remaining = deadline - time.perf_counter()

                    if remaining <= 0:
                        break

                    self._condition.wait(timeout=remaining)

            self._process_batch(batch)

    def _process_batch(self, batch):
        active = [
            job for job in batch
            if not job.future.cancelled()
        ]

        if not active:
            return

        started_at = time.perf_counter()

        pairs = [
            (job.query, candidate["content"])
            for job in active
            for candidate in job.candidates
        ]

        try:
            model = get_cross_encoder()

            with _inference_lock:
                scores = model.predict(
                    pairs,
                    batch_size=resolve_batch_size(),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )

                if resolve_device() == "mps":
                    torch.mps.synchronize()

            finished_at = time.perf_counter()

            scores = np.asarray(scores).reshape(-1)

            if len(scores) != len(pairs):
                raise RuntimeError(
                    "Cross-Encoder score count mismatch."
                )

            offset = 0

            for job in active:
                count = len(job.candidates)
                job_scores = scores[offset:offset + count]
                offset += count

                ranked = [
                    {
                        **candidate,
                        "cross_encoder_score": float(score),
                    }
                    for candidate, score in zip(
                        job.candidates,
                        job_scores,
                    )
                ]

                ranked.sort(
                    key=lambda item: (
                        -item["cross_encoder_score"],
                        item["document_id"],
                    )
                )

                profile = {
                    "batch_queue_wait_ms": round(
                        (started_at - job.submitted_at) * 1000,
                        3,
                    ),
                    "batch_inference_ms": round(
                        (finished_at - started_at) * 1000,
                        3,
                    ),
                    "batch_request_count": len(active),
                    "batch_pair_count": len(pairs),
                }

                if not job.future.cancelled():
                    job.future.set_result(
                        (ranked[:job.top_k], profile)
                    )

        except Exception as exc:
            for job in active:
                if not job.future.done():
                    job.future.set_exception(exc)

    def close(self):
        """Reject queued jobs and wait briefly for active inference."""

        with self._condition:
            if self._closed:
                return

            self._closed = True

            while self._queue:
                job = self._queue.popleft()

                if not job.future.done():
                    job.future.set_exception(
                        RuntimeError(
                            "Dynamic batching scheduler is shutting down."
                        )
                    )

            self._condition.notify_all()

        self._worker.join(timeout=10.0)

        if self._worker.is_alive():
            print(
                "[DynamicBatch] Warning: worker did not stop "
                "within 10 seconds."
            )


_scheduler = None
_scheduler_lock = __import__("threading").Lock()


def get_dynamic_batch_reranker():
    global _scheduler

    with _scheduler_lock:
        if _scheduler is None:
            _scheduler = DynamicBatchReranker(
                max_requests=int(
                    os.getenv("RERANK_MAX_BATCH_REQUESTS", "4")
                ),
                max_pairs=int(
                    os.getenv("RERANK_MAX_BATCH_PAIRS", "512")
                ),
                max_wait_ms=float(
                    os.getenv("RERANK_MAX_BATCH_WAIT_MS", "5")
                ),
                max_queue_size=int(
                    os.getenv("RERANK_MAX_QUEUE_SIZE", "128")
                ),
                timeout_seconds=float(
                    os.getenv("RERANK_BATCH_TIMEOUT_SECONDS", "10")
                ),
            )

        return _scheduler


def shutdown_dynamic_batch_reranker():
    """Shut down the process-local batching scheduler."""

    global _scheduler

    with _scheduler_lock:
        scheduler = _scheduler
        _scheduler = None

    if scheduler is not None:
        scheduler.close()
