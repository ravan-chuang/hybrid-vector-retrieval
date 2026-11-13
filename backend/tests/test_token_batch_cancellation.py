"""Cancellation and queue-capacity regression tests."""

from concurrent.futures import Future
from threading import Event, Thread
from time import perf_counter
from unittest.mock import patch

import numpy as np
import pytest

from backend.api.services.dynamic_batch import (
    BatchJob,
    BatchQueueFullError,
    DynamicBatchReranker,
)
from backend.api.services.token_budget import TokenBudgetEstimate


class FakeModel:
    tokenizer = object()
    max_seq_length = 512

    def predict(self, pairs, **kwargs):
        return np.arange(len(pairs), dtype=np.float32)


def make_job(index, tokens=30):
    return BatchJob(
        query=f"q{index}",
        candidates=[
            {
                "document_id": index,
                "content": f"document {index}",
            }
        ],
        top_k=1,
        future=Future(),
        submitted_at=perf_counter(),
        token_count=tokens,
    )


def fake_estimate(tokenizer, query, candidates, max_length):
    return TokenBudgetEstimate(
        pair_count=len(candidates),
        total_tokens=30,
        max_pair_tokens=30,
    )


def test_cancelled_jobs_release_queue_capacity():
    """Cancelled jobs release capacity before admitting new requests."""

    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from unittest.mock import patch

    worker_release = Event()
    worker_ready = Event()

    original_worker_loop = DynamicBatchReranker._worker_loop

    def controlled_worker_loop(self):
        worker_ready.set()
        if not worker_release.wait(timeout=10):
            raise TimeoutError("Worker was not released.")
        return original_worker_loop(self)

    with patch.object(
        DynamicBatchReranker,
        "_worker_loop",
        controlled_worker_loop,
    ):
        scheduler = DynamicBatchReranker(
            batch_policy="token",
            max_queue_size=2,
            max_wait_ms=0,
            timeout_seconds=10,
        )

    try:
        assert worker_ready.wait(timeout=5)

        with patch(
            "backend.api.services.dynamic_batch.get_cross_encoder",
            return_value=FakeModel(),
        ), patch(
            "backend.api.services.dynamic_batch.estimate_request_tokens",
            side_effect=fake_estimate,
        ):
            first = make_job(1)
            second = make_job(2)
            first.future.cancel()

            with scheduler._condition:
                scheduler._queue.append(first)
                scheduler._queue.append(second)

            with ThreadPoolExecutor(max_workers=1) as executor:
                submitted = executor.submit(
                    scheduler.submit,
                    query="new request",
                    candidates=[
                        {
                            "document_id": 3,
                            "content": "new document",
                        }
                    ],
                    top_k=1,
                )

                # Wait for submission to finish admission.
                # Worker is blocked, so it cannot consume jobs.
                from time import monotonic

                deadline = monotonic() + 5

                while monotonic() < deadline:
                    with scheduler._condition:
                        queued_ids = [
                            job.candidates[0]["document_id"]
                            for job in scheduler._queue
                        ]

                    if queued_ids == [2, 3]:
                        break

                    if submitted.done():
                        submitted.result()

                    worker_release.wait(timeout=0.001)

                assert queued_ids == [2, 3], (
                    f"Unexpected queue contents: {queued_ids}"
                )

                worker_release.set()

                ranked, profile = submitted.result(timeout=10)

                assert ranked[0]["document_id"] == 3
                assert profile["batch_pair_count"] >= 1

    finally:
        worker_release.set()
        scheduler.close()


def test_cancelled_jobs_excluded_from_token_budget():
    """Cancelled jobs must not consume batch token capacity."""

    scheduler = DynamicBatchReranker(
        batch_policy="token",
        max_requests=4,
        max_pairs=10,
        max_tokens=100,
        max_wait_ms=0,
    )

    recorded_batches = []

    class RecordingModel(FakeModel):
        def predict(self, pairs, **kwargs):
            recorded_batches.append(
                [query for query, _ in pairs]
            )
            return np.arange(len(pairs), dtype=np.float32)

    try:
        with patch(
            "backend.api.services.dynamic_batch.get_cross_encoder",
            return_value=RecordingModel(),
        ), patch(
            "backend.api.services.dynamic_batch.resolve_device",
            return_value="cpu",
        ):
            cancelled = make_job(1, tokens=90)
            cancelled.future.cancel()

            active_a = make_job(2, tokens=40)
            active_b = make_job(3, tokens=40)

            with scheduler._condition:
                scheduler._queue.extend(
                    [cancelled, active_a, active_b]
                )
                scheduler._condition.notify_all()

            active_a.future.result(timeout=5)
            active_b.future.result(timeout=5)

            assert recorded_batches == [["q2", "q3"]]

    finally:
        scheduler.close()


def test_cancellation_during_inference_is_isolated():
    """Cancelling one request must not fail another request."""

    started = Event()
    release = Event()

    class BlockingModel(FakeModel):
        def predict(self, pairs, **kwargs):
            started.set()

            if not release.wait(timeout=5):
                raise TimeoutError("Inference was not released.")

            return np.arange(len(pairs), dtype=np.float32)

    scheduler = DynamicBatchReranker(
        batch_policy="token",
        max_requests=2,
        max_pairs=10,
        max_tokens=100,
        max_wait_ms=0,
    )

    cancelled = make_job(1, tokens=30)
    active = make_job(2, tokens=30)

    try:
        with patch(
            "backend.api.services.dynamic_batch.get_cross_encoder",
            return_value=BlockingModel(),
        ), patch(
            "backend.api.services.dynamic_batch.resolve_device",
            return_value="cpu",
        ):
            with scheduler._condition:
                scheduler._queue.extend([cancelled, active])
                scheduler._condition.notify_all()

            assert started.wait(timeout=5)

            cancelled.future.cancel()
            release.set()

            ranked, profile = active.future.result(timeout=5)

            assert ranked[0]["document_id"] == 2
            assert profile["batch_request_count"] == 2
            assert cancelled.future.cancelled()

    finally:
        release.set()
        scheduler.close()
