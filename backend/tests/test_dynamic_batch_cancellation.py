"""Concurrency and cancellation tests for dynamic batching."""

import threading
import time
from concurrent.futures import Future
from unittest.mock import patch

import numpy as np

from backend.api.services.dynamic_batch import (
    BatchJob,
    DynamicBatchReranker,
)


def make_candidate(document_id):
    return {
        "document_id": document_id,
        "external_id": str(document_id),
        "content": f"Document {document_id}",
        "lexical_rank": 1,
        "vector_rank": None,
    }


def test_cancelled_future_does_not_fail_other_jobs():
    """Cancelling one request must not fail another in the batch."""

    scheduler = DynamicBatchReranker(
        max_requests=4,
        max_pairs=512,
        max_wait_ms=5,
    )

    try:
        cancelled_future = Future()
        healthy_future = Future()

        cancelled_job = BatchJob(
            query="test query",
            candidates=[make_candidate(1)],
            top_k=1,
            future=cancelled_future,
            submitted_at=time.perf_counter(),
        )

        healthy_job = BatchJob(
            query="test query",
            candidates=[make_candidate(2)],
            top_k=1,
            future=healthy_future,
            submitted_at=time.perf_counter(),
        )

        inference_started = threading.Event()
        release_inference = threading.Event()

        class FakeModel:
            def predict(self, pairs, **kwargs):
                inference_started.set()

                if not release_inference.wait(timeout=5):
                    raise TimeoutError(
                        "Fake inference was not released."
                    )

                return np.array(
                    [0.9, 0.8],
                    dtype=np.float32,
                )

        with (
            patch(
                "backend.api.services.dynamic_batch.get_cross_encoder",
                return_value=FakeModel(),
            ),
            patch(
                "backend.api.services.dynamic_batch.resolve_device",
                return_value="cpu",
            ),
        ):
            worker = threading.Thread(
                target=scheduler._process_batch,
                args=([cancelled_job, healthy_job],),
            )

            worker.start()

            assert inference_started.wait(timeout=5)

            assert cancelled_future.cancel()

            release_inference.set()

            worker.join(timeout=5)

            assert not worker.is_alive()
            assert cancelled_future.cancelled()

            ranked, profile = healthy_future.result(timeout=1)

            assert len(ranked) == 1
            assert ranked[0]["document_id"] == 2

            assert profile["batch_request_count"] == 2
            assert profile["batch_pair_count"] == 2

    finally:
        release_inference.set()
        scheduler.close()
