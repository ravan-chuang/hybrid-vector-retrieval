"""Tests for dynamic batch identity instrumentation."""

from concurrent.futures import Future
from unittest.mock import patch

import numpy as np

from backend.api.services.dynamic_batch import (
    BatchJob,
    DynamicBatchReranker,
)


class FakeModel:
    def predict(self, pairs, **kwargs):
        return np.arange(len(pairs), dtype=float)


def make_job(index):
    return BatchJob(
        query=f"query-{index}",
        candidates=[
            {
                "document_id": f"doc-{index}",
                "content": f"content-{index}",
            }
        ],
        top_k=1,
        future=Future(),
        submitted_at=0.0,
    )


def test_same_batch_shares_id_and_different_batches_do_not():
    scheduler = DynamicBatchReranker(
        max_requests=4,
        max_pairs=100,
        max_wait_ms=0,
    )

    try:
        first = [make_job(1), make_job(2)]
        second = [make_job(3), make_job(4)]

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
            scheduler._process_batch(first)
            scheduler._process_batch(second)

        first_ids = [
            job.future.result()[1]["batch_id"]
            for job in first
        ]

        second_ids = [
            job.future.result()[1]["batch_id"]
            for job in second
        ]

        assert first_ids[0] == first_ids[1]
        assert second_ids[0] == second_ids[1]
        assert first_ids[0] != second_ids[0]

    finally:
        scheduler.close()


def test_empty_request_has_no_batch_id():
    scheduler = DynamicBatchReranker(
        max_wait_ms=0,
    )

    try:
        results, profile = scheduler.submit(
            query="empty",
            candidates=[],
            top_k=10,
        )

        assert results == []
        assert profile["batch_id"] is None

    finally:
        scheduler.close()
