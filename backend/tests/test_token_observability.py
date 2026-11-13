"""Token-aware scheduler observability contract tests."""

from concurrent.futures import Future
from time import perf_counter
from unittest.mock import patch

import numpy as np

from backend.api.services.dynamic_batch import (
    BatchJob,
    DynamicBatchReranker,
)


class FakeModel:
    def predict(self, pairs, **kwargs):
        return np.arange(
            len(pairs),
            dtype=np.float32,
        )


def test_token_profile_values():
    scheduler = DynamicBatchReranker(
        batch_policy="token",
        max_tokens=100,
    )

    future = Future()

    job = BatchJob(
        query="query",
        candidates=[
            {
                "document_id": 1,
                "content": "document",
            }
        ],
        top_k=1,
        future=future,
        submitted_at=perf_counter(),
        token_count=60,
        token_counting_ms=1.25,
    )

    try:
        with patch(
            "backend.api.services.dynamic_batch.get_cross_encoder",
            return_value=FakeModel(),
        ), patch(
            "backend.api.services.dynamic_batch.resolve_device",
            return_value="cpu",
        ):
            scheduler._process_batch([job])

        _, profile = future.result(timeout=2)

        assert profile["token_counting_ms"] == 1.25
        assert profile["batch_token_count"] == 60
        assert profile["batch_token_utilization"] == 0.6

    finally:
        scheduler.close()


def test_pair_policy_token_metrics():
    scheduler = DynamicBatchReranker(
        batch_policy="pair",
    )

    future = Future()

    job = BatchJob(
        query="query",
        candidates=[
            {
                "document_id": 1,
                "content": "document",
            }
        ],
        top_k=1,
        future=future,
        submitted_at=perf_counter(),
    )

    try:
        with patch(
            "backend.api.services.dynamic_batch.get_cross_encoder",
            return_value=FakeModel(),
        ), patch(
            "backend.api.services.dynamic_batch.resolve_device",
            return_value="cpu",
        ):
            scheduler._process_batch([job])

        _, profile = future.result(timeout=2)

        assert profile["token_counting_ms"] == 0.0
        assert profile["batch_token_count"] is None
        assert profile["batch_token_utilization"] is None

    finally:
        scheduler.close()
