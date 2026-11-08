"""Deterministic FIFO token-budget batch formation tests."""

from concurrent.futures import Future
from time import perf_counter
from unittest.mock import patch

import numpy as np
import pytest

from backend.api.services.dynamic_batch import (
    BatchJob,
    DynamicBatchReranker,
)


class RecordingModel:
    def __init__(self):
        self.batches = []

    def predict(self, pairs, **kwargs):
        self.batches.append([
            query for query, _ in pairs
        ])

        return np.arange(
            len(pairs),
            dtype=np.float32,
        )


@pytest.mark.parametrize(
    "token_costs,budget,expected",
    [
        ([30, 40], 100, [["q0", "q1"]]),
        ([60, 60], 100, [["q0"], ["q1"]]),
        ([40, 70, 20], 100, [["q0"], ["q1", "q2"]]),
        ([30, 40, 50], 100, [["q0", "q1"], ["q2"]]),
    ],
)
def test_fifo_token_batch_formation(
    token_costs,
    budget,
    expected,
):
    model = RecordingModel()

    scheduler = DynamicBatchReranker(
        batch_policy="token",
        max_requests=4,
        max_pairs=100,
        max_tokens=budget,
        max_wait_ms=0,
        timeout_seconds=10,
    )

    try:
        with patch(
            "backend.api.services.dynamic_batch.get_cross_encoder",
            return_value=model,
        ), patch(
            "backend.api.services.dynamic_batch.resolve_device",
            return_value="cpu",
        ):
            futures = []

            with scheduler._condition:
                for i, cost in enumerate(token_costs):
                    future = Future()

                    job = BatchJob(
                        query=f"q{i}",
                        candidates=[
                            {
                                "document_id": i,
                                "content": "document",
                            }
                        ],
                        top_k=1,
                        future=future,
                        submitted_at=perf_counter(),
                        token_count=cost,
                    )

                    scheduler._queue.append(job)
                    futures.append(future)

                scheduler._condition.notify_all()

            for future in futures:
                future.result(timeout=10)

            assert model.batches == expected

    finally:
        scheduler.close()
