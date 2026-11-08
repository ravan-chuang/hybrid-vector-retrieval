"""Token-aware dynamic batching tests."""

from concurrent.futures import Future
from unittest.mock import patch

import pytest

from backend.api.services.dynamic_batch import (
    DynamicBatchReranker,
)
from backend.api.services.token_budget import (
    TokenBudgetEstimate,
)


class FakeModel:
    tokenizer = object()
    max_seq_length = 512


def candidate(document_id):
    return {
        "document_id": document_id,
        "content": "test document",
    }


def test_invalid_batch_policy():
    with pytest.raises(ValueError):
        DynamicBatchReranker(
            batch_policy="unknown",
        )


def test_invalid_token_budget():
    with pytest.raises(ValueError):
        DynamicBatchReranker(
            max_tokens=0,
        )


def test_request_exceeds_token_budget():
    with patch(
        "backend.api.services.dynamic_batch.get_cross_encoder",
        return_value=FakeModel(),
    ), patch(
        "backend.api.services.dynamic_batch.estimate_request_tokens",
        return_value=TokenBudgetEstimate(
            pair_count=1,
            total_tokens=100,
            max_pair_tokens=100,
        ),
    ):
        scheduler = DynamicBatchReranker(
            batch_policy="token",
            max_tokens=50,
        )

        try:
            with pytest.raises(ValueError):
                scheduler.submit(
                    query="query",
                    candidates=[candidate(1)],
                    top_k=1,
                )
        finally:
            scheduler.close()


def test_pair_policy_does_not_tokenize():
    with patch(
        "backend.api.services.dynamic_batch.estimate_request_tokens",
    ) as estimator:
        scheduler = DynamicBatchReranker(
            batch_policy="pair",
        )

        scheduler.close()

        estimator.assert_not_called()


def test_token_budget_splits_batches():
    """Two requests exceeding the combined token budget
    must not execute in the same batch.
    """

    import threading
    import time
    import numpy as np

    class RecordingModel:
        tokenizer = object()
        max_seq_length = 512

        def __init__(self):
            self.batch_sizes = []
            self.lock = threading.Lock()

        def predict(self, pairs, **kwargs):
            with self.lock:
                self.batch_sizes.append(len(pairs))

            return np.arange(
                len(pairs),
                dtype=np.float32,
            )

    model = RecordingModel()

    def fake_estimator(
        tokenizer,
        query,
        candidates,
        max_length,
    ):
        return TokenBudgetEstimate(
            pair_count=len(candidates),
            total_tokens=60,
            max_pair_tokens=60,
        )

    with patch(
        "backend.api.services.dynamic_batch.get_cross_encoder",
        return_value=model,
    ), patch(
        "backend.api.services.dynamic_batch.estimate_request_tokens",
        side_effect=fake_estimator,
    ), patch(
        "backend.api.services.dynamic_batch.resolve_device",
        return_value="cpu",
    ):
        scheduler = DynamicBatchReranker(
            max_requests=4,
            max_pairs=100,
            max_wait_ms=20,
            max_queue_size=10,
            timeout_seconds=5,
            batch_policy="token",
            max_tokens=100,
        )

        results = []
        errors = []

        def submit_request(document_id):
            try:
                result = scheduler.submit(
                    query="query",
                    candidates=[candidate(document_id)],
                    top_k=1,
                )
                results.append(result)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(
                target=submit_request,
                args=(i,),
            )
            for i in range(2)
        ]

        try:
            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join(timeout=5)

            assert not errors
            assert len(results) == 2

            # Each request costs 60 tokens.
            # Combined cost 120 > max_tokens=100.
            assert model.batch_sizes == [1, 1]

        finally:
            scheduler.close()
