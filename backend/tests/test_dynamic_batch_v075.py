from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np

from backend.api.services.dynamic_batch import (
    DynamicBatchReranker,
)


class FakeCrossEncoder:
    def predict(self, pairs, **kwargs):
        return np.asarray(
            [float(len(text)) for _, text in pairs]
        )


def make_candidates(prefix, count=3):
    return [
        {
            "document_id": i + 1,
            "external_id": f"{prefix}-{i}",
            "content": f"{prefix}-document-{i}",
            "lexical_rank": i + 1,
            "vector_rank": None,
        }
        for i in range(count)
    ]


def test_dynamic_batch_single_request():
    with patch(
        "backend.api.services.dynamic_batch.get_cross_encoder",
        return_value=FakeCrossEncoder(),
    ), patch(
        "backend.api.services.dynamic_batch.resolve_device",
        return_value="cpu",
    ):
        scheduler = DynamicBatchReranker(
            max_requests=4,
            max_pairs=100,
            max_wait_ms=1,
        )

        try:
            results, profile = scheduler.submit(
                query="test",
                candidates=make_candidates("a"),
                top_k=2,
            )

            assert len(results) == 2
            assert profile["batch_request_count"] >= 1
        finally:
            scheduler.close()


def test_dynamic_batch_concurrent_requests():
    with patch(
        "backend.api.services.dynamic_batch.get_cross_encoder",
        return_value=FakeCrossEncoder(),
    ), patch(
        "backend.api.services.dynamic_batch.resolve_device",
        return_value="cpu",
    ):
        scheduler = DynamicBatchReranker(
            max_requests=4,
            max_pairs=100,
            max_wait_ms=20,
        )

        try:
            def submit(index):
                return scheduler.submit(
                    query=f"query-{index}",
                    candidates=make_candidates(str(index)),
                    top_k=2,
                )

            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(submit, range(4)))

            assert len(results) == 4
            assert all(len(ranked) == 2 for ranked, _ in results)
            assert all(
                profile["batch_pair_count"] <= 100
                for _, profile in results
            )
        finally:
            scheduler.close()


def test_dynamic_batch_shutdown_rejects_new_jobs():
    scheduler = DynamicBatchReranker(
        max_requests=4,
        max_pairs=100,
        max_wait_ms=1,
    )

    scheduler.close()

    try:
        scheduler.submit(
            query="test",
            candidates=make_candidates("shutdown"),
            top_k=2,
        )
    except RuntimeError as exc:
        assert "closed" in str(exc)
    else:
        raise AssertionError(
            "Scheduler accepted a request after shutdown."
        )
