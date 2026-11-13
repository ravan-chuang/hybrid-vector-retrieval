"""Real Cross-Encoder ranking parity across batching policies."""

import numpy as np

from backend.api.services.cross_encoder import (
    get_cross_encoder,
)
from backend.api.services.dynamic_batch import (
    DynamicBatchReranker,
)


def test_pair_and_token_policy_ranking_parity():
    model = get_cross_encoder()

    query = "What is dynamic batching?"

    candidates = [
        {
            "document_id": 1,
            "content": (
                "Dynamic batching combines inference requests "
                "to improve accelerator utilization."
            ),
        },
        {
            "document_id": 2,
            "content": (
                "Database indexing improves query execution."
            ),
        },
        {
            "document_id": 3,
            "content": (
                "Batch inference processes multiple inputs "
                "together using a neural network."
            ),
        },
        {
            "document_id": 4,
            "content": (
                "Machine learning systems require careful "
                "resource scheduling."
            ),
        },
    ]

    pair_scheduler = DynamicBatchReranker(
        batch_policy="pair",
        max_requests=4,
        max_pairs=100,
        max_wait_ms=0,
        timeout_seconds=30,
    )

    token_scheduler = DynamicBatchReranker(
        batch_policy="token",
        max_requests=4,
        max_pairs=100,
        max_tokens=4096,
        max_wait_ms=0,
        timeout_seconds=30,
    )

    try:
        pair_ranked, pair_profile = pair_scheduler.submit(
            query=query,
            candidates=candidates,
            top_k=4,
        )

        token_ranked, token_profile = token_scheduler.submit(
            query=query,
            candidates=candidates,
            top_k=4,
        )

        pair_ids = [
            item["document_id"]
            for item in pair_ranked
        ]

        token_ids = [
            item["document_id"]
            for item in token_ranked
        ]

        assert pair_ids == token_ids

        pair_scores = np.array([
            item["cross_encoder_score"]
            for item in pair_ranked
        ])

        token_scores = np.array([
            item["cross_encoder_score"]
            for item in token_ranked
        ])

        np.testing.assert_allclose(
            pair_scores,
            token_scores,
            rtol=1e-4,
            atol=1e-4,
        )

        assert pair_profile["batch_pair_count"] == 4
        assert token_profile["batch_pair_count"] == 4

        assert pair_profile["batch_token_count"] is None
        assert token_profile["batch_token_count"] > 0

        assert (
            0 < token_profile["batch_token_utilization"] <= 1
        )

    finally:
        pair_scheduler.close()
        token_scheduler.close()
