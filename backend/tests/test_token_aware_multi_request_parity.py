"""Multi-request ranking parity for token-aware scheduling."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from backend.api.services.dynamic_batch import DynamicBatchReranker


REQUESTS = [
    (
        "What is dynamic batching?",
        [
            {
                "document_id": 100 + i,
                "content": text,
            }
            for i, text in enumerate([
                "Dynamic batching combines inference requests.",
                "Batch inference improves accelerator utilization.",
                "Database indexing improves SQL performance.",
                "Neural network serving requires scheduling.",
            ])
        ],
    ),
    (
        "What is vector retrieval?",
        [
            {
                "document_id": 200 + i,
                "content": text,
            }
            for i, text in enumerate([
                "Vector retrieval uses embedding similarity.",
                "Approximate nearest neighbor search scales retrieval.",
                "A relational database stores structured records.",
                "Dense retrieval represents text using embeddings.",
            ])
        ],
    ),
    (
        "What is quantization?",
        [
            {
                "document_id": 300 + i,
                "content": text,
            }
            for i, text in enumerate([
                "Quantization reduces numerical precision.",
                "Product quantization compresses embedding vectors.",
                "Gradient descent optimizes neural networks.",
                "Vector compression reduces memory consumption.",
            ])
        ],
    ),
    (
        "What is a cross-encoder?",
        [
            {
                "document_id": 400 + i,
                "content": text,
            }
            for i, text in enumerate([
                "Cross-encoders jointly encode query-document pairs.",
                "Reranking improves retrieval relevance.",
                "A tokenizer converts text into token IDs.",
                "Cross-encoders score candidate passages.",
            ])
        ],
    ),
]


def run_policy(policy):
    scheduler = DynamicBatchReranker(
        batch_policy=policy,
        max_requests=4,
        max_pairs=16,
        max_tokens=100,
        max_wait_ms=20,
        timeout_seconds=60,
    )

    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(
                    scheduler.submit,
                    query,
                    candidates,
                    4,
                )
                for query, candidates in REQUESTS
            ]

            return [
                future.result(timeout=60)
                for future in futures
            ]
    finally:
        scheduler.close()


def test_multi_request_ranking_parity():
    pair_results = run_policy("pair")
    token_results = run_policy("token")

    assert len(pair_results) == len(token_results) == 4

    for (pair_ranked, _), (token_ranked, token_profile) in zip(
        pair_results,
        token_results,
    ):
        pair_ids = [
            item["document_id"]
            for item in pair_ranked
        ]

        token_ids = [
            item["document_id"]
            for item in token_ranked
        ]

        assert pair_ids == token_ids

        pair_scores = np.asarray([
            item["cross_encoder_score"]
            for item in pair_ranked
        ])

        token_scores = np.asarray([
            item["cross_encoder_score"]
            for item in token_ranked
        ])

        np.testing.assert_allclose(
            pair_scores,
            token_scores,
            rtol=1e-4,
            atol=1e-4,
        )

        assert token_profile["batch_token_count"] <= 100
        assert token_profile["batch_token_utilization"] <= 1
