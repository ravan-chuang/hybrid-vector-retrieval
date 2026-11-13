import math

from backend.api.services.cross_encoder import (
    rerank_candidates_profiled,
)
from backend.api.services.dynamic_batch import (
    DynamicBatchReranker,
)


QUERIES = [
    "what is information retrieval",
    "how does machine learning work",
    "what causes climate change",
]


DOCUMENTS = [
    "Information retrieval is the process of finding relevant information.",
    "Machine learning allows computers to learn patterns from data.",
    "Climate change is driven by greenhouse gas emissions.",
    "Search engines retrieve and rank documents for user queries.",
    "Neural networks are computational models inspired by biological neurons.",
    "Photosynthesis converts sunlight into chemical energy.",
    "Information retrieval systems use ranking algorithms.",
    "Supervised learning uses labeled training examples.",
    "Carbon dioxide contributes to global warming.",
    "Database indexes accelerate data retrieval.",
]


def make_candidates():
    return [
        {
            "document_id": i + 1,
            "external_id": f"doc-{i + 1}",
            "content": document,
            "lexical_rank": i + 1,
            "vector_rank": None,
        }
        for i, document in enumerate(DOCUMENTS)
    ]


def test_cross_encoder_ranking_parity():
    scheduler = DynamicBatchReranker(
        max_requests=4,
        max_pairs=100,
        max_wait_ms=5,
        timeout_seconds=30,
    )

    try:
        for query in QUERIES:
            candidates = make_candidates()

            baseline, _ = rerank_candidates_profiled(
                query=query,
                candidates=candidates,
                top_k=10,
            )

            batched, _ = scheduler.submit(
                query=query,
                candidates=candidates,
                top_k=10,
            )

            baseline_ids = [
                item["document_id"]
                for item in baseline
            ]

            batched_ids = [
                item["document_id"]
                for item in batched
            ]

            assert baseline_ids == batched_ids, (
                f"Ranking mismatch for query: {query}"
            )

            for left, right in zip(baseline, batched):
                assert math.isclose(
                    left["cross_encoder_score"],
                    right["cross_encoder_score"],
                    rel_tol=1e-5,
                    abs_tol=1e-5,
                ), (
                    f"Score mismatch for query: {query}"
                )

    finally:
        scheduler.close()
