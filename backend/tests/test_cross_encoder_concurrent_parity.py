import math
import threading
from concurrent.futures import ThreadPoolExecutor

from backend.api.services.cross_encoder import (
    rerank_candidates_profiled,
)
from backend.api.services.dynamic_batch import DynamicBatchReranker


QUERIES = [
    "what is information retrieval",
    "how does machine learning work",
    "what causes climate change",
    "how do database indexes work",
]

DOCUMENTS = [
    "Information retrieval finds relevant documents.",
    "Machine learning discovers patterns from data.",
    "Greenhouse gases contribute to climate change.",
    "Database indexes accelerate query execution.",
    "Neural networks contain interconnected layers.",
    "Search engines rank documents by relevance.",
    "Carbon dioxide contributes to global warming.",
    "Supervised learning uses labeled examples.",
    "B-tree indexes support efficient database lookup.",
    "Photosynthesis converts sunlight into energy.",
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


def test_real_cross_encoder_concurrent_parity():
    # First establish the reference ranking.
    baselines = {}

    for query in QUERIES:
        ranked, _ = rerank_candidates_profiled(
            query=query,
            candidates=make_candidates(),
            top_k=10,
        )
        baselines[query] = ranked

    scheduler = DynamicBatchReranker(
        max_requests=4,
        max_pairs=100,
        max_wait_ms=200,
        timeout_seconds=60,
    )

    barrier = threading.Barrier(4)

    def submit(query):
        barrier.wait(timeout=10)
        return scheduler.submit(
            query=query,
            candidates=make_candidates(),
            top_k=10,
        )

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            outputs = list(pool.map(submit, QUERIES))
    finally:
        scheduler.close()

    for query, (batched, profile) in zip(QUERIES, outputs):
        baseline = baselines[query]

        baseline_ids = [
            row["document_id"] for row in baseline
        ]
        batched_ids = [
            row["document_id"] for row in batched
        ]

        assert baseline_ids == batched_ids, (
            f"Ranking mismatch: {query}"
        )

        for left, right in zip(baseline, batched):
            assert math.isclose(
                left["cross_encoder_score"],
                right["cross_encoder_score"],
                rel_tol=1e-4,
                abs_tol=1e-4,
            ), f"Score mismatch: {query}"

        assert profile["batch_request_count"] == 4
        assert profile["batch_pair_count"] == 40

        print(
            f"PASS | query={query} | "
            f"batch_requests={profile['batch_request_count']} | "
            f"batch_pairs={profile['batch_pair_count']}"
        )
