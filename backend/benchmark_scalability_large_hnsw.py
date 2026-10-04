import os
import time
import statistics

import numpy as np
import psycopg
from sentence_transformers import SentenceTransformer


DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://retrieval:retrieval@127.0.0.1:5433/retrieval_db",
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TABLE_NAME = "scalability_large_documents"

TOP_K = 10
QUERY_COUNT = 100
WARMUP_RUNS = 10
EF_VALUES = [10, 20, 40, 80, 160]


def vector_to_pg(vector):
    return "[" + ",".join(map(str, vector.tolist())) + "]"


def percentile(values, p):
    return float(np.percentile(values, p))


def summarize(latencies):
    mean_ms = statistics.mean(latencies)
    return {
        "mean": mean_ms,
        "p50": percentile(latencies, 50),
        "p95": percentile(latencies, 95),
        "p99": percentile(latencies, 99),
        "qps": 1000.0 / mean_ms,
    }


def recall_at_k(exact_ids, ann_ids):
    return len(set(exact_ids) & set(ann_ids)) / TOP_K


def load_queries():
    # Disable automatic prepared statements for benchmark control.
    with psycopg.connect(
        DB_URL,
        prepare_threshold=None,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
            document_count = cur.fetchone()[0]

            cur.execute(
                f"""
                SELECT content
                FROM {TABLE_NAME}
                WHERE document_id %% 5000 = 1
                ORDER BY document_id
                LIMIT %s
                """,
                (QUERY_COUNT,),
            )

            texts = [row[0] for row in cur.fetchall()]

    if len(texts) != QUERY_COUNT:
        raise RuntimeError(
            f"Expected {QUERY_COUNT} queries, got {len(texts)}"
        )

    return document_count, texts


def benchmark_exact(query_vectors):
    sql = f"""
        SELECT document_id
        FROM {TABLE_NAME}
        ORDER BY embedding <=> %s::vector
        LIMIT {TOP_K}
    """

    results = []
    latencies = []

    # Dedicated exact connection.
    with psycopg.connect(
        DB_URL,
        prepare_threshold=None,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute("SET enable_indexscan = off")
            cur.execute("SET enable_bitmapscan = off")

            print("\nExact execution plan:")

            cur.execute(
                f"""
                EXPLAIN
                SELECT document_id
                FROM {TABLE_NAME}
                ORDER BY embedding <=> %s::vector
                LIMIT {TOP_K}
                """,
                (query_vectors[0],),
            )

            plan = "\n".join(row[0] for row in cur.fetchall())
            print(plan)

            if "Seq Scan" not in plan:
                raise RuntimeError(
                    "Exact benchmark is not using a sequential scan."
                )

            print("\nExact warmup...")

            for vector in query_vectors[:WARMUP_RUNS]:
                cur.execute(sql, (vector,))
                cur.fetchall()

            print("Computing exact ground truth...")

            for i, vector in enumerate(query_vectors, start=1):
                start = time.perf_counter()

                cur.execute(sql, (vector,))
                ids = [row[0] for row in cur.fetchall()]

                elapsed = (
                    time.perf_counter() - start
                ) * 1000.0

                results.append(ids)
                latencies.append(elapsed)

                if i % 20 == 0:
                    print(
                        f"Exact {i:3d}/{QUERY_COUNT} "
                        f"| latest={elapsed:.2f} ms"
                    )

    return results, summarize(latencies)


def benchmark_hnsw(query_vectors, exact_results):
    sql = f"""
        SELECT document_id
        FROM {TABLE_NAME}
        ORDER BY embedding <=> %s::vector
        LIMIT {TOP_K}
    """

    all_results = []

    # Completely separate ANN connection.
    with psycopg.connect(
        DB_URL,
        prepare_threshold=None,
    ) as conn:
        with conn.cursor() as cur:
            # Explicitly benchmark HNSW.
            cur.execute("SET enable_seqscan = off")

            print("\nHNSW execution plan:")

            cur.execute("SET hnsw.ef_search = 40")

            cur.execute(
                f"""
                EXPLAIN
                SELECT document_id
                FROM {TABLE_NAME}
                ORDER BY embedding <=> %s::vector
                LIMIT {TOP_K}
                """,
                (query_vectors[0],),
            )

            plan = "\n".join(row[0] for row in cur.fetchall())
            print(plan)

            if "idx_scalability_large_embedding_hnsw" not in plan:
                raise RuntimeError(
                    "HNSW benchmark is not using the HNSW index."
                )

            for ef in EF_VALUES:
                print(
                    f"\nBenchmarking HNSW "
                    f"ef_search={ef}..."
                )

                cur.execute(
                    f"SET hnsw.ef_search = {int(ef)}"
                )

                for vector in query_vectors[:WARMUP_RUNS]:
                    cur.execute(sql, (vector,))
                    cur.fetchall()

                latencies = []
                recalls = []

                for i, vector in enumerate(query_vectors):
                    start = time.perf_counter()

                    cur.execute(sql, (vector,))
                    ann_ids = [
                        row[0]
                        for row in cur.fetchall()
                    ]

                    elapsed = (
                        time.perf_counter() - start
                    ) * 1000.0

                    latencies.append(elapsed)

                    recalls.append(
                        recall_at_k(
                            exact_results[i],
                            ann_ids,
                        )
                    )

                stats = summarize(latencies)

                result = {
                    "ef": ef,
                    "recall": statistics.mean(recalls),
                    **stats,
                }

                all_results.append(result)

                print(
                    f"Recall@10={result['recall']:.4f} "
                    f"| P50={result['p50']:.3f} ms "
                    f"| P95={result['p95']:.3f} ms "
                    f"| QPS={result['qps']:.2f}"
                )

    return all_results


def main():
    print("=" * 75)
    print("500K HNSW Recall / Latency Benchmark")
    print("=" * 75)

    document_count, query_texts = load_queries()

    print(f"Documents: {document_count:,}")
    print(f"Queries:   {QUERY_COUNT}")
    print(f"Top-K:     {TOP_K}")
    print(f"ef_search: {EF_VALUES}")

    print("\nLoading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Encoding queries...")

    embeddings = model.encode(
        query_texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    query_vectors = [
        vector_to_pg(vector)
        for vector in embeddings
    ]

    exact_results, exact_stats = benchmark_exact(
        query_vectors
    )

    hnsw_results = benchmark_hnsw(
        query_vectors,
        exact_results,
    )

    print("\n" + "=" * 90)
    print("500K Benchmark Results")
    print("=" * 90)

    print(
        f"{'Method':<20}"
        f"{'Recall@10':>12}"
        f"{'Mean(ms)':>12}"
        f"{'P50(ms)':>12}"
        f"{'P95(ms)':>12}"
        f"{'P99(ms)':>12}"
        f"{'QPS':>12}"
    )

    print(
        f"{'Exact':<20}"
        f"{1.0:>12.4f}"
        f"{exact_stats['mean']:>12.3f}"
        f"{exact_stats['p50']:>12.3f}"
        f"{exact_stats['p95']:>12.3f}"
        f"{exact_stats['p99']:>12.3f}"
        f"{exact_stats['qps']:>12.2f}"
    )

    for result in hnsw_results:
        name = f"HNSW ef={result['ef']}"

        print(
            f"{name:<20}"
            f"{result['recall']:>12.4f}"
            f"{result['mean']:>12.3f}"
            f"{result['p50']:>12.3f}"
            f"{result['p95']:>12.3f}"
            f"{result['p99']:>12.3f}"
            f"{result['qps']:>12.2f}"
        )

    print("=" * 90)

    print("\nP50 speedup over exact:")

    for result in hnsw_results:
        speedup = (
            exact_stats["p50"] /
            result["p50"]
        )

        print(
            f"  ef={result['ef']:<3}: "
            f"{speedup:.2f}x"
        )


if __name__ == "__main__":
    main()
