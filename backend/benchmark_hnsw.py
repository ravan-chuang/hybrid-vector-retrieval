import os
import time
import statistics
import psycopg
from sentence_transformers import SentenceTransformer

DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://retrieval:retrieval@127.0.0.1:5433/retrieval_db",
)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 10
WARMUP_RUNS = 5
MEASURE_RUNS = 50

EF_VALUES = [10, 20, 40, 80, 160]

QUERIES = [
    "stock market and financial news",
    "technology companies and software",
    "international politics and government",
    "sports teams and competitions",
    "business mergers and acquisitions",
    "scientific research and discoveries",
    "computer technology and internet",
    "economic growth and employment",
    "football and professional sports",
    "global conflicts and diplomacy",
]


def to_vector_string(vector):
    return "[" + ",".join(str(float(x)) for x in vector) + "]"


def percentile(values, p):
    values = sorted(values)
    index = (len(values) - 1) * p
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    fraction = index - lower

    return (
        values[lower] * (1 - fraction)
        + values[upper] * fraction
    )


def summarize(latencies):
    mean = statistics.mean(latencies)

    return {
        "mean": mean,
        "p50": percentile(latencies, 0.50),
        "p95": percentile(latencies, 0.95),
        "p99": percentile(latencies, 0.99),
        "qps": 1000.0 / mean,
    }


SQL = """
    SELECT benchmark_id
    FROM benchmark_documents
    WHERE embedding IS NOT NULL
    ORDER BY embedding <=> %s::vector
    LIMIT %s
"""


print("=" * 75)
print("HNSW Recall-Latency Benchmark")
print("=" * 75)

model = SentenceTransformer(MODEL_NAME)

print("Encoding queries...")

embeddings = model.encode(
    QUERIES,
    normalize_embeddings=True
)

vectors = [
    to_vector_string(v)
    for v in embeddings
]


with psycopg.connect(DB_URL) as conn:
    with conn.cursor() as cur:

        # --------------------------------------------------
        # Exact ground truth
        # --------------------------------------------------

        print("\nGenerating exact ground truth...")

        cur.execute("SET enable_indexscan = off")
        cur.execute("SET enable_bitmapscan = off")

        exact_results = []
        exact_latencies = []

        # Warmup
        for i in range(WARMUP_RUNS):
            vector = vectors[i % len(vectors)]

            cur.execute(SQL, (vector, TOP_K))
            cur.fetchall()

        # Measurement
        for i in range(MEASURE_RUNS):
            vector = vectors[i % len(vectors)]

            start = time.perf_counter()

            cur.execute(SQL, (vector, TOP_K))
            rows = cur.fetchall()

            elapsed = (
                time.perf_counter() - start
            ) * 1000

            exact_latencies.append(elapsed)

            if i < len(vectors):
                exact_results.append(
                    [row[0] for row in rows]
                )

        exact_stats = summarize(exact_latencies)

        # Restore ANN index usage
        cur.execute("SET enable_indexscan = on")
        cur.execute("SET enable_bitmapscan = on")

        print(
            f"Exact P50: {exact_stats['p50']:.3f} ms | "
            f"P95: {exact_stats['p95']:.3f} ms"
        )

        # --------------------------------------------------
        # HNSW
        # --------------------------------------------------

        results_table = []

        for ef in EF_VALUES:

            print(f"\nTesting HNSW ef_search={ef}...")

            cur.execute(
                f"SET hnsw.ef_search = {ef}"
            )

            # Warmup
            for i in range(WARMUP_RUNS):
                vector = vectors[i % len(vectors)]

                cur.execute(SQL, (vector, TOP_K))
                cur.fetchall()

            latencies = []
            recall_scores = []

            for i in range(MEASURE_RUNS):
                query_index = i % len(vectors)
                vector = vectors[query_index]

                start = time.perf_counter()

                cur.execute(SQL, (vector, TOP_K))
                rows = cur.fetchall()

                elapsed = (
                    time.perf_counter() - start
                ) * 1000

                latencies.append(elapsed)

                ann_ids = {
                    row[0]
                    for row in rows
                }

                exact_ids = set(
                    exact_results[query_index]
                )

                recall = (
                    len(ann_ids & exact_ids)
                    / TOP_K
                )

                recall_scores.append(recall)

            stats = summarize(latencies)

            avg_recall = statistics.mean(
                recall_scores
            )

            results_table.append(
                (
                    ef,
                    avg_recall,
                    stats
                )
            )


print("\n")
print("=" * 75)
print("Benchmark Results")
print("=" * 75)

print(
    f"{'Method':<15}"
    f"{'Recall@10':>12}"
    f"{'Mean(ms)':>12}"
    f"{'P50(ms)':>12}"
    f"{'P95(ms)':>12}"
    f"{'P99(ms)':>12}"
    f"{'QPS':>12}"
)

print("-" * 87)

print(
    f"{'Exact':<15}"
    f"{1.0:>12.4f}"
    f"{exact_stats['mean']:>12.3f}"
    f"{exact_stats['p50']:>12.3f}"
    f"{exact_stats['p95']:>12.3f}"
    f"{exact_stats['p99']:>12.3f}"
    f"{exact_stats['qps']:>12.2f}"
)

for ef, recall, stats in results_table:
    print(
        f"{('HNSW ef=' + str(ef)):<15}"
        f"{recall:>12.4f}"
        f"{stats['mean']:>12.3f}"
        f"{stats['p50']:>12.3f}"
        f"{stats['p95']:>12.3f}"
        f"{stats['p99']:>12.3f}"
        f"{stats['qps']:>12.2f}"
    )

print("=" * 87)
