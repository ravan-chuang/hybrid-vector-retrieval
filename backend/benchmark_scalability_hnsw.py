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
QUERY_COUNT = 100
WARMUP = 10

EF_VALUES = [10, 20, 40, 80, 160]


def to_vector_string(vector):
    return "[" + ",".join(str(float(x)) for x in vector) + "]"


def percentile(values, p):
    values = sorted(values)

    pos = (len(values) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)

    return (
        values[lo] * (hi - pos)
        + values[hi] * (pos - lo)
        if hi != lo
        else values[lo]
    )


def stats(values):
    mean = statistics.mean(values)

    return {
        "mean": mean,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "qps": 1000.0 / mean,
    }


SQL = """
SELECT document_id
FROM scalability_documents
ORDER BY embedding <=> %s::vector
LIMIT %s
"""


print("=" * 80)
print("100K HNSW Recall-Latency Benchmark")
print("=" * 80)

model = SentenceTransformer(MODEL_NAME)

# Deterministic sample spread across the corpus
with psycopg.connect(DB_URL) as conn:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT content
            FROM scalability_documents
            WHERE document_id %% 1000 = 1
            ORDER BY document_id
            LIMIT %s
            """,
            (QUERY_COUNT,)
        )

        query_texts = [
            row[0]
            for row in cur.fetchall()
        ]

print(f"Unique queries: {len(query_texts)}")

query_embeddings = model.encode(
    query_texts,
    batch_size=64,
    normalize_embeddings=True,
    show_progress_bar=True
)

vectors = [
    to_vector_string(v)
    for v in query_embeddings
]

with psycopg.connect(DB_URL) as conn:
    with conn.cursor() as cur:

        # -----------------------------------
        # Exact ground truth
        # -----------------------------------

        print("\nGenerating exact ground truth...")

        cur.execute("SET enable_indexscan = off")
        cur.execute("SET enable_bitmapscan = off")

        exact_results = []
        exact_latencies = []

        for vector in vectors[:WARMUP]:
            cur.execute(SQL, (vector, TOP_K))
            cur.fetchall()

        for i, vector in enumerate(vectors, start=1):

            start = time.perf_counter()

            cur.execute(SQL, (vector, TOP_K))
            rows = cur.fetchall()

            elapsed = (
                time.perf_counter() - start
            ) * 1000

            exact_latencies.append(elapsed)

            exact_results.append(
                [row[0] for row in rows]
            )

            if i % 20 == 0:
                print(
                    f"Exact ground truth: "
                    f"{i}/{QUERY_COUNT}"
                )

        exact_stats = stats(exact_latencies)

        cur.execute("SET enable_indexscan = on")
        cur.execute("SET enable_bitmapscan = on")

        # -----------------------------------
        # HNSW
        # -----------------------------------

        hnsw_results = []

        for ef in EF_VALUES:

            print(f"\nTesting HNSW ef_search={ef}")

            cur.execute(
                f"SET hnsw.ef_search = {ef}"
            )

            for vector in vectors[:WARMUP]:
                cur.execute(SQL, (vector, TOP_K))
                cur.fetchall()

            latencies = []
            recalls = []

            for i, vector in enumerate(vectors):

                start = time.perf_counter()

                cur.execute(
                    SQL,
                    (vector, TOP_K)
                )

                rows = cur.fetchall()

                elapsed = (
                    time.perf_counter() - start
                ) * 1000

                latencies.append(elapsed)

                ann = {
                    row[0]
                    for row in rows
                }

                exact = set(
                    exact_results[i]
                )

                recalls.append(
                    len(ann & exact) / TOP_K
                )

            hnsw_results.append(
                (
                    ef,
                    statistics.mean(recalls),
                    stats(latencies)
                )
            )


print("\n" + "=" * 94)
print("Results")
print("=" * 94)

print(
    f"{'Method':<20}"
    f"{'Recall@10':>12}"
    f"{'Mean(ms)':>12}"
    f"{'P50(ms)':>12}"
    f"{'P95(ms)':>12}"
    f"{'P99(ms)':>12}"
    f"{'QPS':>12}"
)

print("-" * 94)

print(
    f"{'Exact':<20}"
    f"{1.0:>12.4f}"
    f"{exact_stats['mean']:>12.3f}"
    f"{exact_stats['p50']:>12.3f}"
    f"{exact_stats['p95']:>12.3f}"
    f"{exact_stats['p99']:>12.3f}"
    f"{exact_stats['qps']:>12.2f}"
)

for ef, recall, s in hnsw_results:

    print(
        f"{('HNSW ef=' + str(ef)):<20}"
        f"{recall:>12.4f}"
        f"{s['mean']:>12.3f}"
        f"{s['p50']:>12.3f}"
        f"{s['p95']:>12.3f}"
        f"{s['p99']:>12.3f}"
        f"{s['qps']:>12.2f}"
    )

print("=" * 94)
