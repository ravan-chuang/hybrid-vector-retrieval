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
WARMUP_RUNS = 10
MEASURE_RUNS = 100

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
    "Microsoft software products",
    "oil prices and energy markets",
    "airlines and international travel",
    "mobile phones and wireless technology",
    "banking and financial services",
    "presidential election campaign",
    "baseball teams and players",
    "computer security vulnerabilities",
    "economic policy and interest rates",
    "international trade agreements",
]


def to_vector_string(vector):
    return "[" + ",".join(str(float(x)) for x in vector) + "]"


def percentile(values, p):
    values = sorted(values)

    pos = (len(values) - 1) * p
    lower = int(pos)
    upper = min(lower + 1, len(values) - 1)
    fraction = pos - lower

    return (
        values[lower] * (1 - fraction)
        + values[upper] * fraction
    )


print("=" * 72)
print("100K Exact Vector Search Benchmark")
print("=" * 72)

model = SentenceTransformer(MODEL_NAME)

print("Encoding benchmark queries...")

embeddings = model.encode(
    QUERIES,
    normalize_embeddings=True
)

vectors = [
    to_vector_string(v)
    for v in embeddings
]

SQL = """
    SELECT document_id
    FROM scalability_documents
    WHERE embedding IS NOT NULL
    ORDER BY embedding <=> %s::vector
    LIMIT %s
"""

latencies = []

with psycopg.connect(DB_URL) as conn:
    with conn.cursor() as cur:

        # Force exact scan even if an ANN index is added later.
        cur.execute("SET enable_indexscan = off")
        cur.execute("SET enable_bitmapscan = off")

        print(
            f"Warmup: {WARMUP_RUNS} queries"
        )

        for i in range(WARMUP_RUNS):
            cur.execute(
                SQL,
                (
                    vectors[i % len(vectors)],
                    TOP_K
                )
            )
            cur.fetchall()

        print(
            f"Measured runs: {MEASURE_RUNS}"
        )

        for i in range(MEASURE_RUNS):
            vector = vectors[i % len(vectors)]

            start = time.perf_counter()

            cur.execute(
                SQL,
                (vector, TOP_K)
            )

            cur.fetchall()

            latency_ms = (
                time.perf_counter() - start
            ) * 1000

            latencies.append(latency_ms)

mean = statistics.mean(latencies)
p50 = percentile(latencies, 0.50)
p95 = percentile(latencies, 0.95)
p99 = percentile(latencies, 0.99)
qps = 1000.0 / mean

print("\n" + "=" * 72)
print("Results")
print("=" * 72)

print(f"Documents:  100,000")
print(f"Top-K:      {TOP_K}")
print(f"Runs:       {MEASURE_RUNS}")
print(f"Mean:       {mean:.3f} ms")
print(f"P50:        {p50:.3f} ms")
print(f"P95:        {p95:.3f} ms")
print(f"P99:        {p99:.3f} ms")
print(f"Approx QPS: {qps:.2f}")

print("=" * 72)
