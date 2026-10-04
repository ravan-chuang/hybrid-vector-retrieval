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


print("=" * 65)
print("Exact Vector Search Benchmark")
print("=" * 65)

model = SentenceTransformer(MODEL_NAME)

print("Encoding benchmark queries...")

query_embeddings = model.encode(
    QUERIES,
    normalize_embeddings=True
)

query_vectors = [
    to_vector_string(vector)
    for vector in query_embeddings
]

sql = """
    SELECT benchmark_id
    FROM benchmark_documents
    WHERE embedding IS NOT NULL
    ORDER BY embedding <=> %s::vector
    LIMIT %s
"""

latencies = []

with psycopg.connect(DB_URL) as conn:
    with conn.cursor() as cur:

        print(f"Warmup runs: {WARMUP_RUNS}")

        for i in range(WARMUP_RUNS):
            qv = query_vectors[i % len(query_vectors)]
            cur.execute(sql, (qv, TOP_K))
            cur.fetchall()

        print(f"Measured runs: {MEASURE_RUNS}")

        for i in range(MEASURE_RUNS):
            qv = query_vectors[i % len(query_vectors)]

            start = time.perf_counter()

            cur.execute(sql, (qv, TOP_K))
            results = cur.fetchall()

            elapsed_ms = (
                time.perf_counter() - start
            ) * 1000

            latencies.append(elapsed_ms)

            if len(results) != TOP_K:
                raise RuntimeError(
                    f"Expected {TOP_K} results, "
                    f"got {len(results)}"
                )


p50 = percentile(latencies, 0.50)
p95 = percentile(latencies, 0.95)
p99 = percentile(latencies, 0.99)

mean_latency = statistics.mean(latencies)

qps = 1000.0 / mean_latency


print("\n" + "=" * 65)
print("Exact Search Results")
print("=" * 65)

print(f"Corpus size:       10,000")
print(f"Vector dimension:  384")
print(f"Distance:          cosine (<=>)")
print(f"Top-K:             {TOP_K}")
print(f"Measured queries:  {MEASURE_RUNS}")

print("-" * 65)

print(f"Mean latency:      {mean_latency:.3f} ms")
print(f"P50 latency:       {p50:.3f} ms")
print(f"P95 latency:       {p95:.3f} ms")
print(f"P99 latency:       {p99:.3f} ms")
print(f"Approx. QPS:       {qps:.2f}")

print("=" * 65)
