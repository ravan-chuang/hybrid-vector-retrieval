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


def vector_to_pg(vector):
    return "[" + ",".join(map(str, vector.tolist())) + "]"


def percentile(values, p):
    return float(np.percentile(values, p))


def main():
    print("=" * 65)
    print("500K Exact Vector Search Benchmark")
    print("=" * 65)

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*) FROM {TABLE_NAME}"
            )
            document_count = cur.fetchone()[0]

            print(f"Documents: {document_count:,}")
            print(f"Queries:   {QUERY_COUNT}")
            print(f"Top-K:     {TOP_K}")
            print(f"Warmups:   {WARMUP_RUNS}")

            # Deterministic query sample spread across the corpus.
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

            query_texts = [row[0] for row in cur.fetchall()]

            if len(query_texts) != QUERY_COUNT:
                raise RuntimeError(
                    f"Expected {QUERY_COUNT} queries, "
                    f"got {len(query_texts)}"
                )

            print("\nEncoding query set...")

            query_embeddings = model.encode(
                query_texts,
                batch_size=64,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            query_vectors = [
                vector_to_pg(vector)
                for vector in query_embeddings
            ]

            # Force exact brute-force execution.
            cur.execute("SET enable_indexscan = off")
            cur.execute("SET enable_bitmapscan = off")

            sql = f"""
                SELECT document_id
                FROM {TABLE_NAME}
                ORDER BY embedding <=> %s::vector
                LIMIT {TOP_K}
            """

            print("Running warmups...")

            for vector in query_vectors[:WARMUP_RUNS]:
                cur.execute(sql, (vector,))
                cur.fetchall()

            print("Running measured queries...")

            latencies = []

            for i, vector in enumerate(query_vectors, start=1):
                start = time.perf_counter()

                cur.execute(sql, (vector,))
                cur.fetchall()

                latency_ms = (
                    time.perf_counter() - start
                ) * 1000.0

                latencies.append(latency_ms)

                if i % 10 == 0:
                    print(
                        f"Completed {i:3d}/{QUERY_COUNT} "
                        f"| latest={latency_ms:.2f} ms"
                    )

    mean_ms = statistics.mean(latencies)
    p50_ms = percentile(latencies, 50)
    p95_ms = percentile(latencies, 95)
    p99_ms = percentile(latencies, 99)
    qps = 1000.0 / mean_ms

    print("\n" + "=" * 65)
    print("Exact Search Results")
    print("=" * 65)
    print(f"Documents:   {document_count:,}")
    print(f"Queries:     {QUERY_COUNT}")
    print(f"Top-K:       {TOP_K}")
    print(f"Mean:        {mean_ms:.3f} ms")
    print(f"P50:         {p50_ms:.3f} ms")
    print(f"P95:         {p95_ms:.3f} ms")
    print(f"P99:         {p99_ms:.3f} ms")
    print(f"Approx QPS:  {qps:.2f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
