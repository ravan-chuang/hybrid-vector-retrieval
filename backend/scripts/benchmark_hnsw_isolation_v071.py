import json
import os
import re
import statistics
import time

import psycopg
from sentence_transformers import SentenceTransformer


QUERY = "what is information retrieval"
EF_SEARCH = 160
TOP_K = 50
REPEATS = 10

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    vector = model.encode(
        QUERY,
        normalize_embeddings=True,
    ).tolist()

    # pgvector accepts its standard textual vector representation.
    vector_literal = "[" + ",".join(
        str(float(value)) for value in vector
    ) + "]"

    sql = """
        SELECT
            document_id,
            external_id,
            content
        FROM public.msmarco_full_documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """

    rows = []

    with psycopg.connect(
        os.environ["DATABASE_URL"]
    ) as conn:
        with conn.cursor() as cur:

            for i in range(REPEATS):
                cur.execute(
                    "SET LOCAL enable_seqscan = off"
                )

                cur.execute(
                    f"SET LOCAL hnsw.ef_search = {EF_SEARCH}"
                )

                start = time.perf_counter()

                cur.execute(
                    sql,
                    (vector_literal, TOP_K),
                )

                result = cur.fetchall()

                wall_ms = (
                    time.perf_counter() - start
                ) * 1000

                cur.execute(
                    """
                    EXPLAIN (
                        ANALYZE,
                        BUFFERS,
                        FORMAT JSON
                    )
                    SELECT
                        document_id,
                        external_id,
                        content
                    FROM public.msmarco_full_documents
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (vector_literal, TOP_K),
                )

                explain = cur.fetchone()[0]

                if isinstance(explain, str):
                    explain = json.loads(explain)

                plan = explain[0]

                execution_ms = plan["Execution Time"]

                rows.append({
                    "repeat": i,
                    "wall_ms": wall_ms,
                    "execution_ms": execution_ms,
                    "result_count": len(result),
                    "plan": plan["Plan"],
                })

                print(
                    f"[{i + 1:02d}/{REPEATS}] "
                    f"Wall={wall_ms:9.3f} ms | "
                    f"PG={execution_ms:9.3f} ms | "
                    f"Results={len(result)}"
                )

    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)

    for key in ["wall_ms", "execution_ms"]:
        values = [row[key] for row in rows]

        print(
            f"{key:20s}"
            f"mean={statistics.mean(values):9.3f} ms "
            f"median={statistics.median(values):9.3f} ms "
            f"max={max(values):9.3f} ms"
        )

    print("\nExecution Plan (last run):")
    print(json.dumps(rows[-1]["plan"], indent=2))

    output = (
        "results/msmarco_full/latency/"
        "hnsw_isolation_v071.json"
    )

    os.makedirs(
        os.path.dirname(output),
        exist_ok=True,
    )

    with open(output, "w") as f:
        json.dump(rows, f, indent=2)

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
