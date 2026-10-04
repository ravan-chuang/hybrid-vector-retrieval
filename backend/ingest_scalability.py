import os
import sys
import time
import psycopg
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://retrieval:retrieval@127.0.0.1:5433/retrieval_db",
)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE = 256


def to_vector_string(vector):
    return "[" + ",".join(str(float(x)) for x in vector) + "]"


def main():
    if len(sys.argv) != 2:
        print(
            "Usage: python backend/ingest_scalability.py "
            "<document_count>"
        )
        sys.exit(1)

    target_count = int(sys.argv[1])

    if target_count > 120000:
        print(
            "AG News supports this script only up to "
            "120,000 documents."
        )
        sys.exit(1)

    print("=" * 72)
    print("Scalability Corpus Ingestion")
    print("=" * 72)
    print(f"Target documents: {target_count:,}")
    print(f"Model: {MODEL_NAME}")
    print(f"Batch size: {BATCH_SIZE}")

    print("\nLoading AG News...")

    dataset = load_dataset(
        "fancyzhx/ag_news",
        split=f"train[:{target_count}]"
    )

    texts = dataset["text"]

    print(f"Loaded: {len(texts):,} documents")

    print("\nLoading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    print("\nGenerating embeddings...")

    embedding_start = time.perf_counter()

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embedding_time = (
        time.perf_counter() - embedding_start
    )

    print(
        f"\nEmbedding time: "
        f"{embedding_time:.2f} s"
    )

    print(
        f"Embedding throughput: "
        f"{len(texts) / embedding_time:.2f} docs/s"
    )

    print("\nInserting into PostgreSQL...")

    insert_start = time.perf_counter()

    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:

            cur.execute(
                "TRUNCATE TABLE scalability_documents "
                "RESTART IDENTITY"
            )

            for start in range(
                0,
                len(texts),
                BATCH_SIZE
            ):
                end = min(
                    start + BATCH_SIZE,
                    len(texts)
                )

                rows = []

                for i in range(start, end):
                    rows.append(
                        (
                            f"agnews-{i}",
                            texts[i],
                            to_vector_string(
                                embeddings[i]
                            ),
                        )
                    )

                cur.executemany(
                    """
                    INSERT INTO scalability_documents
                        (
                            external_id,
                            content,
                            embedding
                        )
                    VALUES
                        (
                            %s,
                            %s,
                            %s::vector
                        )
                    """,
                    rows
                )

                if (
                    end % 10000 == 0
                    or end == len(texts)
                ):
                    print(
                        f"Inserted "
                        f"{end:,}/{len(texts):,}"
                    )

        conn.commit()

    insert_time = (
        time.perf_counter() - insert_start
    )

    print(
        f"\nInsert time: "
        f"{insert_time:.2f} s"
    )

    print(
        f"Insert throughput: "
        f"{len(texts) / insert_time:.2f} rows/s"
    )

    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:

            cur.execute(
                """
                SELECT COUNT(*)
                FROM scalability_documents
                """
            )

            count = cur.fetchone()[0]

            cur.execute(
                """
                SELECT pg_size_pretty(
                    pg_total_relation_size(
                        'scalability_documents'
                    )
                )
                """
            )

            size = cur.fetchone()[0]

    print("\n" + "=" * 72)
    print(f"Rows:       {count:,}")
    print(f"Table size: {size}")
    print("=" * 72)


if __name__ == "__main__":
    main()
