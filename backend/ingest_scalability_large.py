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
TABLE_NAME = "scalability_large_documents"

BATCH_SIZE = 256
COMMIT_EVERY = 10_000


def vector_to_pg(vector):
    return "[" + ",".join(map(str, vector.tolist())) + "]"


def insert_batch(cur, rows):
    cur.executemany(
        f"""
        INSERT INTO {TABLE_NAME}
            (external_id, content, embedding)
        VALUES (%s, %s, %s::vector)
        """,
        rows,
    )


def main():
    if len(sys.argv) != 2:
        print(
            "Usage: python backend/ingest_scalability_large.py "
            "<document_count>"
        )
        sys.exit(1)

    target_count = int(sys.argv[1])

    if target_count <= 0:
        raise ValueError("document_count must be greater than 0")

    print("=" * 60)
    print("Large-Scale MS MARCO Ingestion")
    print("=" * 60)
    print(f"Target documents: {target_count:,}")
    print(f"Embedding model:  {MODEL_NAME}")
    print(f"Batch size:       {BATCH_SIZE}")
    print(f"Commit interval:  {COMMIT_EVERY:,}")
    print("=" * 60)

    print("\nLoading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Opening MS MARCO streaming dataset...")
    dataset = load_dataset(
        "sentence-transformers/msmarco-corpus",
        "passage",
        split="train",
        streaming=True,
    )

    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            print("\nClearing scalability_large_documents...")
            cur.execute(
                f"TRUNCATE TABLE {TABLE_NAME} RESTART IDENTITY"
            )
            conn.commit()

            batch_pids = []
            batch_texts = []

            inserted = 0
            embedding_seconds = 0.0
            insertion_seconds = 0.0
            total_start = time.perf_counter()

            for row in dataset:
                if inserted + len(batch_texts) >= target_count:
                    break

                text = row["text"]

                if not text:
                    continue

                batch_pids.append(row["pid"])
                batch_texts.append(text)

                if len(batch_texts) < BATCH_SIZE:
                    continue

                embed_start = time.perf_counter()

                embeddings = model.encode(
                    batch_texts,
                    batch_size=BATCH_SIZE,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )

                embedding_seconds += (
                    time.perf_counter() - embed_start
                )

                rows = [
                    (
                        f"msmarco-{pid}",
                        text,
                        vector_to_pg(embedding),
                    )
                    for pid, text, embedding in zip(
                        batch_pids,
                        batch_texts,
                        embeddings,
                    )
                ]

                insert_start = time.perf_counter()
                insert_batch(cur, rows)
                insertion_seconds += (
                    time.perf_counter() - insert_start
                )

                inserted += len(rows)

                batch_pids.clear()
                batch_texts.clear()

                if (
                    inserted % COMMIT_EVERY < BATCH_SIZE
                    or inserted >= target_count
                ):
                    conn.commit()

                if inserted % 10_000 < BATCH_SIZE:
                    elapsed = time.perf_counter() - total_start
                    throughput = inserted / elapsed

                    print(
                        f"Inserted {inserted:,}/{target_count:,} "
                        f"({inserted / target_count * 100:.1f}%) "
                        f"| {throughput:.1f} docs/s"
                    )

            # Final partial batch
            remaining = target_count - inserted

            if batch_texts and remaining > 0:
                batch_pids = batch_pids[:remaining]
                batch_texts = batch_texts[:remaining]

                embed_start = time.perf_counter()

                embeddings = model.encode(
                    batch_texts,
                    batch_size=BATCH_SIZE,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )

                embedding_seconds += (
                    time.perf_counter() - embed_start
                )

                rows = [
                    (
                        f"msmarco-{pid}",
                        text,
                        vector_to_pg(embedding),
                    )
                    for pid, text, embedding in zip(
                        batch_pids,
                        batch_texts,
                        embeddings,
                    )
                ]

                insert_start = time.perf_counter()
                insert_batch(cur, rows)
                insertion_seconds += (
                    time.perf_counter() - insert_start
                )

                inserted += len(rows)
                conn.commit()

            total_seconds = time.perf_counter() - total_start

            cur.execute(
                f"SELECT COUNT(*) FROM {TABLE_NAME}"
            )
            db_count = cur.fetchone()[0]

            cur.execute(
                """
                SELECT pg_size_pretty(
                    pg_total_relation_size(%s)
                )
                """,
                (TABLE_NAME,),
            )
            total_size = cur.fetchone()[0]

    print("\n" + "=" * 60)
    print("Ingestion Complete")
    print("=" * 60)
    print(f"Inserted rows:        {inserted:,}")
    print(f"Database rows:        {db_count:,}")
    print(f"Embedding time:       {embedding_seconds:.2f} s")
    print(
        f"Embedding throughput: "
        f"{inserted / embedding_seconds:.2f} docs/s"
    )
    print(f"Insert time:          {insertion_seconds:.2f} s")
    print(
        f"Insert throughput:    "
        f"{inserted / insertion_seconds:.2f} rows/s"
    )
    print(f"Total pipeline time:  {total_seconds:.2f} s")
    print(
        f"Overall throughput:   "
        f"{inserted / total_seconds:.2f} docs/s"
    )
    print(f"Total relation size:  {total_size}")
    print("=" * 60)


if __name__ == "__main__":
    main()
