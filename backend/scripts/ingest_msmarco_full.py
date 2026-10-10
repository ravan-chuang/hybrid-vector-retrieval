import argparse
import os
import signal
import sys
import time

import psycopg
from datasets import load_dataset
from psycopg.rows import tuple_row


DATASET_NAME = "sentence-transformers/msmarco-corpus"
DATASET_CONFIG = "passage"
DATASET_SPLIT = "train"
TABLE_NAME = "msmarco_full_documents"

stop_requested = False


def handle_signal(signum, frame):
    global stop_requested
    if not stop_requested:
        print("\nStop requested. Finishing current batch safely...")
        stop_requested = True
    else:
        print("\nForced termination.")
        sys.exit(1)


signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


def get_database_url():
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Run: "
            "set -a; source .env; set +a"
        )
    return url


def get_progress(conn):
    with conn.cursor(row_factory=tuple_row) as cur:
        cur.execute(
            f"""
            SELECT
                COUNT(*) AS row_count,
                COALESCE(MAX(document_id), 0) AS max_document_id
            FROM {TABLE_NAME}
            """
        )
        row_count, max_document_id = cur.fetchone()

    return int(row_count), int(max_document_id)


def insert_batch(conn, batch):
    if not batch:
        return 0

    with conn.cursor() as cur:
        cur.executemany(
            f"""
            INSERT INTO {TABLE_NAME}
                (document_id, external_id, content)
            VALUES (%s, %s, %s)
            ON CONFLICT (document_id) DO NOTHING
            """,
            batch,
        )

    conn.commit()
    return len(batch)


def main():
    parser = argparse.ArgumentParser(
        description="Resumable full MS MARCO passage ingestion"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100_000,
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Optional limit for smoke testing",
    )

    args = parser.parse_args()

    database_url = get_database_url()

    print("Loading streaming MS MARCO corpus...")

    dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split=DATASET_SPLIT,
        streaming=True,
    )

    with psycopg.connect(database_url) as conn:
        existing_rows, max_document_id = get_progress(conn)

        # document_id = pid + 1
        resume_pid = max_document_id

        print(f"Existing rows : {existing_rows:,}")
        print(f"Max document  : {max_document_id:,}")
        print(f"Resume pid    : {resume_pid:,}")
        print(f"Batch size    : {args.batch_size:,}")

        batch = []
        processed = 0
        inserted_session = 0
        start_time = time.perf_counter()
        last_log = 0

        for row in dataset:
            if stop_requested:
                break

            pid = int(row["pid"])

            # Existing document_id = pid + 1.
            # Therefore max(document_id) corresponds to the next pid
            # that has not yet been inserted for a contiguous corpus.
            if pid < resume_pid:
                continue

            text = row["text"]

            batch.append(
                (
                    pid + 1,
                    f"msmarco-{pid}",
                    text,
                )
            )

            processed += 1

            if len(batch) >= args.batch_size:
                insert_batch(conn, batch)
                inserted_session += len(batch)
                batch.clear()

            if processed - last_log >= args.log_every:
                elapsed = time.perf_counter() - start_time
                rate = processed / elapsed if elapsed > 0 else 0

                print(
                    f"Processed {processed:,} | "
                    f"session inserts {inserted_session:,} | "
                    f"{rate:,.0f} passages/s"
                )

                last_log = processed

            if args.max_docs is not None and processed >= args.max_docs:
                print("Reached --max-docs limit.")
                break

        if batch:
            insert_batch(conn, batch)
            inserted_session += len(batch)

        elapsed = time.perf_counter() - start_time

        final_rows, final_max_document_id = get_progress(conn)

        print()
        print("Ingestion finished")
        print("------------------")
        print(f"Session processed : {processed:,}")
        print(f"Session inserted  : {inserted_session:,}")
        print(f"Database rows     : {final_rows:,}")
        print(f"Max document_id   : {final_max_document_id:,}")
        print(f"Elapsed           : {elapsed:,.1f} s")

        if elapsed > 0:
            print(f"Throughput        : {processed / elapsed:,.0f} passages/s")


if __name__ == "__main__":
    main()
