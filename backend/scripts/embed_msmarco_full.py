import argparse
import os
import signal
import sys
import time

import numpy as np
import psycopg
import torch
from sentence_transformers import SentenceTransformer


TABLE = "msmarco_full_documents"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

stop_requested = False


def handle_signal(signum, frame):
    global stop_requested
    if not stop_requested:
        print("\nStop requested. Finishing current database batch safely...")
        stop_requested = True
    else:
        print("\nForced termination.")
        sys.exit(1)


signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


def vector_to_pg(v):
    return "[" + ",".join(map(str, v.tolist())) + "]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encode-batch-size", type=int, default=128)
    parser.add_argument("--db-batch-size", type=int, default=10_000)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=10_000)
    args = parser.parse_args()

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is not set. "
            "Run: set -a; source .env; set +a"
        )

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"Model            : {MODEL_NAME}")
    print(f"Device           : {device}")
    print(f"Encode batch     : {args.encode_batch_size:,}")
    print(f"DB batch         : {args.db_batch_size:,}")

    model = SentenceTransformer(MODEL_NAME, device=device)

    dim = model.get_embedding_dimension()
    if dim != 384:
        raise RuntimeError(f"Expected 384 dimensions, got {dim}")

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT COUNT(*)
                FROM {TABLE}
                WHERE embedding IS NOT NULL
                """
            )
            already_embedded = cur.fetchone()[0]

        print(f"Already embedded : {already_embedded:,}")

        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT COALESCE(MAX(document_id), 0)
                FROM {TABLE}
                WHERE embedding IS NOT NULL
                """
            )
            last_id = cur.fetchone()[0]

        print(f"Resume after ID  : {last_id:,}")

        processed = 0
        start = time.perf_counter()
        last_log = 0

        while not stop_requested:
            remaining = args.db_batch_size

            if args.max_docs is not None:
                remaining = min(
                    remaining,
                    args.max_docs - processed
                )

                if remaining <= 0:
                    break

            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT document_id, content
                    FROM {TABLE}
                    WHERE document_id > %s
                      AND embedding IS NULL
                    ORDER BY document_id
                    LIMIT %s
                    """,
                    (last_id, remaining),
                )

                rows = cur.fetchall()

            if not rows:
                break

            ids = [row[0] for row in rows]
            texts = [row[1] for row in rows]

            embeddings = model.encode(
                texts,
                batch_size=args.encode_batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            if embeddings.shape != (len(rows), 384):
                raise RuntimeError(
                    f"Unexpected embedding shape: {embeddings.shape}"
                )

            norms = np.linalg.norm(embeddings, axis=1)

            if not np.allclose(norms, 1.0, atol=1e-4):
                raise RuntimeError(
                    "Embedding normalization check failed"
                )

            update_rows = [
                (vector_to_pg(embedding), document_id)
                for document_id, embedding in zip(ids, embeddings)
            ]

            try:
                with conn.cursor() as cur:
                    cur.executemany(
                        f"""
                        UPDATE {TABLE}
                        SET embedding = %s::vector
                        WHERE document_id = %s
                          AND embedding IS NULL
                        """,
                        update_rows,
                    )

                conn.commit()

            except Exception:
                conn.rollback()
                raise

            processed += len(rows)
            last_id = ids[-1]

            if (
                processed - last_log >= args.log_every
                or stop_requested
            ):
                if device == "mps":
                    torch.mps.synchronize()

                elapsed = time.perf_counter() - start
                rate = processed / elapsed if elapsed else 0

                print(
                    f"Embedded {processed:,} this session | "
                    f"last_id={last_id:,} | "
                    f"{rate:,.1f} passages/s"
                )

                last_log = processed

        elapsed = time.perf_counter() - start

        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    COUNT(*) FILTER (
                        WHERE embedding IS NOT NULL
                    ),
                    COUNT(*) FILTER (
                        WHERE embedding IS NULL
                    )
                FROM {TABLE}
                """
            )

            embedded, missing = cur.fetchone()

        print()
        print("Embedding run finished")
        print("----------------------")
        print(f"Session embedded : {processed:,}")
        print(f"Total embedded   : {embedded:,}")
        print(f"Remaining        : {missing:,}")
        print(f"Elapsed          : {elapsed:,.1f} s")

        if elapsed > 0:
            print(
                f"Overall rate     : "
                f"{processed / elapsed:,.1f} passages/s"
            )


if __name__ == "__main__":
    main()
