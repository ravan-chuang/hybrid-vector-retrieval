#!/usr/bin/env python3

import argparse
import json
import os
import time
from pathlib import Path

import psycopg
from dotenv import load_dotenv

from backend.scripts import (
    evaluate_msmarco_full_retrieval_10k as retrieval,
)

MANIFEST_PATH = Path(
    "artifacts/msmarco_full/dev_small_manifest.json"
)

OUT_DIR = Path(
    "results/msmarco_full/relevance/dev_v067"
)

CHECKPOINT_PATH = OUT_DIR / "candidates_checkpoint.json"
RESULT_PATH = OUT_DIR / "ce_candidates_dev.json"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + ".tmp")

    tmp.write_text(
        json.dumps(data, ensure_ascii=False),
        encoding="utf-8",
    )

    tmp.replace(path)


def fetch_contents(conn, document_ids):
    if not document_ids:
        return {}

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT document_id, content
            FROM public.msmarco_full_documents
            WHERE document_id = ANY(%s)
            """,
            (document_ids,),
        )

        return {
            int(doc_id): content
            for doc_id, content in cur.fetchall()
        }


def attach_contents(candidates, documents):
    output = []

    for candidate in candidates:
        doc_id = int(candidate["document_id"])

        if doc_id not in documents:
            raise RuntimeError(
                f"Missing document: {doc_id}"
            )

        output.append({
            "document_id": doc_id,
            "content": documents[doc_id],
            "score": float(candidate["score"]),
        })

    return output


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    if args.checkpoint_every <= 0:
        parser.error("--checkpoint-every must be positive")

    load_dotenv(".env")

    manifest = load_json(MANIFEST_PATH)
    all_queries = manifest["queries"]

    if len(all_queries) != 6980:
        raise RuntimeError(
            f"Expected 6980 queries, got {len(all_queries)}"
        )

    queries = (
        all_queries[:args.limit]
        if args.limit is not None
        else all_queries
    )

    protocol = {
        "dataset": "msmarco_dev_small",
        "queries": len(all_queries),
        "candidate_k": retrieval.CANDIDATE_K,
        "requested_ef_search": retrieval.EF_SEARCH,
        "effective_ef_search": retrieval.EF_SEARCH,
        "query_model": retrieval.MODEL_NAME,
        "normalized_embeddings": True,
        "document_id_mapping": "msmarco_pid_plus_1",
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if CHECKPOINT_PATH.exists():
        checkpoint = load_json(CHECKPOINT_PATH)

        if checkpoint["protocol"] != protocol:
            raise RuntimeError(
                "Checkpoint protocol mismatch"
            )

        completed = checkpoint["completed"]

        print(
            f"Resuming {len(completed)} cached queries"
        )
    else:
        completed = {}

    pending = [
        query
        for query in queries
        if str(query["query_id"]) not in completed
    ]

    print("=" * 65)
    print("v0.6.7 — MS MARCO Dev Candidate Extraction")
    print("=" * 65)
    print("Queries:", len(queries))
    print("Pending:", len(pending))
    print("Candidate K:", retrieval.CANDIDATE_K)
    print("HNSW ef_search:", retrieval.EF_SEARCH)

    if pending:
        vectors = retrieval.encode_queries(pending)

        database_url = os.environ.get("DATABASE_URL")

        if not database_url:
            raise RuntimeError(
                "DATABASE_URL missing from .env"
            )

        started = time.time()

        with psycopg.connect(database_url) as conn:
            for index, query in enumerate(pending, 1):
                qid = str(query["query_id"])
                text = query["query"]

                lexical, lexical_ms = retrieval.lexical_search(
                    conn,
                    text,
                )

                dense, dense_ms = retrieval.dense_search(
                    conn,
                    vectors[qid],
                )

                ids = list(dict.fromkeys(
                    [int(x["document_id"]) for x in lexical]
                    + [int(x["document_id"]) for x in dense]
                ))

                documents = fetch_contents(conn, ids)

                if len(documents) != len(ids):
                    raise RuntimeError(
                        f"Document fetch mismatch: {qid}"
                    )

                completed[qid] = {
                    "query_id": qid,
                    "query": text,
                    "relevant_document_ids": [
                        int(x)
                        for x in query["relevant_document_ids"]
                    ],
                    "lexical": {
                        "latency_ms": lexical_ms,
                        "candidates": attach_contents(
                            lexical,
                            documents,
                        ),
                    },
                    "dense": {
                        "latency_ms": dense_ms,
                        "candidates": attach_contents(
                            dense,
                            documents,
                        ),
                    },
                }

                done = len(completed)

                if (
                    index % args.checkpoint_every == 0
                    or index == len(pending)
                ):
                    save_json_atomic(
                        CHECKPOINT_PATH,
                        {
                            "protocol": protocol,
                            "completed": completed,
                        },
                    )

                    print(
                        f"[{done}/{len(queries)}] "
                        f"FTS={lexical_ms:.1f}ms "
                        f"Dense={dense_ms:.1f}ms "
                        f"Elapsed={time.time()-started:.1f}s"
                    )

    # Only publish the final cache after all 6980 queries.
    if len(completed) == len(all_queries):
        rows = [
            completed[str(query["query_id"])]
            for query in all_queries
        ]

        save_json_atomic(
            RESULT_PATH,
            {
                "protocol": protocol,
                "queries": rows,
            },
        )

        print("\nSaved:", RESULT_PATH)
        print("Total queries:", len(rows))
    else:
        print(
            "\nPartial checkpoint saved:",
            CHECKPOINT_PATH,
        )
        print(
            "Run without --limit to complete all queries."
        )


if __name__ == "__main__":
    main()
