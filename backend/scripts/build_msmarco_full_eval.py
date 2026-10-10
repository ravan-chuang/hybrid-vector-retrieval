#!/usr/bin/env python3

import json
import os
import random
import time
from pathlib import Path

import psycopg
from datasets import load_dataset


SEED = 42
EVAL_SIZE = 500

OUT_DIR = Path("results/msmarco_full/relevance")
ARTIFACT_DIR = Path("artifacts/msmarco_full")

COVERAGE_PATH = OUT_DIR / "qrels_coverage.json"
MANIFEST_PATH = OUT_DIR / "eval_manifest.json"
ARTIFACT_COVERAGE_PATH = ARTIFACT_DIR / "qrels_coverage.json"
ARTIFACT_MANIFEST_PATH = ARTIFACT_DIR / "eval_manifest.json"


def get_database_url():
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Run: "
            "set -a; source .env; set +a"
        )
    return url


def load_query_texts():
    print("Loading MS MARCO query texts...")

    ds = load_dataset(
        "sentence-transformers/msmarco",
        "queries",
        split="train",
        streaming=True,
    )

    queries = {}

    for i, row in enumerate(ds, start=1):
        qid = str(row["query_id"])

        # Current dataset schema normally exposes "query".
        # Keep fallback for compatibility with alternate versions.
        text = row.get("query")
        if text is None:
            text = row.get("text")

        if text is None:
            raise RuntimeError(
                f"Could not find query text field. Row keys: {list(row.keys())}"
            )

        queries[qid] = text

        if i % 100_000 == 0:
            print(f"  loaded {i:,} query texts")

    print(f"Query texts loaded: {len(queries):,}")
    return queries


def load_positive_qrels():
    print("\nStreaming MS MARCO labeled-list qrels...")

    ds = load_dataset(
        "sentence-transformers/msmarco",
        "labeled-list",
        split="train",
        streaming=True,
    )

    positives = {}
    rows_seen = 0
    positive_judgments = 0

    for row in ds:
        rows_seen += 1
        qid = str(row["query_id"])

        docs = row["doc_ids"]
        labels = row["labels"]

        if len(docs) != len(labels):
            raise RuntimeError(
                f"doc_ids/labels length mismatch for query {qid}"
            )

        relevant = [
            int(doc_id)
            for doc_id, label in zip(docs, labels)
            if int(label) == 1
        ]

        if relevant:
            positives.setdefault(qid, set()).update(relevant)
            positive_judgments += len(relevant)

        if rows_seen % 10_000 == 0:
            print(
                f"  rows={rows_seen:,} "
                f"queries_with_positive={len(positives):,} "
                f"positive_judgments_seen={positive_judgments:,}"
            )

    # Convert sets to sorted lists for deterministic serialization.
    positives = {
        qid: sorted(doc_ids)
        for qid, doc_ids in positives.items()
    }

    unique_positive_judgments = sum(
        len(doc_ids) for doc_ids in positives.values()
    )

    print(f"Labeled-list rows       : {rows_seen:,}")
    print(f"Queries with positives  : {len(positives):,}")
    print(f"Unique positive qrels   : {unique_positive_judgments:,}")

    return positives, rows_seen, unique_positive_judgments


def check_corpus_coverage(conn, positives):
    print("\nChecking positive-qrel coverage against full corpus...")

    all_positive_pids = sorted({
        pid
        for doc_ids in positives.values()
        for pid in doc_ids
    })

    # MS MARCO pid is zero-based in the HF corpus.
    # Our PostgreSQL document_id = pid + 1.
    expected_document_ids = [pid + 1 for pid in all_positive_pids]

    covered_document_ids = set()

    batch_size = 10_000

    with conn.cursor() as cur:
        for start in range(0, len(expected_document_ids), batch_size):
            batch = expected_document_ids[start:start + batch_size]

            cur.execute(
                """
                SELECT document_id
                FROM msmarco_full_documents
                WHERE document_id = ANY(%s)
                """,
                (batch,),
            )

            covered_document_ids.update(
                row[0] for row in cur.fetchall()
            )

            done = min(start + batch_size, len(expected_document_ids))
            if done % 50_000 == 0 or done == len(expected_document_ids):
                print(
                    f"  checked {done:,}/{len(expected_document_ids):,} "
                    f"unique relevant passages"
                )

    covered_pids = {
        document_id - 1
        for document_id in covered_document_ids
    }

    total_judgments = 0
    covered_judgments = 0
    eligible_queries = {}

    for qid, relevant_pids in positives.items():
        total_judgments += len(relevant_pids)

        covered = [
            pid for pid in relevant_pids
            if pid in covered_pids
        ]

        covered_judgments += len(covered)

        if covered:
            eligible_queries[qid] = covered

    return {
        "unique_positive_passages": len(all_positive_pids),
        "covered_unique_positive_passages": len(covered_pids),
        "total_positive_judgments": total_judgments,
        "covered_positive_judgments": covered_judgments,
        "eligible_queries": eligible_queries,
    }


def main():
    started = time.time()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    database_url = get_database_url()

    queries = load_query_texts()
    positives, labeled_rows, unique_positive_judgments = (
        load_positive_qrels()
    )

    with psycopg.connect(database_url) as conn:
        coverage = check_corpus_coverage(conn, positives)

    eligible = {}

    missing_query_text = 0

    for qid, relevant_pids in coverage["eligible_queries"].items():
        text = queries.get(qid)

        if text is None:
            missing_query_text += 1
            continue

        eligible[qid] = {
            "query_id": qid,
            "query": text,
            "relevant_pids": relevant_pids,
            "relevant_document_ids": [
                pid + 1 for pid in relevant_pids
            ],
        }

    if len(eligible) < EVAL_SIZE:
        raise RuntimeError(
            f"Only {len(eligible):,} eligible queries; "
            f"cannot sample {EVAL_SIZE:,}."
        )

    rng = random.Random(SEED)
    sampled_qids = rng.sample(sorted(eligible.keys()), EVAL_SIZE)

    eval_queries = [
        eligible[qid]
        for qid in sampled_qids
    ]

    total_positive = coverage["total_positive_judgments"]
    covered_positive = coverage["covered_positive_judgments"]

    judgment_coverage = (
        covered_positive / total_positive
        if total_positive
        else 0.0
    )

    unique_passage_coverage = (
        coverage["covered_unique_positive_passages"]
        / coverage["unique_positive_passages"]
        if coverage["unique_positive_passages"]
        else 0.0
    )

    coverage_summary = {
        "dataset": "sentence-transformers/msmarco",
        "config": "labeled-list",
        "split": "train",
        "corpus": "sentence-transformers/msmarco-corpus passage train",
        "corpus_table": "msmarco_full_documents",
        "corpus_size": 8_841_823,
        "mapping": "document_id = pid + 1",
        "labeled_list_rows": labeled_rows,
        "queries_with_positive_qrels": len(positives),
        "unique_positive_judgments": unique_positive_judgments,
        "unique_positive_passages": coverage[
            "unique_positive_passages"
        ],
        "covered_unique_positive_passages": coverage[
            "covered_unique_positive_passages"
        ],
        "unique_positive_passage_coverage": unique_passage_coverage,
        "total_positive_judgments": total_positive,
        "covered_positive_judgments": covered_positive,
        "positive_judgment_coverage": judgment_coverage,
        "eligible_queries_with_corpus_positive": len(
            coverage["eligible_queries"]
        ),
        "eligible_queries_with_text": len(eligible),
        "missing_query_text": missing_query_text,
    }

    manifest = {
        "protocol": "msmarco-full-relevance-v1",
        "seed": SEED,
        "sample_size": EVAL_SIZE,
        "selection": (
            "Uniform random sample without replacement from queries "
            "having >=1 positive labeled-list judgment present in the "
            "full local MS MARCO passage corpus and available query text."
        ),
        "qrels_source": (
            "sentence-transformers/msmarco labeled-list training split"
        ),
        "corpus_size": 8_841_823,
        "queries": eval_queries,
    }

    for path in (COVERAGE_PATH, ARTIFACT_COVERAGE_PATH):
        path.write_text(
            json.dumps(coverage_summary, indent=2),
            encoding="utf-8",
        )

    for path in (MANIFEST_PATH, ARTIFACT_MANIFEST_PATH):
        path.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

    elapsed = time.time() - started

    print("\n=== Full-corpus qrels coverage ===")
    print(
        f"Positive judgments      : "
        f"{total_positive:,}"
    )
    print(
        f"Covered judgments       : "
        f"{covered_positive:,}"
    )
    print(
        f"Judgment coverage       : "
        f"{judgment_coverage:.4%}"
    )
    print(
        f"Unique positive passages: "
        f"{coverage['unique_positive_passages']:,}"
    )
    print(
        f"Covered unique passages : "
        f"{coverage['covered_unique_positive_passages']:,}"
    )
    print(
        f"Unique passage coverage : "
        f"{unique_passage_coverage:.4%}"
    )
    print(
        f"Eligible queries        : "
        f"{len(eligible):,}"
    )
    print(
        f"Frozen evaluation sample: "
        f"{len(eval_queries):,}"
    )
    print(f"Seed                    : {SEED}")
    print(f"Elapsed                 : {elapsed:.1f}s")

    print("\nSaved:")
    print(f"  {COVERAGE_PATH}")
    print(f"  {MANIFEST_PATH}")
    print(f"  {ARTIFACT_COVERAGE_PATH}")
    print(f"  {ARTIFACT_MANIFEST_PATH}")


if __name__ == "__main__":
    main()
