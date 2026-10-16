import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from backend import retrieval_cpp
from backend.scripts.evaluate_cpp_hybrid import CORPUS_PATH


BASE = Path("results/msmarco_full/relevance")

CACHE_PATH = BASE / "ce_candidates_10k.json"

CPP_RESULTS_PATH = (
    BASE / "cpp_v065/results_10k_10000q.json"
)

PG_RESULTS_PATH = (
    BASE / "ce_rerank_results_10k.json"
)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def candidate_ids(candidates):
    return {
        int(item["document_id"])
        for item in candidates
    }


def recall_at_candidates(candidate_set, relevant_set):
    if not relevant_set:
        return 0.0

    return len(candidate_set & relevant_set) / len(relevant_set)


def summarize(values):
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
    )

    args = parser.parse_args()

    print("Loading candidate cache...")
    cache = load_json(CACHE_PATH)

    print("Loading C++ BM25 index...")
    engine = retrieval_cpp.SearchEngine()
    engine.load_corpus(str(CORPUS_PATH))

    records = []

    stats = Counter()

    for i, row in enumerate(cache["queries"][:args.limit]):

        query_id = str(row["query_id"])
        query = row["query"]

        relevant = {
            int(x)
            for x in row["relevant_document_ids"]
        }

        pg_ids = candidate_ids(
            row["lexical"]["candidates"]
        )

        dense_ids = candidate_ids(
            row["dense"]["candidates"]
        )

        cpp_results = engine.search_dense(query, 50)

        cpp_ids = {
            int(item.doc_id)
            for item in cpp_results
        }

        pg_union = pg_ids | dense_ids
        cpp_union = cpp_ids | dense_ids

        pg_relevant = pg_union & relevant
        cpp_relevant = cpp_union & relevant

        cpp_rescued = cpp_relevant - pg_relevant
        pg_only = pg_relevant - cpp_relevant

        cpp_lexical_only = cpp_ids - dense_ids
        pg_lexical_only = pg_ids - dense_ids

        record = {
            "query_id": query_id,
            "query": query,

            "relevant_count": len(relevant),

            "pg_candidate_count": len(pg_union),
            "cpp_candidate_count": len(cpp_union),

            "pg_candidate_recall": recall_at_candidates(
                pg_union, relevant
            ),

            "cpp_candidate_recall": recall_at_candidates(
                cpp_union, relevant
            ),

            "pg_lexical_only_count": len(pg_lexical_only),
            "cpp_lexical_only_count": len(cpp_lexical_only),

            "cpp_rescued_relevant_count": len(cpp_rescued),
            "pg_only_relevant_count": len(pg_only),

            "cpp_rescued_document_ids": sorted(cpp_rescued),
            "pg_only_document_ids": sorted(pg_only),
        }

        records.append(record)

        if cpp_rescued:
            stats["cpp_rescue_queries"] += 1

        if pg_only:
            stats["pg_rescue_queries"] += 1

        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1}/{min(args.limit, len(cache['queries']))}")

    pg_recall = [
        r["pg_candidate_recall"]
        for r in records
    ]

    cpp_recall = [
        r["cpp_candidate_recall"]
        for r in records
    ]

    delta = [
        c - p
        for c, p in zip(cpp_recall, pg_recall)
    ]

    summary = {
        "queries": len(records),

        "pg_candidate_recall": summarize(pg_recall),

        "cpp_candidate_recall": summarize(cpp_recall),

        "candidate_recall_delta": summarize(delta),

        "cpp_rescue_queries": stats["cpp_rescue_queries"],

        "pg_rescue_queries": stats["pg_rescue_queries"],

        "mean_cpp_lexical_only": float(
            np.mean([
                r["cpp_lexical_only_count"]
                for r in records
            ])
        ),

        "mean_pg_lexical_only": float(
            np.mean([
                r["pg_lexical_only_count"]
                for r in records
            ])
        ),
    }

    output_dir = BASE / "cpp_v066"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "candidate_complementarity.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "per_query": records,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n=== Candidate Complementarity ===")

    print(
        "PostgreSQL Candidate Recall:",
        f"{summary['pg_candidate_recall']['mean']:.6f}"
    )

    print(
        "C++ Candidate Recall:",
        f"{summary['cpp_candidate_recall']['mean']:.6f}"
    )

    print(
        "Delta:",
        f"{summary['candidate_recall_delta']['mean']:+.6f}"
    )

    print(
        "C++ Rescue Queries:",
        summary["cpp_rescue_queries"]
    )

    print(
        "PG Rescue Queries:",
        summary["pg_rescue_queries"]
    )

    print(
        "Mean C++ Lexical-only Candidates:",
        f"{summary['mean_cpp_lexical_only']:.3f}"
    )

    print(
        "Mean PG Lexical-only Candidates:",
        f"{summary['mean_pg_lexical_only']:.3f}"
    )

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
