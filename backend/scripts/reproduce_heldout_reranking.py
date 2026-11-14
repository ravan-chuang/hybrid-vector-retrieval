"""Recompute the held-out reranking comparison from per-query hit ranks."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from backend.analyze_reranking import BOOTSTRAP_SAMPLES, SEED, paired_bootstrap


ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts" / "relevance"
FIXTURE = ARTIFACTS / "heldout_reranking" / "hit_ranks.csv"
REFERENCE = ARTIFACTS / "final_reranking_analysis.json"
METRIC_NAMES = ("mrr@10", "ndcg@10", "recall@10")
TOP_K = 10


def discount(rank):
    return 1 / math.log2(rank + 1)


def score(hits, relevant_count):
    return (
        1 / hits[0] if hits else 0.0,
        sum(map(discount, hits)) / sum(
            discount(rank) for rank in range(1, min(relevant_count, TOP_K) + 1)
        ),
        len(hits) / relevant_count,
    )


def rankings_to_hits(rows):
    query_ids = set()
    for index, row in enumerate(rows, 1):
        query_id = row["query_id"]
        if not query_id or query_id in query_ids:
            raise ValueError(f"Missing or repeated query ID on row {index}")
        query_ids.add(query_id)
        relevant = set(filter(None, row["relevant_doc_ids"].split("|")))
        yield {
            "row": str(index),
            "relevant_count": str(len(relevant)),
            "dense_hits": hit_ranks(row["dense_reranked_top10"], relevant),
            "union_hits": hit_ranks(row["union_reranked_top10"], relevant),
        }


def hit_ranks(ranking, relevant):
    ids = ranking.split("|") if ranking else []
    if len(ids) > TOP_K or len(set(ids)) != len(ids) or "" in ids:
        raise ValueError("Expected a Top-10 ranking with distinct passage IDs")
    return "|".join(str(rank) for rank, pid in enumerate(ids, 1) if pid in relevant)


def load_scores(rows, expected_queries):
    if len(rows) != expected_queries:
        raise ValueError(f"Expected {expected_queries} rows, found {len(rows)}")

    dense_scores = []
    union_scores = []
    for index, row in enumerate(rows, 1):
        if int(row["row"]) != index:
            raise ValueError(f"Expected contiguous row indices, found {row['row']!r}")
        relevant_count = int(row["relevant_count"])
        if relevant_count < 1:
            raise ValueError(f"Row {index} has no relevance judgments")
        for name, scores in (("dense_hits", dense_scores), ("union_hits", union_scores)):
            ranks = [int(rank) for rank in row[name].split("|")] if row[name] else []
            if ranks != sorted(set(ranks)) or len(ranks) > relevant_count or any(
                not 1 <= rank <= TOP_K for rank in ranks
            ):
                raise ValueError(f"Invalid {name} on row {index}: {ranks}")
            scores.append(score(ranks, relevant_count))
    return np.asarray(dense_scores), np.asarray(union_scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--input", type=Path, help="Read an exported hit-rank CSV.")
    sources.add_argument(
        "--rankings-csv", type=Path, help="Read a per-query CSV from the reranker."
    )
    parser.add_argument("--check", action="store_true", help="Check the archived comparison.")
    args = parser.parse_args()

    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    expected_queries = reference["evaluation"]["queries"]
    source = args.rankings_csv or args.input or FIXTURE
    with source.open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        if args.rankings_csv:
            rows = rankings_to_hits(rows)
        dense, union = load_scores(list(rows), expected_queries)
    rng = np.random.default_rng(SEED)
    comparisons = {
        metric: paired_bootstrap(dense[:, index], union[:, index], rng)
        for index, metric in enumerate(METRIC_NAMES)
    }

    print(f"{len(dense)} paired queries, {BOOTSTRAP_SAMPLES:,} bootstrap resamples, seed {SEED}")
    print()
    print("| Metric | Dense + CE | Union + CE | Delta | Paired 95% CI | Wins / Ties / Losses |")
    print("| --- | ---: | ---: | ---: | --- | ---: |")
    for metric, row in comparisons.items():
        print(
            f"| {metric} | {row['dense_mean']:.6f} | {row['union_mean']:.6f} "
            f"| {row['delta_union_minus_dense']:+.6f} "
            f"| [{row['ci95_lower']:+.6f}, {row['ci95_upper']:+.6f}] "
            f"| {row['wins']} / {row['ties']} / {row['losses']} |"
        )

    if args.check:
        for metric, row in comparisons.items():
            for field, value in row.items():
                expected = reference["paired_results"][metric][field]
                if not math.isclose(value, expected, abs_tol=1e-12, rel_tol=0):
                    raise SystemExit(f"{metric} {field}: expected {expected}, got {value}")
        print("\nAll comparison statistics match the archived reference.")


if __name__ == "__main__":
    main()
