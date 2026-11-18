"""Reproduce historical candidate coverage and Cross-Encoder budget summaries."""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean

from backend.scripts.reproduce_candidate_contribution import verify


ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
ARCHIVE = ARTIFACTS / "evaluations/candidate-coverage"
MANIFEST = ARTIFACTS / "msmarco_full/dev_small_manifest.json"
BUDGETS = (10, 20, 30, 40, 50)
METHODS = ("dense", "rrf")


def checked_bytes(path, sha256):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError(f"Digest mismatch for {path.name}")
    return data


def load_observations(source):
    manifest = json.loads(checked_bytes(MANIFEST, source["manifest_sha256"]))
    qrels = {
        str(query["query_id"]): len(set(query["relevant_document_ids"]))
        for query in manifest["queries"]
    }
    packed = checked_bytes(ARCHIVE / "observations.jsonl.gz", source["observations_sha256"])
    seen = set()
    splits = {"validation": [], "test": []}
    for line in gzip.decompress(packed).splitlines():
        item = json.loads(line)
        query_id = item["query_id"]
        if query_id not in qrels or query_id in seen:
            raise ValueError(f"Unknown or repeated query ID {query_id}")
        seen.add(query_id)
        split = "validation" if hashlib.sha256(query_id.encode()).digest()[0] < 128 else "test"
        splits[split].append((item, qrels[query_id]))
    if seen != qrels.keys() or len(seen) != source["query_count"]:
        raise ValueError("Observations do not cover the manifest query IDs")
    return splits


def coverage_metrics(row, relevant):
    bm25 = len(row["bm25_hits"])
    dense = len(row["dense_hits"])
    union = len(row["fused_hits"])
    shared = bm25 + dense - union
    fused_at_20 = sum(rank <= 20 for rank in row["fused_hits"])
    fused_at_50 = sum(rank <= 50 for rank in row["fused_hits"])
    return {
        "bm25_recall_50": bm25 / relevant,
        "dense_recall_50": dense / relevant,
        "union_recall": union / relevant,
        "candidate_recall_20": fused_at_20 / relevant,
        "candidate_recall_50": fused_at_50 / relevant,
        "candidate_recall_100": union / relevant,
        "bm25_only_relevant": bm25 - shared,
        "dense_only_relevant": dense - shared,
        "shared_relevant": shared,
        "union_relevant": union,
        "rrf_50_relevant": fused_at_50,
        "union_size": row["union_size"],
        "overlap_size": row["overlap_size"],
        "relevant_count": relevant,
    }


def coverage_summary(splits):
    rows = [
        coverage_metrics(row, relevant)
        for split in splits.values()
        for row, relevant in split
    ]
    summary = {metric: fmean(row[metric] for row in rows) for metric in rows[0]}
    total_union = sum(row["union_relevant"] for row in rows)
    summary["candidate_retention_50"] = (
        sum(row["rrf_50_relevant"] for row in rows) / total_union if total_union else None
    )
    return summary


def rank_metrics(hits, relevant):
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(relevant, 10) + 1))
    return {
        "mrr_at_10": 1 / hits[0] if hits else 0.0,
        "ndcg_at_10": sum(1 / math.log2(rank + 1) for rank in hits) / ideal,
        "recall_at_10": len(hits) / relevant,
    }


def budget_and_ablation(splits):
    budgets = {}
    ablation = {}
    for name, rows in splits.items():
        budgets[name] = {}
        for index, budget in enumerate(BUDGETS):
            comparison = {}
            for method in METHODS:
                metrics = [
                    rank_metrics(row["ce_hits"][method][index], relevant)
                    for row, relevant in rows
                ]
                comparison[method] = {key: fmean(row[key] for row in metrics) for key in metrics[0]}
            budgets[name][str(budget)] = comparison
        ablation[name] = {
            "query_count": len(rows),
            "dense_ce": budgets[name]["50"]["dense"],
            "rrf_ce": budgets[name]["50"]["rrf"],
            "mean_scored_candidates": fmean(row["scored_candidates"] for row, _ in rows),
        }
    return budgets, ablation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Compare with the historical summaries")
    args = parser.parse_args()
    source = json.loads((ARCHIVE / "source.json").read_text())
    splits = load_observations(source)
    budgets, ablation = budget_and_ablation(splits)
    actual = {"coverage": coverage_summary(splits), "budget_sweep": budgets, "ablation_top50": ablation}
    if args.check:
        expected = json.loads((ARCHIVE / "expected.json").read_text())
        verify(actual, expected, path="candidate evaluation")
        print(f"Reproduced coverage and Cross-Encoder budget summaries for {source['query_count']:,} queries.")
    else:
        print(json.dumps(actual, indent=2))


if __name__ == "__main__":
    main()
