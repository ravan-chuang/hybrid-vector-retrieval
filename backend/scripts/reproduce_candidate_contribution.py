"""Recompute archived MS MARCO candidate-contribution summaries."""

import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path


ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts/evaluations/candidate-contribution"
METRICS = ("mrr_at_10", "ndcg_at_10", "recall_at_10")
CATEGORIES = ("rescue", "loss", "both", "neither")


def metric_summary(rows, metric):
    differences = [float(row[metric]) for row in rows]
    total = math.fsum(differences)
    return {
        "mean_delta": total / len(differences) if differences else 0.0,
        "improved": sum(value > 0 for value in differences),
        "regressed": sum(value < 0 for value in differences),
        "unchanged": sum(value == 0 for value in differences),
    }, total


def analyze(rows):
    summary = {"query_count": len(rows), "categories": {}, "overall": {}}
    for metric in METRICS:
        summary["overall"][metric] = metric_summary(rows, metric)[0]
    for category in CATEGORIES:
        group = [row for row in rows if row["category"] == category]
        metrics = {}
        for metric in METRICS:
            stats, total = metric_summary(group, metric)
            metrics[metric] = {**stats, "contribution_to_overall": total / len(rows)}
        summary["categories"][category] = {
            "count": len(group), "fraction": len(group) / len(rows), "metrics": metrics,
        }
    return summary


def verify(actual, expected, path="splits"):
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ValueError(f"Keys differ for {path}")
        for name, target in expected.items():
            verify(actual[name], target, f"{path}.{name}")
    elif isinstance(expected, float):
        if not math.isclose(actual, expected, abs_tol=1e-12, rel_tol=0):
            raise ValueError(f"Value differs for {path}")
    elif actual != expected:
        raise ValueError(f"Value differs for {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify the archived answer")
    args = parser.parse_args()
    observations = ARTIFACTS / "observations.csv.gz"
    source = json.loads((ARTIFACTS / "source.json").read_text())
    if hashlib.sha256(observations.read_bytes()).hexdigest() != source["observations_sha256"]:
        raise ValueError("Observation archive digest mismatch")

    splits = defaultdict(list)
    ids = set()
    with gzip.open(observations, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            if (row["category"] not in CATEGORIES or row["split"] not in ("validation", "test")
                    or row["query_id"] in ids):
                raise ValueError("Invalid category, split, or duplicate query ID")
            ids.add(row["query_id"])
            splits[row["split"]].append(row)
    actual = {name: analyze(rows) for name, rows in splits.items()}
    if args.check:
        expected = json.loads((ARTIFACTS / "expected.json").read_text())
        verify(actual, expected)
        print(f"Reproduced candidate contributions across {len(ids):,} queries.")
    else:
        print(json.dumps(actual, indent=2))


if __name__ == "__main__":
    main()
