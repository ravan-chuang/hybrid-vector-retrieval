"""Analyze batch-level inference metrics from open-loop benchmark JSON."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def analyze(path):
    data = json.loads(Path(path).read_text())

    records = data.get("records", [])

    groups = defaultdict(list)

    missing_id = 0

    for record in records:
        if not record.get("success"):
            continue

        profile = record.get("profile") or {}
        batch_id = profile.get("batch_id")

        if not batch_id:
            missing_id += 1
            continue

        groups[batch_id].append(record)

    complete = []
    incomplete = []
    inconsistent = []

    for batch_id, members in groups.items():
        profiles = [
            member["profile"]
            for member in members
        ]

        expected_counts = {
            p.get("batch_request_count")
            for p in profiles
        }

        expected_pairs = {
            p.get("batch_pair_count")
            for p in profiles
        }

        inference_times = {
            p.get("batch_inference_ms")
            for p in profiles
        }

        observed_requests = len(members)

        observed_pairs = sum(
            member.get("candidate_count", 0)
            for member in members
        )

        if (
            len(expected_counts) != 1
            or len(expected_pairs) != 1
            or len(inference_times) != 1
        ):
            inconsistent.append(batch_id)
            continue

        expected_requests = next(iter(expected_counts))
        expected_pair_count = next(iter(expected_pairs))
        inference_ms = next(iter(inference_times))

        if (
            not isinstance(expected_requests, int)
            or isinstance(expected_requests, bool)
            or not isinstance(expected_pair_count, int)
            or isinstance(expected_pair_count, bool)
            or not isinstance(inference_ms, (int, float))
            or isinstance(inference_ms, bool)
            or not math.isfinite(inference_ms)
            or inference_ms <= 0
        ):
            inconsistent.append(batch_id)
            continue

        row = {
            "batch_id": batch_id,
            "requests": observed_requests,
            "pairs": observed_pairs,
            "inference_ms": inference_ms,
        }

        if (
            observed_requests == expected_requests
            and observed_pairs == expected_pair_count
        ):
            complete.append(row)
        else:
            incomplete.append({
                **row,
                "expected_requests": expected_requests,
                "expected_pairs": expected_pair_count,
            })

    print(f"\n=== {Path(path).name} ===")

    print("Successful requests:", sum(
        bool(r.get("success")) for r in records
    ))

    print("Requests missing batch_id:", missing_id)

    print("Observed unique batches:", len(groups))

    print("Complete batches:", len(complete))
    print("Incomplete batches:", len(incomplete))
    print("Inconsistent batches:", len(inconsistent))

    if not complete:
        print("No complete batches available.")
        return

    pairs = np.array([
        b["pairs"] for b in complete
    ], dtype=float)

    requests = np.array([
        b["requests"] for b in complete
    ], dtype=float)

    inference = np.array([
        b["inference_ms"] for b in complete
    ], dtype=float)

    pair_rate = (
        pairs.sum() /
        (inference.sum() / 1000.0)
    )

    print("\n--- COMPLETE BATCH METRICS ---")

    print("Mean requests/batch:", round(
        float(requests.mean()), 3
    ))

    print("Mean pairs/batch:", round(
        float(pairs.mean()), 3
    ))

    print("Inference mean (ms):", round(
        float(inference.mean()), 3
    ))

    print("Inference p50 (ms):", round(
        float(np.percentile(inference, 50)), 3
    ))

    print("Inference p95 (ms):", round(
        float(np.percentile(inference, 95)), 3
    ))

    print("Inference p99 (ms):", round(
        float(np.percentile(inference, 99)), 3
    ))

    print("Inference pair rate (pairs/s):", round(
        float(pair_rate), 3
    ))

    print("Total pairs:", int(pairs.sum()))
    print("Total inference seconds:", round(
        float(inference.sum() / 1000.0), 3
    ))

    if incomplete:
        print("\n--- INCOMPLETE BATCH EXAMPLES ---")
        for batch in incomplete[:5]:
            print(batch)

    if inconsistent:
        print("\n--- INCONSISTENT BATCH EXAMPLES ---")
        for batch_id in inconsistent[:5]:
            print(batch_id)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "files",
        nargs="+",
        help="Open-loop benchmark JSON files",
    )

    args = parser.parse_args()

    for path in args.files:
        analyze(path)


if __name__ == "__main__":
    main()
