"""v0.7.5: Cross-request batching feasibility benchmark."""

import argparse
import json
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

INPUT_PATH = Path(
    "results/msmarco_full/parity/v071/fixed_candidates.json"
)

OUTPUT_DIR = Path(
    "results/msmarco_full/latency/api_v075/cross_request_batch"
)


def load_requests():
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))

    requests = []

    for record in data:
        query = record["query"]

        pairs = [
            (query, candidate["content"])
            for candidate in record["candidates"]
        ]

        requests.append(pairs)

    if len(requests) != 10:
        raise ValueError(
            f"Expected 10 requests, found {len(requests)}"
        )

    return requests


def make_groups(requests, group_size):
    return [
        requests[i:i + group_size]
        for i in range(0, len(requests), group_size)
    ]


def run_grouped_inference(model, requests, group_size, batch_size):
    """Return per-request scores in original request order."""

    groups = make_groups(requests, group_size)
    all_scores = []

    for group in groups:
        merged_pairs = [
            pair
            for request_pairs in group
            for pair in request_pairs
        ]

        scores = model.predict(
            merged_pairs,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        torch.mps.synchronize()

        scores = np.asarray(
            scores,
            dtype=np.float64,
        ).reshape(-1)

        offset = 0

        for request_pairs in group:
            count = len(request_pairs)

            all_scores.append(
                scores[offset:offset + count].copy()
            )

            offset += count

        if offset != len(scores):
            raise RuntimeError("Score split mismatch.")

    return all_scores


def compare_rankings(reference, current, top_k=10):
    max_abs_diff = 0.0
    exact_top10 = 0

    for reference_scores, current_scores in zip(
        reference,
        current,
    ):
        if len(reference_scores) != len(current_scores):
            raise RuntimeError("Candidate count mismatch.")

        difference = float(
            np.max(
                np.abs(
                    reference_scores - current_scores
                )
            )
        )

        max_abs_diff = max(
            max_abs_diff,
            difference,
        )

        reference_order = sorted(
            range(len(reference_scores)),
            key=lambda i: (-reference_scores[i], i),
        )[:top_k]

        current_order = sorted(
            range(len(current_scores)),
            key=lambda i: (-current_scores[i], i),
        )[:top_k]

        if reference_order == current_order:
            exact_top10 += 1

    return {
        "max_abs_score_difference": max_abs_diff,
        "exact_top10_queries": exact_top10,
        "total_queries": len(reference),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--group-sizes",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 10],
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable.")

    if args.batch_size < 1:
        parser.error("Batch size must be positive.")

    if args.warmup < 0 or args.repeats < 1:
        parser.error("Invalid warmup or repeat count.")

    if any(size < 1 for size in args.group_sizes):
        parser.error("Group sizes must be positive.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    requests = load_requests()

    pair_count = sum(len(request) for request in requests)

    print("=" * 76)
    print("v0.7.5 CROSS-REQUEST BATCHING FEASIBILITY")
    print("=" * 76)

    print("Device: MPS")
    print(f"Queries: {len(requests)}")
    print(f"Pairs: {pair_count}")
    print(f"Group sizes: {args.group_sizes}")
    print(f"Internal batch size: {args.batch_size}")

    model = CrossEncoder(
        MODEL_NAME,
        device="mps",
        activation_fn=torch.nn.Identity(),
    )

    print("\nCreating reference scores...")

    reference = run_grouped_inference(
        model,
        requests,
        group_size=1,
        batch_size=args.batch_size,
    )

    records = []

    rng = random.Random(args.seed)

    for repeat in range(args.repeats):
        order = list(args.group_sizes)
        rng.shuffle(order)

        print(f"\nRepeat {repeat + 1}: {order}")

        for group_size in order:
            for _ in range(args.warmup):
                run_grouped_inference(
                    model,
                    requests,
                    group_size,
                    args.batch_size,
                )

            torch.mps.synchronize()

            start = time.perf_counter()

            scores = run_grouped_inference(
                model,
                requests,
                group_size,
                args.batch_size,
            )

            torch.mps.synchronize()

            elapsed = time.perf_counter() - start

            parity = compare_rankings(
                reference,
                scores,
            )

            record = {
                "repeat": repeat + 1,
                "group_size": group_size,
                "request_count": len(requests),
                "pair_count": pair_count,
                "latency_ms": elapsed * 1000,
                "requests_per_second": len(requests) / elapsed,
                "pairs_per_second": pair_count / elapsed,
                **parity,
            }

            records.append(record)

            print(
                f"G={group_size:<2} "
                f"Latency={record['latency_ms']:.2f} ms "
                f"Req/s={record['requests_per_second']:.2f} "
                f"Pairs/s={record['pairs_per_second']:.2f} "
                f"Top10={parity['exact_top10_queries']}/10"
            )

    summaries = []

    for group_size in args.group_sizes:
        group_records = [
            row
            for row in records
            if row["group_size"] == group_size
        ]

        summary = {
            "group_size": group_size,
            "mean_latency_ms": statistics.mean(
                row["latency_ms"]
                for row in group_records
            ),
            "median_latency_ms": statistics.median(
                row["latency_ms"]
                for row in group_records
            ),
            "mean_requests_per_second": statistics.mean(
                row["requests_per_second"]
                for row in group_records
            ),
            "mean_pairs_per_second": statistics.mean(
                row["pairs_per_second"]
                for row in group_records
            ),
            "worst_max_abs_score_difference": max(
                row["max_abs_score_difference"]
                for row in group_records
            ),
            "min_exact_top10_queries": min(
                row["exact_top10_queries"]
                for row in group_records
            ),
        }

        summaries.append(summary)

    metadata = {
        "version": "v0.7.5",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "device": "mps",
        "internal_batch_size": args.batch_size,
        "group_sizes": args.group_sizes,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "seed": args.seed,
        "request_count": len(requests),
        "pair_count": pair_count,
        "measurement": (
            "Offline grouped CrossEncoder.predict wall time; "
            "includes tokenization and MPS synchronization. "
            "Excludes HTTP, retrieval, queue waiting, and scheduling."
        ),
    }

    for filename, payload in [
        ("metadata.json", metadata),
        ("per_run.json", records),
        ("summary.json", summaries),
    ]:
        (OUTPUT_DIR / filename).write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    baseline = next(
        row
        for row in summaries
        if row["group_size"] == 1
    )

    print("\n" + "=" * 76)
    print("SUMMARY")
    print("=" * 76)

    for row in summaries:
        speedup = (
            baseline["mean_latency_ms"]
            / row["mean_latency_ms"]
        )

        print(
            f"G={row['group_size']:<2} "
            f"Mean={row['mean_latency_ms']:.2f} ms "
            f"Req/s={row['mean_requests_per_second']:.2f} "
            f"Speedup={speedup:.3f}x "
            f"Top10={row['min_exact_top10_queries']}/10"
        )

    print(f"\nResults: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
