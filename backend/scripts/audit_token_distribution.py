"""Audit Cross-Encoder token distribution on frozen candidates."""

import argparse
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from backend.api.services.cross_encoder import get_cross_encoder
from backend.api.services.token_budget import estimate_request_tokens


def percentile(values, p):
    values = sorted(values)

    if not values:
        return None

    position = (len(values) - 1) * p / 100
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)

    fraction = position - lower

    return (
        values[lower] * (1 - fraction)
        + values[upper] * fraction
    )


def describe(values):
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "mean": statistics.mean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values) if values else None,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default=(
            "reports/performance/cross_encoder/"
            "datasets/msmarco_frozen_k50.json"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "reports/performance/cross_encoder/"
            "token_distribution_audit.json"
        ),
    )

    parser.add_argument(
        "--max-batch-requests",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--max-batch-pairs",
        type=int,
        default=800,
    )

    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[8192, 16384, 32768, 65536],
    )

    args = parser.parse_args()

    if args.max_batch_requests < 1 or args.max_batch_pairs < 1:
        parser.error("Batch limits must be positive.")

    if not args.budgets or any(b < 1 for b in args.budgets):
        parser.error("All token budgets must be positive.")

    dataset_path = Path(args.dataset)
    raw = dataset_path.read_bytes()
    dataset = json.loads(raw)

    model = get_cross_encoder()
    tokenizer = model.tokenizer
    max_length = model.max_seq_length

    request_tokens = []
    request_pairs = []
    pair_tokens = []
    request_records = []

    for index, record in enumerate(dataset["queries"]):
        query = record["query"]
        candidates = record["candidates"]

        estimate = estimate_request_tokens(
            tokenizer=tokenizer,
            query=query,
            candidates=candidates,
            max_length=max_length,
        )

        encoded = tokenizer(
            [query] * len(candidates),
            [candidate["content"] for candidate in candidates],
            truncation=True,
            max_length=max_length,
            padding=False,
            add_special_tokens=True,
            return_attention_mask=False,
        )

        lengths = [
            len(ids)
            for ids in encoded["input_ids"]
        ]

        assert sum(lengths) == estimate.total_tokens

        request_tokens.append(estimate.total_tokens)
        request_pairs.append(estimate.pair_count)
        pair_tokens.extend(lengths)

        request_records.append({
            "index": index,
            "pairs": estimate.pair_count,
            "tokens": estimate.total_tokens,
            "max_pair_tokens": estimate.max_pair_tokens,
        })

    # Deterministic, non-overlapping groups in dataset order.
    groups = []
    current = []
    current_pairs = 0

    for record in request_records:
        exceeds_requests = (
            len(current) >= args.max_batch_requests
        )
        exceeds_pairs = (
            current_pairs + record["pairs"]
            > args.max_batch_pairs
        )

        if current and (exceeds_requests or exceeds_pairs):
            groups.append(current)
            current = []
            current_pairs = 0

        current.append(record)
        current_pairs += record["pairs"]

    if current:
        groups.append(current)

    group_tokens = [
        sum(record["tokens"] for record in group)
        for group in groups
    ]

    group_pairs = [
        sum(record["pairs"] for record in group)
        for group in groups
    ]

    budgets = {}

    for budget in args.budgets:
        oversized_requests = sum(
            tokens > budget
            for tokens in request_tokens
        )

        oversized_groups = sum(
            tokens > budget
            for tokens in group_tokens
        )

        budgets[str(budget)] = {
            "oversized_request_count": oversized_requests,
            "oversized_request_rate": (
                oversized_requests / len(request_tokens)
                if request_tokens else 0.0
            ),
            "oversized_group_count": oversized_groups,
            "oversized_group_rate": (
                oversized_groups / len(group_tokens)
                if group_tokens else 0.0
            ),
        }

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "model_max_seq_length": max_length,
        "max_batch_requests": args.max_batch_requests,
        "max_batch_pairs": args.max_batch_pairs,
        "request_tokens": describe(request_tokens),
        "request_pairs": describe(request_pairs),
        "pair_tokens": describe(pair_tokens),
        "group_tokens": describe(group_tokens),
        "group_pairs": describe(group_pairs),
        "budgets": budgets,
        "requests": request_records,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(report, indent=2)
    )

    print("=== TOKEN DISTRIBUTION AUDIT ===")
    print("Dataset SHA256:", report["dataset_sha256"])
    print("Max sequence length:", max_length)
    print("Request tokens:", report["request_tokens"])
    print("Pair tokens:", report["pair_tokens"])
    print("Group tokens:", report["group_tokens"])
    print("Budget feasibility:")

    for budget, result in budgets.items():
        print(
            f"  {budget}: "
            f"oversized_requests="
            f"{result['oversized_request_count']} "
            f"oversized_groups="
            f"{result['oversized_group_count']}"
        )

    print("Saved:", output_path)


if __name__ == "__main__":
    main()
