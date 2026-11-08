"""Frozen-candidate HTTP benchmark for Cross-Encoder serving."""

import argparse
import concurrent.futures
import hashlib
import json
import random
import statistics
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ENDPOINTS = {
    "baseline": "/internal/benchmark/rerank",
    "batched": "/internal/benchmark/rerank/batched",
}


def percentile(values, p):
    values = sorted(values)
    if not values:
        return None

    index = (len(values) - 1) * p / 100
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)

    fraction = index - lower

    return (
        values[lower] * (1 - fraction)
        + values[upper] * fraction
    )


def request_once(base_url, endpoint, payload):
    body = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        base_url + endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()

    with urllib.request.urlopen(
        request,
        timeout=60,
    ) as response:
        result = json.load(response)

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000

    return {
        "latency_ms": elapsed_ms,
        "result": result,
        "candidate_count": len(payload["candidates"]),
    }


def benchmark(
    base_url,
    endpoint,
    payloads,
    concurrency,
):
    started_at_utc = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=concurrency
    ) as executor:
        futures = [
            executor.submit(
                request_once,
                base_url,
                endpoint,
                payload,
            )
            for payload in payloads
        ]

        outputs = [
            future.result()
            for future in futures
        ]

    duration = time.perf_counter() - start
    finished_at_utc = datetime.now(timezone.utc).isoformat()

    latencies = [
        output["latency_ms"]
        for output in outputs
    ]

    pairs = sum(
        output["candidate_count"]
        for output in outputs
    )

    summary = {
        "requests": len(outputs),
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "duration_seconds": duration,
        "qps": len(outputs) / duration,
        "pairs_per_second": pairs / duration,
        "mean_ms": statistics.mean(latencies),
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "p99_ms": percentile(latencies, 99),
        "total_pairs": pairs,
    }

    if endpoint == ENDPOINTS["batched"]:
        profiles = [
            output["result"]
            for output in outputs
        ]

        summary["mean_batch_requests"] = (
            statistics.mean(
                profile["batch_request_count"]
                for profile in profiles
            )
        )

        summary["mean_batch_pairs"] = (
            statistics.mean(
                profile["batch_pair_count"]
                for profile in profiles
            )
        )

        summary["mean_queue_wait_ms"] = (
            statistics.mean(
                profile["batch_queue_wait_ms"]
                for profile in profiles
            )
        )

        summary["mean_backlog_wait_ms"] = (
            statistics.mean(
                profile["batch_backlog_wait_ms"]
                for profile in profiles
            )
        )

        summary["mean_formation_wait_ms"] = (
            statistics.mean(
                profile["batch_formation_wait_ms"]
                for profile in profiles
            )
        )

        summary["mean_batch_inference_ms"] = (
            statistics.mean(
                profile["batch_inference_ms"]
                for profile in profiles
            )
        )

        errors = [
            abs(
                profile["batch_queue_wait_ms"]
                - profile["batch_backlog_wait_ms"]
                - profile["batch_formation_wait_ms"]
            )
            for profile in profiles
        ]

        summary["queue_decomposition_max_error_ms"] = (
            max(errors, default=0.0)
        )

        if summary["queue_decomposition_max_error_ms"] > 0.002:
            raise AssertionError(
                "Queue wait decomposition exceeded tolerance."
            )

        summary["mean_token_counting_ms"] = statistics.mean(
            profile["token_counting_ms"]
            for profile in profiles
        )

        summary["p95_token_counting_ms"] = percentile(
            [
                profile["token_counting_ms"]
                for profile in profiles
            ],
            95,
        )

        summary["p95_queue_wait_ms"] = percentile(
            [
                profile["batch_queue_wait_ms"]
                for profile in profiles
            ],
            95,
        )

        token_counts = [
            profile["batch_token_count"]
            for profile in profiles
            if profile["batch_token_count"] is not None
        ]

        token_utilizations = [
            profile["batch_token_utilization"]
            for profile in profiles
            if profile["batch_token_utilization"] is not None
        ]

        summary["mean_batch_tokens"] = (
            statistics.mean(token_counts)
            if token_counts else None
        )

        summary["mean_token_utilization"] = (
            statistics.mean(token_utilizations)
            if token_utilizations else None
        )

    return summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
    )

    parser.add_argument(
        "--dataset",
        default=(
            "reports/performance/cross_encoder/"
            "datasets/msmarco_frozen_k50.json"
        ),
    )

    parser.add_argument(
        "--requests",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=[1, 4, 8],
    )

    parser.add_argument(
        "--pipelines",
        nargs="+",
        choices=list(ENDPOINTS),
        default=list(ENDPOINTS),
    )

    parser.add_argument(
        "--output",
        default=(
            "reports/performance/cross_encoder/"
            "frozen_http_benchmark.json"
        ),
    )

    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=16,
    )

    args = parser.parse_args()

    if args.warmup_requests < 1:
        parser.error("--warmup-requests must be positive.")

    dataset_path = Path(args.dataset)
    dataset_bytes = dataset_path.read_bytes()

    dataset_sha256 = hashlib.sha256(
        dataset_bytes
    ).hexdigest()

    dataset = json.loads(dataset_bytes)

    records = dataset["queries"]

    payloads = [
        {
            "query": record["query"],
            "top_k": 10,
            "candidates": record["candidates"],
        }
        for record in records
    ]

    random_generator = random.Random(42)

    print("=== FROZEN HTTP BENCHMARK ===")
    print("Dataset SHA256:", dataset_sha256)
    print("Queries:", len(payloads))

    # Stage 1: cold-start warmup.
    for pipeline in args.pipelines:
        endpoint = ENDPOINTS[pipeline]

        request_once(
            args.base_url,
            endpoint,
            payloads[0],
        )

    # Stage 2: concurrent scheduler warmup.
    for pipeline in args.pipelines:
        endpoint = ENDPOINTS[pipeline]

        warmup_payloads = [
            payloads[i % len(payloads)]
            for i in range(args.warmup_requests)
        ]

        benchmark(
            args.base_url,
            endpoint,
            warmup_payloads,
            concurrency=max(args.concurrency),
        )

    results = []

    for concurrency in args.concurrency:
        for repetition in range(args.repetitions):
            workload = [
                payloads[i % len(payloads)]
                for i in range(args.requests)
            ]

            random_generator.shuffle(workload)

            pipeline_order = list(args.pipelines)
            random_generator.shuffle(pipeline_order)

            for pipeline in pipeline_order:
                print(
                    f"\nC={concurrency} "
                    f"rep={repetition + 1} "
                    f"pipeline={pipeline}"
                )

                summary = benchmark(
                    args.base_url,
                    ENDPOINTS[pipeline],
                    workload,
                    concurrency,
                )

                record = {
                    "concurrency": concurrency,
                    "repetition": repetition + 1,
                    "pipeline": pipeline,
                    **summary,
                }

                results.append(record)

                print(
                    f"QPS={summary['qps']:.3f} "
                    f"P95={summary['p95_ms']:.3f}ms "
                    f"Pairs/s={summary['pairs_per_second']:.1f}"
                )

                if pipeline == "batched":
                    print(
                        "Queue="
                        f"{summary['mean_queue_wait_ms']:.2f}ms "
                        "Backlog="
                        f"{summary['mean_backlog_wait_ms']:.2f}ms "
                        "Formation="
                        f"{summary['mean_formation_wait_ms']:.2f}ms "
                        "Inference="
                        f"{summary['mean_batch_inference_ms']:.2f}ms "
                        "BatchReq="
                        f"{summary['mean_batch_requests']:.2f}"
                    )

    output = {
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "dataset_sha256": dataset_sha256,
        "configuration": vars(args),
        "results": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            output,
            indent=2,
        )
    )

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
