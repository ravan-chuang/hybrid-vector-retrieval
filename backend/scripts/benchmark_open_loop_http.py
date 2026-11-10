"""Poisson open-loop HTTP benchmark for frozen Cross-Encoder requests."""

import argparse
import concurrent.futures
import hashlib
import json
import math
import random
import statistics
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ENDPOINTS = {
    "baseline": "/internal/benchmark/rerank",
    "batched": "/internal/benchmark/rerank/batched",
}


def percentile(values, p):
    if not values:
        return None

    values = sorted(values)

    position = (len(values) - 1) * p / 100
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)

    weight = position - lower

    return (
        values[lower] * (1 - weight)
        + values[upper] * weight
    )


def summarize(values):
    if not values:
        return {
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
        }

    return {
        "mean": statistics.mean(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }


def load_dataset(path):
    data = json.loads(path.read_text())

    if isinstance(data, list):
        payloads = data
    elif isinstance(data, dict):
        if isinstance(data.get("queries"), list):
            payloads = data["queries"]
        elif isinstance(data.get("requests"), list):
            payloads = data["requests"]
        else:
            raise ValueError(
                "Unsupported frozen dataset structure."
            )
    else:
        raise ValueError(
            "Unsupported frozen dataset structure."
        )

    if not payloads:
        raise ValueError("Frozen dataset is empty.")

    for index, payload in enumerate(payloads):
        if not isinstance(payload, dict):
            raise ValueError(
                f"Payload {index} is not an object."
            )

        required = {
            "query",
            "candidates",
        }

        if not required.issubset(payload):
            raise ValueError(
                f"Payload {index} is missing required fields."
            )

    return payloads


def generate_schedule(
    request_count,
    arrival_rate,
    seed,
):
    rng = random.Random(seed)

    offsets = []
    elapsed = 0.0

    for index in range(request_count):
        if index > 0:
            elapsed += rng.expovariate(arrival_rate)

        offsets.append(elapsed)

    return offsets


def send_request(
    base_url,
    endpoint,
    payload,
    request_id,
    scheduled_at,
    experiment_start,
    timeout,
):
    now = time.perf_counter()

    if now < scheduled_at:
        time.sleep(scheduled_at - now)

    dispatch_at = time.perf_counter()

    body = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        base_url + endpoint,
        data=body,
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    status_code = None
    result = None
    error = None

    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout,
        ) as response:
            status_code = response.status
            result = json.load(response)

    except urllib.error.HTTPError as exc:
        status_code = exc.code

        try:
            error = exc.read().decode(
                "utf-8",
                errors="replace",
            )[:1000]
        except Exception:
            error = str(exc)

    except Exception as exc:
        error = (
            f"{type(exc).__name__}: {exc}"
        )

    completed_at = time.perf_counter()

    return {
        "request_id": request_id,
        "scheduled_offset_ms": (
            scheduled_at - experiment_start
        ) * 1000,
        "dispatch_offset_ms": (
            dispatch_at - experiment_start
        ) * 1000,
        "completed_offset_ms": (
            completed_at - experiment_start
        ) * 1000,
        "dispatch_lag_ms": (
            dispatch_at - scheduled_at
        ) * 1000,
        "http_latency_ms": (
            completed_at - dispatch_at
        ) * 1000,
        "scheduled_latency_ms": (
            completed_at - scheduled_at
        ) * 1000,
        "status_code": status_code,
        "success": status_code == 200,
        "error": error,
        "candidate_count": len(
            payload["candidates"]
        ),
        "profile": (
            {
                key: result.get(key)
                for key in (
                    "batch_id",
                    "batch_request_count",
                    "batch_pair_count",
                    "batch_queue_wait_ms",
                    "batch_backlog_wait_ms",
                    "batch_formation_wait_ms",
                    "batch_inference_ms",
                    "token_counting_ms",
                    "batch_token_count",
                    "batch_token_utilization",
                )
            }
            if isinstance(result, dict)
            and status_code == 200
            else None
        ),
    }


def benchmark(
    base_url,
    endpoint,
    payloads,
    arrival_rate,
    request_count,
    workers,
    seed,
    timeout,
):
    offsets = generate_schedule(
        request_count,
        arrival_rate,
        seed,
    )

    rng = random.Random(seed + 1)

    selected_payloads = [
        payloads[rng.randrange(len(payloads))]
        for _ in range(request_count)
    ]

    started_at_utc = (
        datetime.now(timezone.utc).isoformat()
    )

    experiment_start = time.perf_counter()

    records = []

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=workers
    ) as executor:
        futures = []

        for index, (offset, payload) in enumerate(
            zip(offsets, selected_payloads)
        ):
            scheduled_at = (
                experiment_start + offset
            )

            remaining = (
                scheduled_at - time.perf_counter()
            )

            if remaining > 0:
                time.sleep(remaining)

            futures.append(
                executor.submit(
                    send_request,
                    base_url,
                    endpoint,
                    payload,
                    index,
                    scheduled_at,
                    experiment_start,
                    timeout,
                )
            )

        for future in concurrent.futures.as_completed(
            futures
        ):
            records.append(future.result())

    experiment_end = time.perf_counter()

    records.sort(
        key=lambda record: record["request_id"]
    )

    duration_seconds = (
        experiment_end - experiment_start
    )

    schedule_duration = (
        offsets[-1] if offsets else 0.0
    )

    successes = [
        record
        for record in records
        if record["success"]
    ]

    status_counts = Counter(
        str(record["status_code"])
        if record["status_code"] is not None
        else "transport_error"
        for record in records
    )

    dispatch_lags = [
        record["dispatch_lag_ms"]
        for record in records
    ]

    http_latencies = [
        record["http_latency_ms"]
        for record in records
    ]

    scheduled_latencies = [
        record["scheduled_latency_ms"]
        for record in records
    ]

    successful_http_latencies = [
        record["http_latency_ms"]
        for record in successes
    ]

    summary = {
        "arrival_rate_rps": arrival_rate,
        "request_count": request_count,
        "success_count": len(successes),
        "failure_count": (
            request_count - len(successes)
        ),
        "success_rate": (
            len(successes) / request_count
        ),
        "status_counts": dict(status_counts),
        "schedule_duration_seconds": (
            schedule_duration
        ),
        "experiment_duration_seconds": (
            duration_seconds
        ),
        "completion_throughput_rps": (
            request_count / duration_seconds
        ),
        "goodput_rps": (
            len(successes) / duration_seconds
        ),
        "dispatch_lag_ms": summarize(
            dispatch_lags
        ),
        "http_latency_ms_all": summarize(
            http_latencies
        ),
        "scheduled_latency_ms_all": summarize(
            scheduled_latencies
        ),
        "http_latency_ms_success": summarize(
            successful_http_latencies
        ),
        "total_successful_pairs": sum(
            record["candidate_count"]
            for record in successes
        ),
        "started_at_utc": started_at_utc,
        "finished_at_utc": (
            datetime.now(timezone.utc).isoformat()
        ),
    }

    profiles = [
        record["profile"]
        for record in successes
        if record["profile"] is not None
    ]

    if profiles:
        profile_metrics = (
            "batch_request_count",
            "batch_pair_count",
            "batch_queue_wait_ms",
            "batch_backlog_wait_ms",
            "batch_formation_wait_ms",
            "batch_inference_ms",
            "token_counting_ms",
            "batch_token_count",
            "batch_token_utilization",
        )

        summary["server_metrics"] = {}

        for metric in profile_metrics:
            values = [
                profile[metric]
                for profile in profiles
                if isinstance(
                    profile.get(metric),
                    (int, float),
                )
            ]

            summary["server_metrics"][metric] = (
                summarize(values)
            )

    return {
        "summary": summary,
        "records": records,
    }


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
        "--pipeline",
        choices=ENDPOINTS,
        default="batched",
    )

    parser.add_argument(
        "--arrival-rate",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--requests",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=60,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    if args.arrival_rate <= 0:
        parser.error(
            "--arrival-rate must be positive"
        )

    if args.requests <= 0:
        parser.error(
            "--requests must be positive"
        )

    if args.workers <= 0:
        parser.error(
            "--workers must be positive"
        )

    dataset_path = Path(args.dataset)

    dataset_bytes = dataset_path.read_bytes()

    payloads = load_dataset(dataset_path)

    result = benchmark(
        base_url=args.base_url,
        endpoint=ENDPOINTS[args.pipeline],
        payloads=payloads,
        arrival_rate=args.arrival_rate,
        request_count=args.requests,
        workers=args.workers,
        seed=args.seed,
        timeout=args.timeout,
    )

    output = {
        "created_at": (
            datetime.now(timezone.utc).isoformat()
        ),
        "dataset_sha256": hashlib.sha256(
            dataset_bytes
        ).hexdigest(),
        "configuration": vars(args),
        **result,
    }

    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(output, indent=2)
    )

    summary = result["summary"]

    print(
        f"Offered Rate: "
        f"{args.arrival_rate:.2f} RPS"
    )

    print(
        f"Success: "
        f"{summary['success_count']}/"
        f"{summary['request_count']}"
    )

    print(
        f"Goodput: "
        f"{summary['goodput_rps']:.3f} RPS"
    )

    print(
        f"HTTP p95: "
        f"{summary['http_latency_ms_all']['p95']:.3f} ms"
    )

    print(
        f"Scheduled p95: "
        f"{summary['scheduled_latency_ms_all']['p95']:.3f} ms"
    )

    print(
        f"Dispatch Lag p95: "
        f"{summary['dispatch_lag_ms']['p95']:.3f} ms"
    )

    print("Saved:", output_path)


if __name__ == "__main__":
    main()
