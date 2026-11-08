"""Fixed-duration Poisson open-loop HTTP benchmark.

Warmup and measurement are separate phases.
Measurement requests are attributed by scheduled arrival time.
All submitted measurement requests are accounted for after drain.
"""

import argparse
import concurrent.futures
import hashlib
import json
import random
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from benchmark_open_loop_http import (
    ENDPOINTS,
    load_dataset,
    send_request,
    summarize,
)


def generate_duration_schedule(duration, arrival_rate, seed):
    if duration <= 0:
        raise ValueError("Duration must be positive.")
    if arrival_rate <= 0:
        raise ValueError("Arrival rate must be positive.")

    rng = random.Random(seed)
    offsets = []
    elapsed = 0.0

    while True:
        elapsed += rng.expovariate(arrival_rate)

        if elapsed >= duration:
            break

        offsets.append(elapsed)

    return offsets


def run_phase(
    *,
    phase,
    base_url,
    endpoint,
    payloads,
    arrival_rate,
    duration,
    workers,
    seed,
    timeout,
):
    offsets = generate_duration_schedule(
        duration,
        arrival_rate,
        seed,
    )

    rng = random.Random(seed + 1)

    selected_payloads = [
        payloads[rng.randrange(len(payloads))]
        for _ in offsets
    ]

    records = []
    futures = []

    phase_start = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=workers
    ) as executor:

        for index, (offset, payload) in enumerate(
            zip(offsets, selected_payloads)
        ):
            scheduled_at = phase_start + offset

            remaining = scheduled_at - time.perf_counter()

            if remaining > 0:
                time.sleep(remaining)

            future = executor.submit(
                send_request,
                base_url,
                endpoint,
                payload,
                index,
                scheduled_at,
                phase_start,
                timeout,
            )

            futures.append(future)

        # Keep the phase window fixed even when the
        # final scheduled arrival occurs earlier.
        remaining = (
            phase_start + duration - time.perf_counter()
        )

        if remaining > 0:
            time.sleep(remaining)

        arrivals_end = time.perf_counter()

        for future in concurrent.futures.as_completed(
            futures
        ):
            records.append(future.result())

    phase_end = time.perf_counter()

    records.sort(key=lambda record: record["request_id"])

    for record in records:
        record["phase"] = phase

    return {
        "phase": phase,
        "duration_seconds": duration,
        "scheduled_count": len(offsets),
        "arrivals_elapsed_seconds": (
            arrivals_end - phase_start
        ),
        "drain_seconds": max(
            0.0,
            phase_end - arrivals_end,
        ),
        "records": records,
    }


def summarize_measurement(phase_result, arrival_rate):
    records = phase_result["records"]

    successful = [
        record
        for record in records
        if record["success"]
    ]

    failures = [
        record
        for record in records
        if not record["success"]
    ]

    duration = phase_result["duration_seconds"]

    statuses = Counter(
        str(record["status_code"])
        if record["status_code"] is not None
        else "transport_error"
        for record in records
    )

    profiles = [
        record["profile"]
        for record in successful
        if isinstance(record.get("profile"), dict)
    ]

    profile_metrics = {}

    for key in (
        "batch_request_count",
        "batch_pair_count",
        "batch_queue_wait_ms",
        "batch_backlog_wait_ms",
        "batch_formation_wait_ms",
        "batch_inference_ms",
        "token_counting_ms",
        "batch_token_count",
        "batch_token_utilization",
    ):
        values = [
            profile[key]
            for profile in profiles
            if isinstance(profile.get(key), (int, float))
        ]

        profile_metrics[key] = summarize(values)

    return {
        "configured_arrival_rate_rps": arrival_rate,
        "measurement_seconds": duration,
        "scheduled_request_count": len(records),
        "success_count": len(successful),
        "failure_count": len(failures),
        "status_counts": dict(statuses),
        "generated_arrival_rate_rps": (
            len(records) / duration
        ),
        "successes_per_measurement_second": (
            len(successful) / duration
        ),
        "http_latency_ms": summarize([
            record["http_latency_ms"]
            for record in records
        ]),
        "scheduled_latency_ms": summarize([
            record["scheduled_latency_ms"]
            for record in records
        ]),
        "successful_http_latency_ms": summarize([
            record["http_latency_ms"]
            for record in successful
        ]),
        "dispatch_lag_ms": summarize([
            record["dispatch_lag_ms"]
            for record in records
        ]),
        "max_dispatch_lag_ms": max(
            (
                record["dispatch_lag_ms"]
                for record in records
            ),
            default=None,
        ),
        "drain_seconds": phase_result["drain_seconds"],
        "server_profile": profile_metrics,
    }



def calculate_capacity_metrics(records, duration_seconds):
    """Calculate client-side in-flight and completion metrics."""

    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")

    duration_ms = duration_seconds * 1000

    events = []

    for record in records:
        dispatch = record["dispatch_offset_ms"]
        completion = record["completed_offset_ms"]

        if completion < dispatch:
            raise ValueError(
                "Completion timestamp precedes dispatch"
            )

        events.append((dispatch, 1))
        events.append((completion, -1))

    # Process completions before dispatches at equal timestamps.
    events.sort(key=lambda event: (event[0], event[1]))

    inflight = 0
    max_inflight = 0

    for timestamp, change in events:
        if timestamp >= duration_ms:
            break

        inflight += change
        max_inflight = max(max_inflight, inflight)

    completed_in_window = sum(
        record["completed_offset_ms"] < duration_ms
        for record in records
    )

    successful_in_window = sum(
        record["success"]
        and record["completed_offset_ms"] < duration_ms
        for record in records
    )

    completed_in_drain = (
        len(records) - completed_in_window
    )

    if inflight != completed_in_drain:
        raise ValueError(
            "In-flight accounting mismatch"
        )

    return {
        "max_inflight": max_inflight,
        "inflight_at_window_end": inflight,
        "completed_in_window": completed_in_window,
        "completed_in_drain": completed_in_drain,
        "completion_throughput_rps": (
            completed_in_window / duration_seconds
        ),
        "window_goodput_rps": (
            successful_in_window / duration_seconds
        ),
    }

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "reports/performance/cross_encoder/"
            "datasets/msmarco_frozen_k50.json"
        ),
    )
    parser.add_argument(
        "--pipeline",
        choices=sorted(ENDPOINTS),
        default="batched",
    )
    parser.add_argument(
        "--arrival-rate",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=30,
    )
    parser.add_argument(
        "--measurement-seconds",
        type=float,
        default=120,
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
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    if args.arrival_rate <= 0:
        parser.error("--arrival-rate must be positive")

    if args.warmup_seconds <= 0:
        parser.error("--warmup-seconds must be positive")

    if args.measurement_seconds <= 0:
        parser.error("--measurement-seconds must be positive")

    if args.workers < 1:
        parser.error("--workers must be >= 1")

    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    payloads = load_dataset(args.dataset)

    dataset_sha256 = hashlib.sha256(
        args.dataset.read_bytes()
    ).hexdigest()

    started_at = datetime.now(
        timezone.utc
    ).isoformat()

    print("=== OPEN-LOOP DURATION BENCHMARK ===")
    print("Pipeline:", args.pipeline)
    print("Arrival Rate:", args.arrival_rate)
    print("Warmup:", args.warmup_seconds, "s")
    print("Measurement:", args.measurement_seconds, "s")
    print("Workers:", args.workers)

    print("\n[1/2] Warmup")

    warmup = run_phase(
        phase="warmup",
        base_url=args.base_url,
        endpoint=ENDPOINTS[args.pipeline],
        payloads=payloads,
        arrival_rate=args.arrival_rate,
        duration=args.warmup_seconds,
        workers=args.workers,
        seed=args.seed,
        timeout=args.timeout,
    )

    print(
        "Warmup requests:",
        warmup["scheduled_count"],
    )

    print("\n[2/2] Measurement")

    measurement = run_phase(
        phase="measurement",
        base_url=args.base_url,
        endpoint=ENDPOINTS[args.pipeline],
        payloads=payloads,
        arrival_rate=args.arrival_rate,
        duration=args.measurement_seconds,
        workers=args.workers,
        seed=args.seed + 1000,
        timeout=args.timeout,
    )

    summary = summarize_measurement(
        measurement,
        args.arrival_rate,
    )

    summary["capacity"] = calculate_capacity_metrics(
        measurement["records"],
        args.measurement_seconds,
    )

    result = {
        "benchmark_version": "duration_v1",
        "started_at_utc": started_at,
        "dataset": str(args.dataset),
        "dataset_sha256": dataset_sha256,
        "configuration": {
            "base_url": args.base_url,
            "pipeline": args.pipeline,
            "arrival_rate_rps": args.arrival_rate,
            "warmup_seconds": args.warmup_seconds,
            "measurement_seconds": (
                args.measurement_seconds
            ),
            "workers": args.workers,
            "seed": args.seed,
            "timeout_seconds": args.timeout,
        },
        "warmup": {
            key: value
            for key, value in warmup.items()
            if key != "records"
        },
        "measurement": {
            key: value
            for key, value in measurement.items()
            if key != "records"
        },
        "summary": summary,
        "records": measurement["records"],
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )

    print("\n=== RESULT ===")
    print(
        "Requests:",
        summary["scheduled_request_count"],
    )
    print(
        "Success:",
        summary["success_count"],
    )
    print(
        "Failure:",
        summary["failure_count"],
    )
    print(
        "Generated RPS:",
        round(
            summary["generated_arrival_rate_rps"],
            3,
        ),
    )
    print(
        "HTTP p95:",
        summary["http_latency_ms"]["p95"],
    )
    print(
        "Dispatch Lag p95:",
        summary["dispatch_lag_ms"]["p95"],
    )
    print(
        "Drain:",
        round(summary["drain_seconds"], 3),
        "s",
    )
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
