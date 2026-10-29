"""v0.7.2: Concurrent HTTP load benchmark for Hybrid Retrieval."""

import argparse
import json
import math
import random
import statistics
import threading
import time

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests


API_URL = "http://127.0.0.1:8000/search/rerank"

BASE = Path("results/msmarco_full/latency/api_v073")

QUERIES = [
    "what is information retrieval",
    "how does machine learning work",
    "what causes earthquakes",
    "definition of neural networks",
    "how does photosynthesis work",
    "what is database indexing",
    "how do search engines rank pages",
    "what is natural language processing",
    "how does cloud computing work",
    "what is distributed computing",
]

_thread_local = threading.local()


def get_session():
    session = getattr(_thread_local, "session", None)

    if session is None:
        session = requests.Session()
        _thread_local.session = session

    return session


def percentile(values, p):
    if not values:
        return None

    values = sorted(values)

    position = (len(values) - 1) * p / 100

    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return values[lower]

    fraction = position - lower

    return (
        values[lower] * (1 - fraction)
        + values[upper] * fraction
    )


def build_payload(query):
    return {
        "query": query,
        "corpus": "benchmark",
        "lexical_backend": "cpp_bm25",
        "stopword_filter": True,
        "top_k": 10,
        "candidate_k": 50,
        "ef_search": 160,
    }


def send_request(request_id, query, timeout):
    payload = build_payload(query)

    start = time.perf_counter()

    try:
        response = get_session().post(
            API_URL,
            json=payload,
            timeout=timeout,
        )

        # Include full HTTP response transfer and JSON parsing.
        data = response.json()

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000.0

        if response.status_code != 200:
            return {
                "request_id": request_id,
                "query": query,
                "success": False,
                "status_code": response.status_code,
                "client_ms": elapsed_ms,
                "error": str(data)[:500],
            }

        return {
            "request_id": request_id,
            "query": query,
            "success": True,
            "status_code": response.status_code,
            "client_ms": elapsed_ms,
            "server_ms": data.get("total_ms"),
            "embedding_ms": data.get("embedding_ms"),
            "lexical_ms": data.get("lexical_ms"),
            "vector_ms": data.get("vector_ms"),
            "rerank_ms": data.get("rerank_ms"),
            "rerank_lock_wait_ms": data.get(
                "rerank_lock_wait_ms"
            ),
            "rerank_inference_ms": data.get(
                "rerank_inference_ms"
            ),
            "rerank_other_ms": data.get(
                "rerank_other_ms"
            ),
            "candidate_count": data.get("candidate_count"),
            "result_count": len(data.get("results", [])),
        }

    except Exception as exc:
        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000.0

        return {
            "request_id": request_id,
            "query": query,
            "success": False,
            "status_code": None,
            "client_ms": elapsed_ms,
            "error": repr(exc),
        }


def summarize(rows, duration_s, concurrency):
    successful = [
        row for row in rows
        if row["success"]
    ]

    failed = [
        row for row in rows
        if not row["success"]
    ]

    client_latencies = [
        row["client_ms"]
        for row in successful
    ]

    server_latencies = [
        row["server_ms"]
        for row in successful
        if row.get("server_ms") is not None
    ]

    rerank_latencies = [
        row["rerank_ms"]
        for row in successful
        if row.get("rerank_ms") is not None
    ]

    summary = {
        "concurrency": concurrency,
        "requests": len(rows),
        "successful": len(successful),
        "failed": len(failed),
        "error_rate": (
            len(failed) / len(rows)
            if rows else 0
        ),
        "duration_s": duration_s,
        "actual_qps": (
            len(successful) / duration_s
            if duration_s > 0 else 0
        ),
        "client_mean_ms": (
            statistics.mean(client_latencies)
            if client_latencies else None
        ),
        "client_p50_ms": percentile(
            client_latencies, 50
        ),
        "client_p95_ms": percentile(
            client_latencies, 95
        ),
        "client_p99_ms": percentile(
            client_latencies, 99
        ),
        "server_mean_ms": (
            statistics.mean(server_latencies)
            if server_latencies else None
        ),
        "rerank_mean_ms": (
            statistics.mean(rerank_latencies)
            if rerank_latencies else None
        ),
        "status_counts": dict(
            Counter(
                str(row["status_code"])
                for row in rows
            )
        ),
    }

    return summary


def run_level(concurrency, requests_count, timeout, seed):
    rng = random.Random(seed)

    # Every concurrency level uses the same query distribution.
    queries = [
        QUERIES[i % len(QUERIES)]
        for i in range(requests_count)
    ]

    rng.shuffle(queries)

    rows = []

    start = time.perf_counter()

    with ThreadPoolExecutor(
        max_workers=concurrency
    ) as executor:

        futures = [
            executor.submit(
                send_request,
                index,
                query,
                timeout,
            )
            for index, query in enumerate(
                queries,
                start=1,
            )
        ]

        for future in as_completed(futures):
            rows.append(future.result())

    duration_s = time.perf_counter() - start

    rows.sort(key=lambda row: row["request_id"])

    return rows, summarize(
        rows,
        duration_s,
        concurrency,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--concurrency",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16],
    )

    parser.add_argument(
        "--requests",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=120,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if args.requests < 1:
        parser.error("--requests must be >= 1")

    if any(c < 1 for c in args.concurrency):
        parser.error("concurrency must be >= 1")

    BASE.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("v0.7.2 CONCURRENT HYBRID RETRIEVAL BENCHMARK")
    print("=" * 76)

    print("Concurrency levels:", args.concurrency)
    print("Requests per level:", args.requests)
    print("Warmup requests:", args.warmup)

    print("\nWarming up...")

    for index in range(args.warmup):
        result = send_request(
            index,
            QUERIES[index % len(QUERIES)],
            args.timeout,
        )

        if not result["success"]:
            raise RuntimeError(
                f"Warmup failed: {result}"
            )

    print("Warmup complete.\n")

    all_rows = []
    summaries = []

    for concurrency in args.concurrency:
        print("-" * 76)
        print(
            f"Concurrency={concurrency} "
            f"Requests={args.requests}"
        )

        rows, summary = run_level(
            concurrency=concurrency,
            requests_count=args.requests,
            timeout=args.timeout,
            seed=args.seed,
        )

        for row in rows:
            row["concurrency"] = concurrency

        all_rows.extend(rows)
        summaries.append(summary)

        print(
            f"Success={summary['successful']}/"
            f"{summary['requests']}"
        )

        print(
            f"Actual QPS: "
            f"{summary['actual_qps']:.3f}"
        )

        if summary["client_mean_ms"] is not None:
            print(
                f"Client Mean: "
                f"{summary['client_mean_ms']:.3f} ms"
            )

            print(
                f"Client P50: "
                f"{summary['client_p50_ms']:.3f} ms"
            )

            print(
                f"Client P95: "
                f"{summary['client_p95_ms']:.3f} ms"
            )

            print(
                f"Client P99: "
                f"{summary['client_p99_ms']:.3f} ms"
            )

        if summary["rerank_mean_ms"] is not None:
            print(
                f"Rerank Mean: "
                f"{summary['rerank_mean_ms']:.3f} ms"
            )

        if summary["failed"]:
            print(
                "Status counts:",
                summary["status_counts"],
            )

        print()

    timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    metadata = {
        "version": "v0.7.2",
        "timestamp_utc": timestamp,
        "api_url": API_URL,
        "pipeline": "cpp_bm25_stopwords",
        "cross_encoder_device_expected": "mps",
        "cross_encoder_batch_size_expected": 32,
        "requests_per_level": args.requests,
        "concurrency_levels": args.concurrency,
        "warmup": args.warmup,
        "seed": args.seed,
    }

    (BASE / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    (BASE / "per_request.json").write_text(
        json.dumps(all_rows, indent=2),
        encoding="utf-8",
    )

    (BASE / "summary.json").write_text(
        json.dumps(summaries, indent=2),
        encoding="utf-8",
    )

    print("=" * 76)
    print("BENCHMARK COMPLETE")
    print("=" * 76)

    print("Results saved to:", BASE)


if __name__ == "__main__":
    main()
