"""Online concurrency benchmark: baseline vs dynamic batching."""

import argparse
import json
import random
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


QUERIES = [
    "what is information retrieval",
    "how does a search engine work",
    "what causes climate change",
    "how to treat high blood pressure",
    "what is machine learning",
    "how do neural networks learn",
    "what is the function of mitochondria",
    "how does photosynthesis work",
    "what are the symptoms of diabetes",
    "what is the difference between DNA and RNA",
]

ENDPOINTS = {
    "baseline": "/search/rerank",
    "batched": "/search/rerank/batched",
}

_thread_local = threading.local()


def get_session():
    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session()
    return _thread_local.session


def percentile(values, p):
    values = sorted(values)

    if not values:
        return None

    index = (len(values) - 1) * p / 100
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    weight = index - lower

    return values[lower] * (1 - weight) + values[upper] * weight


def run_request(base_url, endpoint, query, timeout):
    payload = {
        "query": query,
        "corpus": "benchmark",
        "lexical_backend": "cpp_bm25",
        "stopword_filter": True,
        "top_k": 10,
        "candidate_k": 50,
        "ef_search": 160,
    }

    start = time.perf_counter()

    try:
        response = get_session().post(
            base_url + endpoint,
            json=payload,
            timeout=timeout,
        )

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        data = response.json()

        return {
            "success": response.status_code == 200,
            "status_code": response.status_code,
            "client_ms": elapsed_ms,
            "server_ms": data.get("total_ms"),
            "rerank_ms": data.get("rerank_ms"),
            "batch_request_count": data.get("batch_request_count"),
            "batch_pair_count": data.get("batch_pair_count"),
            "batch_queue_wait_ms": data.get("batch_queue_wait_ms"),
            "batch_inference_ms": data.get("batch_inference_ms"),
            "error": None if response.status_code == 200
                     else str(data.get("detail")),
        }

    except Exception as exc:
        return {
            "success": False,
            "status_code": None,
            "client_ms": (
                time.perf_counter() - start
            ) * 1000,
            "error": str(exc),
        }


def run_level(
    base_url,
    endpoint,
    concurrency,
    queries,
    requests_count,
    timeout,
):
    start = time.perf_counter()

    with ThreadPoolExecutor(
        max_workers=concurrency
    ) as pool:
        futures = [
            pool.submit(
                run_request,
                base_url,
                endpoint,
                queries[i % len(queries)],
                timeout,
            )
            for i in range(requests_count)
        ]

        rows = [
            future.result()
            for future in as_completed(futures)
        ]

    elapsed = time.perf_counter() - start

    successes = [
        row for row in rows
        if row["success"]
    ]

    latencies = [
        row["client_ms"]
        for row in successes
    ]

    batch_counts = [
        row["batch_request_count"]
        for row in successes
        if row.get("batch_request_count") is not None
    ]

    summary = {
        "concurrency": concurrency,
        "requests": requests_count,
        "successes": len(successes),
        "failures": requests_count - len(successes),
        "elapsed_seconds": elapsed,
        "successful_qps": len(successes) / elapsed,
        "mean_ms": (
            statistics.mean(latencies)
            if latencies else None
        ),
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "p99_ms": percentile(latencies, 99),
        "mean_batch_requests": (
            statistics.mean(batch_counts)
            if batch_counts else None
        ),
        "max_batch_requests": (
            max(batch_counts)
            if batch_counts else None
        ),
    }

    return summary, rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16],
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30,
    )
    parser.add_argument(
        "--output",
        default=(
            "results/msmarco_full/latency/"
            "api_v075/online_concurrency"
        ),
    )

    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    random.seed(42)

    all_summaries = []
    all_rows = []

    for concurrency in args.concurrency:
        for name, endpoint in ENDPOINTS.items():
            print(
                f"\nRunning {name}, concurrency={concurrency}",
                flush=True,
            )

            summary, rows = run_level(
                args.base_url,
                endpoint,
                concurrency,
                QUERIES,
                args.requests,
                args.timeout,
            )

            summary["pipeline"] = name

            all_summaries.append(summary)

            for row in rows:
                row["pipeline"] = name
                row["concurrency"] = concurrency
                all_rows.append(row)

            print(
                f"Success={summary['successes']}/"
                f"{summary['requests']} "
                f"QPS={summary['successful_qps']:.3f} "
                f"P95={summary['p95_ms']}",
                flush=True,
            )

    (output / "summary.json").write_text(
        json.dumps(all_summaries, indent=2)
    )

    (output / "per_request.json").write_text(
        json.dumps(all_rows, indent=2)
    )

    print("\n=== SUMMARY ===")

    for row in all_summaries:
        print(
            f"{row['pipeline']:8s} "
            f"C={row['concurrency']:2d} "
            f"QPS={row['successful_qps']:.3f} "
            f"Mean={row['mean_ms']:.2f} "
            f"P95={row['p95_ms']:.2f} "
            f"Batch={row['mean_batch_requests']}"
        )


if __name__ == "__main__":
    main()
