"""v0.7.9 Retrieval bottleneck profiling benchmark."""

import argparse
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
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

_thread_local = threading.local()

METRICS = [
    "client_ms",
    "total_ms",
    "embedding_ms",
    "lexical_ms",
    "vector_ms",
    "fusion_ms",
    "pool_wait_ms",
    "bm25_search_ms",
    "metadata_fetch_ms",
]


def session():
    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session()
    return _thread_local.session


def percentile(values, p):
    if not values:
        return None

    values = sorted(values)
    pos = (len(values) - 1) * p / 100

    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    fraction = pos - lo

    return values[lo] * (1 - fraction) + values[hi] * fraction


def request_once(base_url, query, timeout):
    payload = {
        "query": query,
        "corpus": "benchmark",
        "lexical_backend": "cpp_bm25",
        "stopword_filter": True,
        "top_k": 10,
        "candidate_k": 50,
        "ef_search": 160,
    }

    started = time.perf_counter()

    try:
        response = session().post(
            base_url.rstrip("/") + "/search/hybrid",
            json=payload,
            timeout=timeout,
        )

        client_ms = (
            time.perf_counter() - started
        ) * 1000

        data = response.json()

        row = {
            "query": query,
            "success": response.status_code == 200,
            "status_code": response.status_code,
            "client_ms": client_ms,
            "error": None,
        }

        if response.status_code != 200:
            row["error"] = str(data.get("detail", data))
            return row

        for metric in METRICS:
            if metric != "client_ms":
                row[metric] = data.get(metric)

        row["result_ids"] = [
            item["document_id"]
            for item in data.get("results", [])
        ]

        return row

    except Exception as exc:
        return {
            "query": query,
            "success": False,
            "status_code": None,
            "client_ms": (
                time.perf_counter() - started
            ) * 1000,
            "error": str(exc),
        }


def summarize(rows, concurrency, elapsed):
    successful = [r for r in rows if r["success"]]

    result = {
        "concurrency": concurrency,
        "requests": len(rows),
        "successes": len(successful),
        "failures": len(rows) - len(successful),
        "elapsed_seconds": elapsed,
        "successful_qps": (
            len(successful) / elapsed
            if elapsed > 0 else None
        ),
    }

    for metric in METRICS:
        values = [
            float(row[metric])
            for row in successful
            if isinstance(row.get(metric), (int, float))
        ]

        result[metric] = {
            "mean": statistics.mean(values) if values else None,
            "p50": percentile(values, 50),
            "p95": percentile(values, 95),
            "p99": percentile(values, 99),
        }

    return result


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
        "--warmup",
        type=int,
        default=10,
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
            "api_v079/retrieval_profiling"
        ),
    )

    args = parser.parse_args()

    if args.requests < 1 or args.warmup < 0:
        parser.error("--requests must be >= 1 and --warmup >= 0")

    if not args.concurrency or any(
        c < 1 for c in args.concurrency
    ):
        parser.error("--concurrency values must be >= 1")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    all_rows = []
    summaries = []

    print("v0.7.9 Retrieval Bottleneck Profiling")
    print("Endpoint: /search/hybrid")
    print("Warmup:", args.warmup)
    print("Requests per level:", args.requests)

    for concurrency in args.concurrency:
        print(f"\nC={concurrency}: warming up...")

        for i in range(args.warmup):
            row = request_once(
                args.base_url,
                QUERIES[i % len(QUERIES)],
                args.timeout,
            )

            if not row["success"]:
                raise RuntimeError(
                    f"Warmup failed at C={concurrency}: "
                    f"{row['error']}"
                )

        print(f"C={concurrency}: measuring...")

        started = time.perf_counter()

        with ThreadPoolExecutor(
            max_workers=concurrency
        ) as pool:
            futures = [
                pool.submit(
                    request_once,
                    args.base_url,
                    QUERIES[i % len(QUERIES)],
                    args.timeout,
                )
                for i in range(args.requests)
            ]

            rows = [
                future.result()
                for future in as_completed(futures)
            ]

        elapsed = time.perf_counter() - started

        for row in rows:
            row["concurrency"] = concurrency

        all_rows.extend(rows)

        summary = summarize(
            rows,
            concurrency,
            elapsed,
        )
        summaries.append(summary)

        print(
            f"C={concurrency:2d} "
            f"QPS={summary['successful_qps']:.2f} "
            f"success={summary['successes']}/"
            f"{summary['requests']} "
            f"client_p95="
            f"{summary['client_ms']['p95']:.2f}ms"
        )

    report = {
        "version": "v0.7.9",
        "benchmark": "retrieval_bottleneck_profiling",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": "/search/hybrid",
        "configuration": {
            "corpus": "benchmark",
            "lexical_backend": "cpp_bm25",
            "stopword_filter": True,
            "top_k": 10,
            "candidate_k": 50,
            "ef_search": 160,
            "requests_per_level": args.requests,
            "warmup_per_level": args.warmup,
            "concurrency": args.concurrency,
        },
        "summary": summaries,
    }

    (output / "summary.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    (output / "per_request.json").write_text(
        json.dumps(all_rows, indent=2),
        encoding="utf-8",
    )

    print("\nSaved to:", output)


if __name__ == "__main__":
    main()
