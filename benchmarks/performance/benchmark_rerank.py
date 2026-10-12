import argparse
import csv
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


QUERIES = {
    "single_term": [
        "nature",
        "science",
        "history",
        "computer",
        "medicine",
        "education",
        "technology",
        "environment",
        "economy",
        "health",
    ],
    "multi_term": [
        "nature disaster",
        "machine learning",
        "climate change",
        "artificial intelligence",
        "database management",
        "renewable energy",
        "heart disease",
        "computer network",
        "natural language processing",
        "financial market",
    ],
    "specific": [
        "Manhattan Project",
        "Apollo 11 moon landing",
        "Albert Einstein relativity",
        "Python programming language",
        "World War II",
        "Amazon rainforest deforestation",
        "Hubble Space Telescope",
        "CRISPR gene editing",
        "PostgreSQL database indexing",
        "Mount Everest elevation",
    ],
}


def percentile(values, p):
    values = sorted(values)

    if not values:
        return None

    position = (len(values) - 1) * p / 100
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return values[lower]

    return (
        values[lower] * (upper - position)
        + values[upper] * (position - lower)
    )


def request_search(url, query, timeout):
    payload = {
        "query": query,
        "corpus": "benchmark",
        "top_k": 10,
        "candidate_k": 50,
        "ef_search": 160,
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()

    with urllib.request.urlopen(
        request,
        timeout=timeout,
    ) as response:
        result = json.load(response)

    client_ms = (time.perf_counter() - start) * 1000

    return result, client_ms


def summarize(rows, key):
    values = [
        row[key]
        for row in rows
        if row.get(key) is not None
    ]

    if not values:
        return None

    return {
        "mean": round(statistics.mean(values), 2),
        "p50": round(percentile(values, 50), 2),
        "p95": round(percentile(values, 95), 2),
        "p99": round(percentile(values, 99), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000/search/rerank",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    output_dir = Path("benchmarks/performance/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    rows = []
    errors = []

    print("\n=== Hybrid Retrieval Performance Benchmark ===")
    print(f"Endpoint: {args.url}")
    print(f"Warm-up: {args.warmup}")
    print(f"Repeats: {args.repeats}")
    print(f"Queries: {sum(len(v) for v in QUERIES.values())}")
    print()

    # Warm-up
    for i in range(args.warmup):
        try:
            result, client_ms = request_search(
                args.url,
                "machine learning",
                args.timeout,
            )

            print(
                f"[Warm-up {i + 1}] "
                f"{client_ms:.1f} ms"
            )

        except Exception as exc:
            print(f"Warm-up failed: {exc}")
            return

    print("\n--- Measurement ---")

    index = 0

    for category, queries in QUERIES.items():
        for query in queries:
            for repeat in range(args.repeats):
                index += 1

                try:
                    result, client_ms = request_search(
                        args.url,
                        query,
                        args.timeout,
                    )

                    row = {
                        "category": category,
                        "query": query,
                        "repeat": repeat + 1,
                        "embedding_ms": result.get("embedding_ms"),
                        "lexical_ms": result.get("lexical_ms"),
                        "vector_ms": result.get("vector_ms"),
                        "rerank_ms": result.get("rerank_ms"),
                        "total_ms": result.get("total_ms"),
                        "client_ms": round(client_ms, 2),
                        "candidate_count": result.get("candidate_count"),
                        "result_count": len(result.get("results", [])),
                    }

                    rows.append(row)

                    print(
                        f"[{index:03d}] "
                        f"{category:12s} "
                        f"{query[:32]:32s} "
                        f"total={row['total_ms']:9.1f} ms "
                        f"lexical={row['lexical_ms']:9.1f} ms "
                        f"rerank={row['rerank_ms']:9.1f} ms "
                        f"candidates={row['candidate_count']}"
                    )

                except Exception as exc:
                    errors.append({
                        "category": category,
                        "query": query,
                        "repeat": repeat + 1,
                        "error": str(exc),
                    })

                    print(f"[ERROR] {query}: {exc}")

    if not rows:
        print("No successful measurements.")
        return

    csv_path = output_dir / f"rerank_{timestamp}.csv"
    json_path = output_dir / f"rerank_{timestamp}.json"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    metrics = [
        "embedding_ms",
        "lexical_ms",
        "vector_ms",
        "rerank_ms",
        "total_ms",
        "client_ms",
        "candidate_count",
    ]

    summary = {
        "metadata": {
            "timestamp_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "endpoint": args.url,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "query_count": sum(
                len(v) for v in QUERIES.values()
            ),
            "successful_requests": len(rows),
            "failed_requests": len(errors),
        },
        "overall": {
            metric: summarize(rows, metric)
            for metric in metrics
        },
        "by_category": {},
        "errors": errors,
    }

    for category in QUERIES:
        subset = [
            row
            for row in rows
            if row["category"] == category
        ]

        summary["by_category"][category] = {
            metric: summarize(subset, metric)
            for metric in metrics
        }

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n=== Overall Summary ===")

    for metric in metrics:
        stats = summary["overall"][metric]

        print(
            f"{metric:18s} "
            f"mean={stats['mean']:9.2f} "
            f"p50={stats['p50']:9.2f} "
            f"p95={stats['p95']:9.2f} "
            f"p99={stats['p99']:9.2f}"
        )

    print("\n=== Category Summary ===")

    for category in QUERIES:
        stats = summary["by_category"][category]["total_ms"]

        if stats:
            print(
                f"{category:12s} "
                f"mean={stats['mean']:9.2f} "
                f"p50={stats['p50']:9.2f} "
                f"p95={stats['p95']:9.2f}"
            )

    print(f"\nCSV:  {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Errors: {len(errors)}")


if __name__ == "__main__":
    main()
