"""Cross-request ranking parity validation for v0.7.5."""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


BASE_URL = "http://127.0.0.1:8000"

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

OUTPUT = Path("results/msmarco_full/parity/v075")
OUTPUT.mkdir(parents=True, exist_ok=True)


def make_payload(query):
    return {
        "query": query,
        "corpus": "benchmark",
        "lexical_backend": "cpp_bm25",
        "stopword_filter": True,
        "top_k": 10,
        "candidate_k": 50,
        "ef_search": 160,
    }


def request_rerank(query, endpoint):
    started = time.perf_counter()

    response = requests.post(
        BASE_URL + endpoint,
        json=make_payload(query),
        timeout=60,
    )

    response.raise_for_status()

    result = response.json()

    return {
        "query": query,
        "client_ms": round(
            (time.perf_counter() - started) * 1000,
            3,
        ),
        "server_ms": result["total_ms"],
        "candidate_count": result["candidate_count"],
        "batch_request_count": result.get(
            "batch_request_count"
        ),
        "batch_pair_count": result.get(
            "batch_pair_count"
        ),
        "batch_queue_wait_ms": result.get(
            "batch_queue_wait_ms"
        ),
        "ranking": [
            {
                "document_id": item["document_id"],
                "score": item["cross_encoder_score"],
            }
            for item in result["results"]
        ],
    }


def main():
    print("=== Baseline Sequential ===", flush=True)

    baseline = {}

    for index, query in enumerate(QUERIES, start=1):
        result = request_rerank(
            query,
            "/search/rerank",
        )

        baseline[query] = result

        print(
            f"[{index:02d}/10] "
            f"Candidates={result['candidate_count']} "
            f"Latency={result['server_ms']:.2f} ms",
            flush=True,
        )

    print("\n=== Batched Concurrent ===", flush=True)

    batched = {}

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {
            pool.submit(
                request_rerank,
                query,
                "/search/rerank/batched",
            ): query
            for query in QUERIES
        }

        for future in as_completed(futures):
            result = future.result()
            batched[result["query"]] = result

            print(
                f"Batch={result['batch_request_count']} "
                f"Pairs={result['batch_pair_count']} "
                f"Latency={result['server_ms']:.2f} ms",
                flush=True,
            )

    print("\n=== Ranking Parity ===")

    comparisons = []

    for query in QUERIES:
        left = baseline[query]
        right = batched[query]

        baseline_ids = [
            item["document_id"]
            for item in left["ranking"]
        ]

        batched_ids = [
            item["document_id"]
            for item in right["ranking"]
        ]

        comparison = {
            "query": query,
            "candidate_count_equal": (
                left["candidate_count"]
                == right["candidate_count"]
            ),
            "top1_equal": (
                baseline_ids[:1]
                == batched_ids[:1]
            ),
            "top10_set_equal": (
                set(baseline_ids)
                == set(batched_ids)
            ),
            "top10_order_equal": (
                baseline_ids == batched_ids
            ),
            "batch_request_count": (
                right["batch_request_count"]
            ),
            "batch_pair_count": (
                right["batch_pair_count"]
            ),
        }

        comparisons.append(comparison)

        status = (
            "PASS"
            if comparison["top10_order_equal"]
            else "FAIL"
        )

        print(
            f"[{status}] {query} "
            f"(batch={comparison['batch_request_count']})"
        )

    summary = {
        "queries": len(QUERIES),
        "top1_equal": sum(
            row["top1_equal"]
            for row in comparisons
        ),
        "top10_set_equal": sum(
            row["top10_set_equal"]
            for row in comparisons
        ),
        "top10_order_equal": sum(
            row["top10_order_equal"]
            for row in comparisons
        ),
        "multi_request_batches": sum(
            row["batch_request_count"] > 1
            for row in comparisons
        ),
    }

    (OUTPUT / "baseline.json").write_text(
        json.dumps(baseline, indent=2)
    )

    (OUTPUT / "batched.json").write_text(
        json.dumps(batched, indent=2)
    )

    (OUTPUT / "comparisons.json").write_text(
        json.dumps(comparisons, indent=2)
    )

    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print("\n=== SUMMARY ===")

    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
