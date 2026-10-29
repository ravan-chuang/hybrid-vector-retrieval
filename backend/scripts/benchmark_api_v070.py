import json
import random
import statistics
import time
import urllib.request
from pathlib import Path

BASE = Path("results/msmarco_full/latency/api_v070")
BASE.mkdir(parents=True, exist_ok=True)

URL = "http://127.0.0.1:8000/search/rerank"

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

PIPELINES = {
    "pg_fts": {
        "lexical_backend": "postgres_fts",
        "stopword_filter": False,
    },
    "cpp_bm25": {
        "lexical_backend": "cpp_bm25",
        "stopword_filter": False,
    },
    "cpp_bm25_stopwords": {
        "lexical_backend": "cpp_bm25",
        "stopword_filter": True,
    },
}


def call_api(query, config):
    payload = {
        "query": query,
        "corpus": "benchmark",
        "top_k": 10,
        "candidate_k": 50,
        "ef_search": 160,
        **config,
    }

    request = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()

    with urllib.request.urlopen(
        request,
        timeout=90,
    ) as response:
        data = json.load(response)

    client_ms = (time.perf_counter() - start) * 1000

    if len(data["results"]) != 10:
        raise RuntimeError(
            f"Expected 10 results, got {len(data['results'])}"
        )

    return {
        "client_ms": client_ms,
        "total_ms": data["total_ms"],
        "embedding_ms": data["embedding_ms"],
        "lexical_ms": data["lexical_ms"],
        "vector_ms": data["vector_ms"],
        "rerank_ms": data["rerank_ms"],
        "candidate_count": data["candidate_count"],
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


def main():
    rows = []

    print("Warming up...")

    for name, config in PIPELINES.items():
        result = call_api(QUERIES[0], config)
        print(name, round(result["total_ms"], 2), "ms")

    jobs = [
        (name, query, repeat)
        for repeat in range(2)
        for query in QUERIES
        for name in PIPELINES
    ]

    random.Random(42).shuffle(jobs)

    for index, (name, query, repeat) in enumerate(jobs, 1):
        result = call_api(query, PIPELINES[name])

        row = {
            "pipeline": name,
            "query": query,
            "repeat": repeat,
            **result,
        }

        rows.append(row)

        print(
            f"[{index:02d}/{len(jobs)}] "
            f"{name:20s} "
            f"{result['total_ms']:9.2f} ms "
            f"CE={result['rerank_ms']:8.2f} ms"
        )

    output = BASE / "per_request.json"

    output.write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 75)
    print("END-TO-END LATENCY SUMMARY")
    print("=" * 75)

    for name in PIPELINES:
        selected = [
            row for row in rows
            if row["pipeline"] == name
        ]

        values = [
            row["total_ms"]
            for row in selected
        ]

        mean = statistics.mean(values)

        print(f"\n{name}")
        print(f"  N       : {len(values)}")
        print(f"  Mean    : {mean:.3f} ms")
        print(f"  P50     : {percentile(values, 50):.3f} ms")
        print(f"  P95     : {percentile(values, 95):.3f} ms")
        print(f"  P99     : {percentile(values, 99):.3f} ms")
        print(f"  QPS     : {1000 / mean:.3f}")

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
