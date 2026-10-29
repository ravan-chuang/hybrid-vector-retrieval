import csv
from pathlib import Path

import random
import statistics
import time

import httpx


URL = "http://127.0.0.1:8001/search/hybrid"

QUERIES = [
    "machine learning",
    "heart disease",
    "database indexing",
    "climate change",
    "artificial intelligence",
    "history of computers",
    "neural networks",
    "information retrieval",
]

BACKENDS = [
    "cpp_bm25",
    "postgresql_fts",
]

WARMUP = 2
REPEATS = 5


def percentile(values, p):
    values = sorted(values)
    index = round((len(values) - 1) * p)
    return values[index]


def request_search(client, query, backend):
    payload = {
        "query": query,
        "corpus": "benchmark",
        "lexical_backend": backend,
        "top_k": 10,
        "candidate_k": 50,
        "ef_search": 100,
        "rrf_k": 60,
        "lexical_weight": 0.5,
        "dense_weight": 0.5,
    }

    start = time.perf_counter()

    response = client.post(URL, json=payload)
    response.raise_for_status()

    http_ms = (time.perf_counter() - start) * 1000

    data = response.json()
    data["http_ms"] = http_ms

    return data


samples = {
    backend: []
    for backend in BACKENDS
}

with httpx.Client(timeout=120) as client:

    print("Warming up...")

    for query in QUERIES:
        for backend in BACKENDS:
            for _ in range(WARMUP):
                request_search(client, query, backend)

    print("Benchmarking...")

    tasks = [
        (query, backend)
        for _ in range(REPEATS)
        for query in QUERIES
        for backend in BACKENDS
    ]

    random.Random(42).shuffle(tasks)

    for i, (query, backend) in enumerate(tasks, 1):
        result = request_search(client, query, backend)
        samples[backend].append(result)

        print(
            f"{i:03d}/{len(tasks)} "
            f"{backend:15s} "
            f"{query:25s} "
            f"{result['total_ms']:.2f} ms"
        )


METRICS = [
    "embedding_ms",
    "lexical_ms",
    "vector_ms",
    "fusion_ms",
    "total_ms",
    "http_ms",
]

print("\n=== Benchmark Results ===")

for backend in BACKENDS:

    print(f"\nBackend: {backend}")
    print(f"Samples: {len(samples[backend])}")

    for metric in METRICS:

        values = [
            sample[metric]
            for sample in samples[backend]
        ]

        print(
            f"{metric:15s} "
            f"P50={percentile(values, 0.50):9.3f} ms "
            f"P95={percentile(values, 0.95):9.3f} ms"
        )


# Save raw benchmark measurements.
output_dir = Path("benchmarks/results")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "hybrid_api_v063.csv"

fieldnames = [
    "query",
    "backend",
    "embedding_ms",
    "lexical_ms",
    "vector_ms",
    "fusion_ms",
    "total_ms",
    "http_ms",
]

with output_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

    for backend, backend_samples in samples.items():
        for sample in backend_samples:
            writer.writerow({
                "query": sample["query"],
                "backend": backend,
                "embedding_ms": sample["embedding_ms"],
                "lexical_ms": sample["lexical_ms"],
                "vector_ms": sample["vector_ms"],
                "fusion_ms": sample["fusion_ms"],
                "total_ms": sample["total_ms"],
                "http_ms": round(sample["http_ms"], 3),
            })

print(f"Benchmark CSV saved: {output_path}")
