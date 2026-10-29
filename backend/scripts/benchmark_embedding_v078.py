"""v0.7.8 CPU embedding concurrency benchmark."""

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

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


def percentile(values, p):
    values = sorted(values)

    if not values:
        return None

    position = (len(values) - 1) * p / 100
    low = int(position)
    high = min(low + 1, len(values) - 1)
    weight = position - low

    return (
        values[low] * (1 - weight)
        + values[high] * weight
    )


def encode_once(model, query):
    start = time.perf_counter()

    vector = model.encode(
        query,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000

    return elapsed_ms, np.asarray(vector)


def run_level(model, concurrency, requests_count):
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                encode_once,
                model,
                QUERIES[i % len(QUERIES)],
            )
            for i in range(requests_count)
        ]

        results = [
            future.result()
            for future in as_completed(futures)
        ]

    elapsed = time.perf_counter() - start

    latencies = [item[0] for item in results]

    return {
        "concurrency": concurrency,
        "requests": requests_count,
        "elapsed_seconds": elapsed,
        "qps": requests_count / elapsed,
        "mean_ms": statistics.mean(latencies),
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "p99_ms": percentile(latencies, 99),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=[1, 4, 8, 16],
    )
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    if args.threads < 1 or args.requests < 1 or args.warmup < 0:
        parser.error("Invalid threads, requests, or warmup")

    if any(c < 1 for c in args.concurrency):
        parser.error("Concurrency values must be >= 1")

    torch.set_num_threads(args.threads)

    model = SentenceTransformer(MODEL_NAME, device="cpu")

    for i in range(args.warmup):
        encode_once(model, QUERIES[i % len(QUERIES)])

    reference_vectors = {
        query: encode_once(model, query)[1]
        for query in QUERIES
    }

    summaries = []

    print(f"\n=== CPU Threads: {args.threads} ===")
    print("Actual intra-op threads:", torch.get_num_threads())

    for concurrency in args.concurrency:
        result = run_level(
            model,
            concurrency,
            args.requests,
        )

        summaries.append(result)

        print(
            f"C={concurrency:2d} "
            f"QPS={result['qps']:8.2f} "
            f"Mean={result['mean_ms']:8.2f}ms "
            f"P95={result['p95_ms']:8.2f}ms"
        )

    # Repeat embeddings to check numerical stability
    max_abs_diff = 0.0

    for query, reference in reference_vectors.items():
        repeated = encode_once(model, query)[1]
        diff = float(
            np.max(np.abs(reference - repeated))
        )
        max_abs_diff = max(max_abs_diff, diff)

    report = {
        "version": "v0.7.8",
        "model": MODEL_NAME,
        "device": "cpu",
        "threads": args.threads,
        "requests_per_level": args.requests,
        "warmup": args.warmup,
        "max_abs_repeat_diff": max_abs_diff,
        "summary": summaries,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("Max repeated embedding difference:", max_abs_diff)
    print("Saved:", output)


if __name__ == "__main__":
    main()
