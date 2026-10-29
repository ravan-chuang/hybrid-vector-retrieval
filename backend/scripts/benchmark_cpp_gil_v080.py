"""v0.8.0 C++ BM25 GIL A/B benchmark."""

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path


QUERIES = [
    "nature",
    "science",
    "history",
    "machine learning",
    "Manhattan Project",
    "heart disease",
    "database indexing",
]


def percentile(values, p):
    values = sorted(values)
    position = (len(values) - 1) * p / 100
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower

    return (
        values[lower] * (1 - fraction)
        + values[upper] * fraction
    )


def load_extension(path):
    spec = importlib.util.spec_from_file_location(
        "retrieval_cpp",
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load extension: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--extension", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument(
        "--corpus",
        default="cpp/data/msmarco_full.bin",
    )
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    if args.warmup < 0 or args.repeats < 1:
        parser.error("Invalid warmup or repeats")

    module = load_extension(
        str(Path(args.extension).resolve())
    )

    engine = module.SearchEngine()

    print(f"Version: {args.label}", flush=True)
    print("Loading corpus...", flush=True)

    start = time.perf_counter()
    loaded = engine.load_corpus(args.corpus)
    load_seconds = time.perf_counter() - start

    print(
        f"Loaded {loaded:,} documents "
        f"in {load_seconds:.2f}s",
        flush=True,
    )

    results = []

    for query in QUERIES:
        for _ in range(args.warmup):
            engine.search_dense(query, 10)

        samples = []

        for _ in range(args.repeats):
            start = time.perf_counter()
            engine.search_dense(query, 10)

            samples.append(
                (time.perf_counter() - start) * 1000
            )

        row = {
            "query": query,
            "mean_ms": statistics.mean(samples),
            "p50_ms": percentile(samples, 50),
            "p95_ms": percentile(samples, 95),
            "p99_ms": percentile(samples, 99),
        }

        results.append(row)

        print(
            f"{query:20s} "
            f"Mean={row['mean_ms']:.3f}ms "
            f"P50={row['p50_ms']:.3f}ms "
            f"P95={row['p95_ms']:.3f}ms",
            flush=True,
        )

    report = {
        "version": args.label,
        "documents": loaded,
        "load_seconds": load_seconds,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "results": results,
    }

    output = Path(args.output)

    if output.exists():
        raise SystemExit(
            f"Refusing to overwrite existing result: {output}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("Saved:", output)


if __name__ == "__main__":
    main()
