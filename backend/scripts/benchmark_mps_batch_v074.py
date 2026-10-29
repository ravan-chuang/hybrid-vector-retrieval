"""v0.7.4: MPS Cross-Encoder batch throughput benchmark."""

import argparse
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

INPUT_PATH = Path(
    "results/msmarco_full/parity/v071/fixed_candidates.json"
)

OUTPUT_DIR = Path(
    "results/msmarco_full/latency/api_v074/batch_throughput"
)


def load_pairs(path: Path):
    """Load real query-document pairs from frozen candidates."""

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        raise ValueError("Expected a list of query records.")

    pairs = []
    query_counts = []

    for record in data:
        query = record["query"]
        candidates = record["candidates"]

        query_counts.append({
            "query": query,
            "candidate_count": len(candidates),
        })

        for candidate in candidates:
            pairs.append((query, candidate["content"]))

    if not pairs:
        raise ValueError("No query-document pairs found.")

    return pairs, query_counts


def synchronize(device: str):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def benchmark_batch(
    model,
    pairs,
    batch_size,
    warmup,
    repeats,
    device,
):
    """Measure full predict() wall-clock time."""

    records = []
    reference_scores = None

    for _ in range(warmup):
        model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        synchronize(device)

    for repeat in range(repeats):
        synchronize(device)

        start = time.perf_counter()

        scores = model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        synchronize(device)

        elapsed_s = time.perf_counter() - start

        scores = np.asarray(scores, dtype=np.float64).reshape(-1)

        if len(scores) != len(pairs):
            raise RuntimeError(
                "Cross-Encoder returned an unexpected score count."
            )

        if not np.isfinite(scores).all():
            raise RuntimeError(
                "Cross-Encoder returned non-finite scores."
            )

        if reference_scores is None:
            reference_scores = scores.copy()

        max_abs_diff = float(
            np.max(np.abs(scores - reference_scores))
        )

        records.append({
            "batch_size": batch_size,
            "repeat": repeat + 1,
            "pair_count": len(pairs),
            "latency_ms": elapsed_s * 1000.0,
            "pairs_per_second": len(pairs) / elapsed_s,
            "max_abs_score_diff_from_first_repeat": max_abs_diff,
        })

    return records, reference_scores


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[8, 16, 32, 64, 128],
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "mps"],
        default="mps",
    )

    args = parser.parse_args()

    if args.warmup < 0 or args.repeats < 1:
        parser.error("--warmup must be >= 0 and --repeats >= 1")

    if any(batch_size < 1 for batch_size in args.batch_sizes):
        parser.error("All batch sizes must be positive.")

    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is not available.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pairs, query_counts = load_pairs(INPUT_PATH)

    print("=" * 76)
    print("v0.7.4 CROSS-ENCODER BATCH THROUGHPUT")
    print("=" * 76)

    print(f"Device: {args.device}")
    print(f"Model: {MODEL_NAME}")
    print(f"Queries: {len(query_counts)}")
    print(f"Pairs: {len(pairs)}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Warmup: {args.warmup}")
    print(f"Repeats: {args.repeats}")

    print("\nLoading Cross-Encoder...")

    model = CrossEncoder(
        MODEL_NAME,
        device=args.device,
        activation_fn=torch.nn.Identity(),
    )

    metadata = {
        "version": "v0.7.4",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "device": args.device,
        "torch_version": torch.__version__,
        "python_platform": platform.platform(),
        "batch_sizes": args.batch_sizes,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "query_count": len(query_counts),
        "pair_count": len(pairs),
        "input_path": str(INPUT_PATH),
        "query_counts": query_counts,
        "measurement": (
            "CrossEncoder.predict wall time including "
            "tokenization, framework overhead, and device synchronization"
        ),
    }

    all_records = []
    summaries = []
    global_reference = None

    for batch_size in args.batch_sizes:
        print("\n" + "-" * 76)
        print(f"Batch Size: {batch_size}")

        records, scores = benchmark_batch(
            model=model,
            pairs=pairs,
            batch_size=batch_size,
            warmup=args.warmup,
            repeats=args.repeats,
            device=args.device,
        )

        if global_reference is None:
            global_reference = scores.copy()

        cross_batch_max_abs_diff = float(
            np.max(np.abs(scores - global_reference))
        )

        all_records.extend(records)

        latencies = [
            record["latency_ms"]
            for record in records
        ]

        throughputs = [
            record["pairs_per_second"]
            for record in records
        ]

        summary = {
            "batch_size": batch_size,
            "pair_count": len(pairs),
            "repeats": len(records),
            "mean_latency_ms": statistics.mean(latencies),
            "median_latency_ms": statistics.median(latencies),
            "std_latency_ms": (
                statistics.stdev(latencies)
                if len(latencies) > 1
                else 0.0
            ),
            "mean_pairs_per_second": statistics.mean(throughputs),
            "median_pairs_per_second": statistics.median(throughputs),
            "max_abs_score_diff_from_first_batch": (
                cross_batch_max_abs_diff
            ),
        }

        summaries.append(summary)

        print(
            f"Mean latency: "
            f"{summary['mean_latency_ms']:.3f} ms"
        )

        print(
            f"Median latency: "
            f"{summary['median_latency_ms']:.3f} ms"
        )

        print(
            f"Mean throughput: "
            f"{summary['mean_pairs_per_second']:.2f} pairs/s"
        )

        print(
            f"Latency std: "
            f"{summary['std_latency_ms']:.3f} ms"
        )

        print(
            f"Max score difference: "
            f"{cross_batch_max_abs_diff:.8f}"
        )

    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (OUTPUT_DIR / "per_run.json").write_text(
        json.dumps(all_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    best = max(
        summaries,
        key=lambda row: row["mean_pairs_per_second"],
    )

    print("\n" + "=" * 76)
    print("BENCHMARK COMPLETE")
    print("=" * 76)

    print(
        f"Best observed batch size: "
        f"{best['batch_size']}"
    )

    print(
        f"Best observed throughput: "
        f"{best['mean_pairs_per_second']:.2f} pairs/s"
    )

    print(f"Results: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
