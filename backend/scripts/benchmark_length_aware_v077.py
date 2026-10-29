"""Compare original-order and length-aware Cross-Encoder inference."""

import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from backend.api.services.cross_encoder import (
    get_cross_encoder,
    resolve_batch_size,
    resolve_device,
)

BASE = Path("results/msmarco_full/latency/api_v077")
INPUT = BASE / "real_candidates.json"
OUTPUT = BASE / "length_aware_profile.json"

PAIR_COUNTS = [128, 256, 384, 512, 768]
WARMUPS = 2
REPEATS = 5


def sync(device):
    if device == "mps":
        torch.mps.synchronize()


def predict(model, pairs, batch_size, device):
    sync(device)
    start = time.perf_counter()

    scores = model.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    sync(device)

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000

    return (
        np.asarray(scores).reshape(-1),
        elapsed_ms,
    )


def padding_stats(lengths, batch_size):
    actual = sum(lengths)
    padded = 0

    for start in range(0, len(lengths), batch_size):
        chunk = lengths[start:start + batch_size]
        padded += len(chunk) * max(chunk)

    return {
        "actual_tokens": actual,
        "padded_tokens": padded,
        "padding_waste_rate": 1 - actual / padded,
    }


def main():
    exported = json.loads(INPUT.read_text())

    pairs = [
        (item["query"], candidate["content"])
        for item in exported
        for candidate in item["candidates"]
    ]

    model = get_cross_encoder()
    tokenizer = model.tokenizer
    batch_size = resolve_batch_size()
    device = resolve_device()

    lengths = [
        len(
            tokenizer(
                query,
                content,
                truncation=True,
                max_length=model.max_seq_length,
            )["input_ids"]
        )
        for query, content in pairs
    ]

    results = []

    for count in PAIR_COUNTS:
        selected_pairs = pairs[:count]
        selected_lengths = lengths[:count]

        order = sorted(
            range(count),
            key=lambda i: (selected_lengths[i], i),
        )

        sorted_pairs = [
            selected_pairs[i] for i in order
        ]

        for _ in range(WARMUPS):
            predict(
                model, selected_pairs, batch_size, device
            )
            predict(
                model, sorted_pairs, batch_size, device
            )

        original_times = []
        sorted_times = []
        max_score_diffs = []

        for _ in range(REPEATS):
            original_scores, original_ms = predict(
                model,
                selected_pairs,
                batch_size,
                device,
            )

            sorted_scores, sorted_ms = predict(
                model,
                sorted_pairs,
                batch_size,
                device,
            )

            restored_scores = np.empty_like(sorted_scores)

            for sorted_position, original_position in enumerate(order):
                restored_scores[original_position] = (
                    sorted_scores[sorted_position]
                )

            original_times.append(original_ms)
            sorted_times.append(sorted_ms)

            max_score_diffs.append(
                float(
                    np.max(
                        np.abs(
                            original_scores - restored_scores
                        )
                    )
                )
            )

        original_mean = statistics.mean(original_times)
        sorted_mean = statistics.mean(sorted_times)

        original_padding = padding_stats(
            selected_lengths,
            batch_size,
        )

        sorted_padding = padding_stats(
            [selected_lengths[i] for i in order],
            batch_size,
        )

        row = {
            "pairs": count,
            "original_mean_ms": original_mean,
            "length_aware_mean_ms": sorted_mean,
            "speedup": original_mean / sorted_mean,
            "original_padding_waste": original_padding[
                "padding_waste_rate"
            ],
            "length_aware_padding_waste": sorted_padding[
                "padding_waste_rate"
            ],
            "max_abs_score_diff": max(max_score_diffs),
            "original_timings_ms": original_times,
            "length_aware_timings_ms": sorted_times,
        }

        results.append(row)

        print(
            f"Pairs={count:4d} "
            f"Original={original_mean:8.2f} ms "
            f"LengthAware={sorted_mean:8.2f} ms "
            f"Speedup={row['speedup']:5.2f}x "
            f"Padding="
            f"{row['original_padding_waste'] * 100:5.1f}%"
            f"->{row['length_aware_padding_waste'] * 100:5.1f}% "
            f"MaxDiff={row['max_abs_score_diff']:.6f}",
            flush=True,
        )

    OUTPUT.write_text(
        json.dumps(results, indent=2)
    )

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
