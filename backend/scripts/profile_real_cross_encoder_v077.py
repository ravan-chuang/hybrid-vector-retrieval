"""Profile Cross-Encoder inference using real retrieval candidates."""

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
OUTPUT = BASE / "real_cross_encoder_profile.json"

PAIR_COUNTS = [64, 128, 256, 384, 512, 768]
WARMUPS = 2
REPEATS = 5


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()


def predict_once(model, pairs, batch_size, device):
    synchronize(device)

    start = time.perf_counter()

    scores = model.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    synchronize(device)

    elapsed_ms = (time.perf_counter() - start) * 1000

    if len(np.asarray(scores).reshape(-1)) != len(pairs):
        raise RuntimeError("Score count mismatch")

    return elapsed_ms


def main():
    exported = json.loads(INPUT.read_text())

    pairs = [
        (item["query"], candidate["content"])
        for item in exported
        for candidate in item["candidates"]
    ]

    model = get_cross_encoder()
    device = resolve_device()
    batch_size = resolve_batch_size()

    print(f"Real pairs available: {len(pairs)}")
    print(f"Device: {device}")
    print(f"Internal batch size: {batch_size}")

    results = []

    for count in PAIR_COUNTS:
        if count > len(pairs):
            continue

        selected = pairs[:count]

        for _ in range(WARMUPS):
            predict_once(
                model, selected, batch_size, device
            )

        timings = [
            predict_once(
                model, selected, batch_size, device
            )
            for _ in range(REPEATS)
        ]

        mean_ms = statistics.mean(timings)

        row = {
            "pairs": count,
            "mean_ms": mean_ms,
            "median_ms": statistics.median(timings),
            "pairs_per_second": count / (mean_ms / 1000),
            "timings_ms": timings,
        }

        results.append(row)

        print(
            f"Pairs={count:4d} "
            f"Predict={mean_ms:8.2f} ms "
            f"Throughput={row['pairs_per_second']:8.2f} pairs/s",
            flush=True,
        )

    OUTPUT.write_text(json.dumps(results, indent=2))

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
