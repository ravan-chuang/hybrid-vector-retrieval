"""v0.7.7 Phase 3: Real-document Cross-Encoder profiling."""

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
OUTPUT = BASE / "real_breakdown_profile.json"

PAIR_COUNTS = [128, 256, 384, 512, 768]
WARMUPS = 2
REPEATS = 5


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()


def profile_once(model, pairs, batch_size, device):
    tokenizer = model.tokenizer
    network = model.model
    network.eval()

    max_length = model.max_seq_length

    timings = {
        "tokenization_ms": 0.0,
        "transfer_ms": 0.0,
        "forward_ms": 0.0,
        "postprocess_ms": 0.0,
    }

    total_actual_tokens = 0
    total_padded_tokens = 0
    output_scores = []

    synchronize(device)
    total_start = time.perf_counter()

    for offset in range(0, len(pairs), batch_size):
        chunk = pairs[offset:offset + batch_size]

        texts_a = [pair[0] for pair in chunk]
        texts_b = [pair[1] for pair in chunk]

        start = time.perf_counter()

        encoded = tokenizer(
            texts_a,
            texts_b,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        timings["tokenization_ms"] += (
            time.perf_counter() - start
        ) * 1000

        attention_mask = encoded["attention_mask"]

        total_actual_tokens += int(
            attention_mask.sum().item()
        )

        total_padded_tokens += int(
            attention_mask.numel()
        )

        start = time.perf_counter()

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        synchronize(device)

        timings["transfer_ms"] += (
            time.perf_counter() - start
        ) * 1000

        start = time.perf_counter()

        with torch.inference_mode():
            logits = network(**encoded).logits

        synchronize(device)

        timings["forward_ms"] += (
            time.perf_counter() - start
        ) * 1000

        start = time.perf_counter()

        scores = logits.detach().cpu().numpy()
        output_scores.append(scores)

        timings["postprocess_ms"] += (
            time.perf_counter() - start
        ) * 1000

    synchronize(device)

    total_ms = (
        time.perf_counter() - total_start
    ) * 1000

    scores = np.concatenate(
        output_scores
    ).reshape(-1)

    if len(scores) != len(pairs):
        raise RuntimeError("Score count mismatch")

    timings["total_ms"] = total_ms

    timings["padding_efficiency"] = (
        total_actual_tokens / total_padded_tokens
    )

    timings["padding_waste_rate"] = (
        1 - timings["padding_efficiency"]
    )

    timings["actual_tokens"] = total_actual_tokens
    timings["padded_tokens"] = total_padded_tokens

    timings["other_ms"] = max(
        0.0,
        total_ms
        - timings["tokenization_ms"]
        - timings["transfer_ms"]
        - timings["forward_ms"]
        - timings["postprocess_ms"],
    )

    return timings


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    exported = json.loads(INPUT.read_text())

    pairs = [
        (item["query"], candidate["content"])
        for item in exported
        for candidate in item["candidates"]
    ]

    model = get_cross_encoder()
    device = resolve_device()
    batch_size = resolve_batch_size()

    print("=== v0.7.7 REAL BREAKDOWN ===")
    print(f"Available pairs: {len(pairs)}")
    print(f"Device: {device}")
    print(f"Internal batch size: {batch_size}")
    print(f"Max sequence length: {model.max_seq_length}")
    print()

    results = []

    for count in PAIR_COUNTS:
        if count > len(pairs):
            continue

        selected = pairs[:count]

        for _ in range(WARMUPS):
            profile_once(
                model,
                selected,
                batch_size,
                device,
            )

        profiles = [
            profile_once(
                model,
                selected,
                batch_size,
                device,
            )
            for _ in range(REPEATS)
        ]

        def avg(key):
            return statistics.mean(
                profile[key] for profile in profiles
            )

        row = {
            "pairs": count,
            "total_mean_ms": avg("total_ms"),
            "tokenization_mean_ms": avg("tokenization_ms"),
            "transfer_mean_ms": avg("transfer_ms"),
            "forward_mean_ms": avg("forward_ms"),
            "postprocess_mean_ms": avg("postprocess_ms"),
            "other_mean_ms": avg("other_ms"),
            "padding_efficiency": avg("padding_efficiency"),
            "padding_waste_rate": avg("padding_waste_rate"),
            "actual_tokens": profiles[0]["actual_tokens"],
            "padded_tokens": profiles[0]["padded_tokens"],
            "timings": profiles,
        }

        row["pairs_per_second"] = (
            count / (row["total_mean_ms"] / 1000)
        )

        results.append(row)

        print(
            f"Pairs={count:4d} "
            f"Total={row['total_mean_ms']:8.2f} ms "
            f"Tokenize={row['tokenization_mean_ms']:7.2f} "
            f"Transfer={row['transfer_mean_ms']:7.2f} "
            f"Forward={row['forward_mean_ms']:8.2f} "
            f"Post={row['postprocess_mean_ms']:7.2f} "
            f"PaddingWaste={row['padding_waste_rate'] * 100:5.1f}% "
            f"Pairs/s={row['pairs_per_second']:8.2f}",
            flush=True,
        )

    OUTPUT.write_text(
        json.dumps(results, indent=2)
    )

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
