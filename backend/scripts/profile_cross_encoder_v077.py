"""v0.7.7 Cross-Encoder CPU/MPS profiling."""

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

OUTPUT = Path(
    "results/msmarco_full/latency/api_v077"
)
OUTPUT.mkdir(parents=True, exist_ok=True)

PAIR_COUNTS = [64, 128, 256, 384, 512, 768, 1024]
REPEATS = 5
WARMUPS = 2


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()


def make_pairs(count):
    queries = [
        "what is information retrieval",
        "how does machine learning work",
        "what causes climate change",
        "how do neural networks learn",
    ]

    passages = [
        (
            "Information retrieval is the process of "
            "finding relevant documents in large collections."
        ),
        (
            "Machine learning algorithms learn patterns "
            "from data to make predictions."
        ),
        (
            "Climate change is influenced by greenhouse "
            "gas emissions and environmental factors."
        ),
        (
            "Neural networks use layers of parameters "
            "optimized through gradient-based learning."
        ),
    ]

    return [
        (
            queries[i % len(queries)],
            passages[i % len(passages)],
        )
        for i in range(count)
    ]


def profile_predict(model, pairs, batch_size, device):
    synchronize(device)
    start = time.perf_counter()

    scores = model.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    synchronize(device)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, len(np.asarray(scores).reshape(-1))


def profile_manual(model, pairs, batch_size, device):
    tokenizer = model.tokenizer
    network = model.model
    network.eval()

    timings = {
        "tokenization_ms": 0.0,
        "transfer_ms": 0.0,
        "forward_ms": 0.0,
        "postprocess_ms": 0.0,
    }

    outputs = []

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
            return_tensors="pt",
        )

        timings["tokenization_ms"] += (
            time.perf_counter() - start
        ) * 1000

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
            result = network(**encoded)
            logits = result.logits

        synchronize(device)

        timings["forward_ms"] += (
            time.perf_counter() - start
        ) * 1000

        start = time.perf_counter()

        scores = logits.detach().cpu().numpy()
        outputs.append(scores)

        timings["postprocess_ms"] += (
            time.perf_counter() - start
        ) * 1000

    return timings, np.concatenate(outputs).reshape(-1)


def main():
    device = resolve_device()
    batch_size = resolve_batch_size()
    model = get_cross_encoder()

    print(f"Device: {device}")
    print(f"Internal batch size: {batch_size}")
    print(f"Warmups: {WARMUPS}")
    print(f"Repeats: {REPEATS}")

    results = []

    for count in PAIR_COUNTS:
        pairs = make_pairs(count)

        for _ in range(WARMUPS):
            profile_predict(
                model, pairs, batch_size, device
            )
            profile_manual(
                model, pairs, batch_size, device
            )

        predict_times = []
        manual_times = []
        max_abs_score_diff = []

        for _ in range(REPEATS):
            predict_ms, score_count = profile_predict(
                model, pairs, batch_size, device
            )

            manual_profile, manual_scores = profile_manual(
                model, pairs, batch_size, device
            )

            predict_times.append(predict_ms)
            manual_times.append(manual_profile)

            if len(manual_scores) != score_count:
                raise RuntimeError(
                    "Manual score count mismatch"
                )

        row = {
            "pairs": count,
            "device": device,
            "batch_size": batch_size,
            "predict_mean_ms": statistics.mean(
                predict_times
            ),
            "predict_median_ms": statistics.median(
                predict_times
            ),
            "tokenization_mean_ms": statistics.mean(
                item["tokenization_ms"]
                for item in manual_times
            ),
            "transfer_mean_ms": statistics.mean(
                item["transfer_ms"]
                for item in manual_times
            ),
            "forward_mean_ms": statistics.mean(
                item["forward_ms"]
                for item in manual_times
            ),
            "postprocess_mean_ms": statistics.mean(
                item["postprocess_ms"]
                for item in manual_times
            ),
        }

        row["predict_pairs_per_second"] = (
            count / (row["predict_mean_ms"] / 1000)
        )

        results.append(row)

        print(
            f"Pairs={count:4d} "
            f"Predict={row['predict_mean_ms']:8.2f} ms "
            f"Tokenize={row['tokenization_mean_ms']:8.2f} ms "
            f"Forward={row['forward_mean_ms']:8.2f} ms "
            f"Pairs/s={row['predict_pairs_per_second']:8.2f}",
            flush=True,
        )

    path = OUTPUT / "cross_encoder_profile.json"

    path.write_text(
        json.dumps(results, indent=2)
    )

    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
