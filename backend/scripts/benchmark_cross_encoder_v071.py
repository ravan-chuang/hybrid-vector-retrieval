import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import psycopg
import torch
from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

QUERY = "what is information retrieval"

CANDIDATE_COUNTS = [50, 100]
BATCH_SIZES = [8, 16, 32, 64]

WARMUP = 2
REPEATS = 5

BASE = Path(
    "results/msmarco_full/latency/ce_v071"
)
BASE.mkdir(parents=True, exist_ok=True)


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()


def load_candidates(limit=100):
    with psycopg.connect(
        os.environ["DATABASE_URL"]
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    content
                FROM public.msmarco_full_documents
                WHERE content IS NOT NULL
                ORDER BY document_id
                LIMIT %s
                """,
                (limit,),
            )

            rows = cur.fetchall()

    return [row[0] for row in rows]


def run_prediction(model, pairs, batch_size, device):
    synchronize(device)

    start = time.perf_counter()

    scores = model.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=False,
    )

    synchronize(device)

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000

    return (
        np.asarray(scores).reshape(-1),
        elapsed_ms,
    )


def topk(scores, k=10):
    return np.argsort(-scores)[:k].tolist()


def percentile(values, p):
    return float(
        np.percentile(values, p)
    )


def main():
    print("PyTorch:", torch.__version__)
    print("MPS available:", torch.backends.mps.is_available())

    devices = ["cpu"]

    if torch.backends.mps.is_available():
        devices.append("mps")

    documents = load_candidates(max(CANDIDATE_COUNTS))

    print("Loaded documents:", len(documents))

    if len(documents) < max(CANDIDATE_COUNTS):
        raise RuntimeError("Not enough documents")

    results = []
    reference_scores = {}

    for device in devices:
        print("\n" + "=" * 70)
        print("DEVICE:", device)
        print("=" * 70)

        model = CrossEncoder(
            MODEL_NAME,
            device=device,
            activation_fn=torch.nn.Identity(),
        )

        for candidate_count in CANDIDATE_COUNTS:
            pairs = [
                (QUERY, document)
                for document in documents[:candidate_count]
            ]

            for batch_size in BATCH_SIZES:
                for _ in range(WARMUP):
                    run_prediction(
                        model,
                        pairs,
                        batch_size,
                        device,
                    )

                timings = []
                scores = None

                for _ in range(REPEATS):
                    scores, elapsed_ms = run_prediction(
                        model,
                        pairs,
                        batch_size,
                        device,
                    )

                    timings.append(elapsed_ms)

                if device == "cpu":
                    reference_scores[candidate_count] = scores.copy()

                reference = reference_scores[candidate_count]

                max_abs_diff = float(
                    np.max(np.abs(scores - reference))
                )

                cpu_top10 = topk(reference)
                device_top10 = topk(scores)

                agreement = (
                    len(set(cpu_top10) & set(device_top10))
                    / len(cpu_top10)
                )

                row = {
                    "device": device,
                    "candidate_count": candidate_count,
                    "batch_size": batch_size,
                    "mean_ms": statistics.mean(timings),
                    "median_ms": statistics.median(timings),
                    "p95_ms": percentile(timings, 95),
                    "min_ms": min(timings),
                    "max_ms": max(timings),
                    "max_abs_score_diff": max_abs_diff,
                    "top10_set_agreement": agreement,
                    "timings_ms": timings,
                }

                results.append(row)

                print(
                    f"N={candidate_count:3d} "
                    f"batch={batch_size:2d} "
                    f"mean={row['mean_ms']:9.3f} ms "
                    f"p95={row['p95_ms']:9.3f} ms "
                    f"agreement={agreement:.3f} "
                    f"max_diff={max_abs_diff:.6f}"
                )

        del model

        if device == "mps":
            torch.mps.empty_cache()

    output = BASE / "benchmark_results.json"

    output.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("BEST CONFIGURATIONS")
    print("=" * 70)

    for candidate_count in CANDIDATE_COUNTS:
        selected = [
            row for row in results
            if row["candidate_count"] == candidate_count
        ]

        best = min(
            selected,
            key=lambda row: row["median_ms"],
        )

        print(
            f"N={candidate_count}: "
            f"{best['device']} "
            f"batch={best['batch_size']} "
            f"median={best['median_ms']:.3f} ms"
        )

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
