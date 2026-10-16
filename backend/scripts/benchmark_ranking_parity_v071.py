"""CPU vs MPS Cross-Encoder ranking parity on fixed API candidates."""

import json
import os
from pathlib import Path

import numpy as np
import requests
import torch

from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

API_URL = "http://127.0.0.1:8000/internal/benchmark/candidates"

BATCH_SIZE = 32
CANDIDATE_K = 50
TOP_K = 50

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

BASE = Path("results/msmarco_full/parity/v071")
BASE.mkdir(parents=True, exist_ok=True)


def fetch_candidates():
    datasets = []

    for index, query in enumerate(QUERIES, start=1):
        payload = {
            "query": query,
            "corpus": "benchmark",
            "lexical_backend": "cpp_bm25",
            "stopword_filter": True,
            "top_k": TOP_K,
            "candidate_k": CANDIDATE_K,
            "ef_search": 160,
        }

        response = requests.post(
            API_URL,
            json=payload,
            timeout=120,
        )

        response.raise_for_status()
        data = response.json()

        candidates = data["candidates"]

        if len(candidates) != data["candidate_count"]:
            raise RuntimeError(
                f"Incomplete candidate set for {query}: "
                f"{len(candidates)} / {data['candidate_count']}"
            )

        datasets.append({
            "query": query,
            "candidates": candidates,
        })

        print(
            f"[{index:02d}/{len(QUERIES)}] "
            f"Candidates={len(candidates)} "
            f"Query={query}"
        )

    return datasets


def score_datasets(datasets, device):
    model = CrossEncoder(
        MODEL_NAME,
        device=device,
        activation_fn=torch.nn.Identity(),
    )

    outputs = []

    for item in datasets:
        query = item["query"]
        candidates = item["candidates"]

        pairs = [
            (query, candidate["content"])
            for candidate in candidates
        ]

        scores = model.predict(
            pairs,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        if device == "mps":
            torch.mps.synchronize()

        outputs.append(
            np.asarray(scores, dtype=np.float64)
        )

    del model

    if device == "mps":
        torch.mps.empty_cache()

    return outputs


def ranked_ids(candidates, scores):
    rows = [
        (
            int(candidate["document_id"]),
            float(score),
        )
        for candidate, score in zip(candidates, scores)
    ]

    rows.sort(key=lambda row: (-row[1], row[0]))

    return [row[0] for row in rows]


def kendall_tau(ids_a, ids_b):
    positions = {
        doc_id: index
        for index, doc_id in enumerate(ids_b)
    }

    order = [positions[doc_id] for doc_id in ids_a]

    concordant = 0
    discordant = 0

    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            if order[i] < order[j]:
                concordant += 1
            else:
                discordant += 1

    total = concordant + discordant

    return (
        (concordant - discordant) / total
        if total
        else 1.0
    )


def main():
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")

    datasets = fetch_candidates()

    (BASE / "fixed_candidates.json").write_text(
        json.dumps(datasets, indent=2),
        encoding="utf-8",
    )

    print("\nScoring on CPU...")
    cpu_scores = score_datasets(datasets, "cpu")

    print("Scoring on MPS...")
    mps_scores = score_datasets(datasets, "mps")

    results = []

    for item, cpu, mps in zip(
        datasets,
        cpu_scores,
        mps_scores,
    ):
        candidates = item["candidates"]

        cpu_ids = ranked_ids(candidates, cpu)
        mps_ids = ranked_ids(candidates, mps)

        cpu_top10 = cpu_ids[:10]
        mps_top10 = mps_ids[:10]

        result = {
            "query": item["query"],
            "candidate_count": len(candidates),
            "max_abs_score_diff": float(
                np.max(np.abs(cpu - mps))
            ),
            "top1_agreement": (
                cpu_top10[0] == mps_top10[0]
            ),
            "top10_set_agreement": (
                len(set(cpu_top10) & set(mps_top10)) / 10
            ),
            "top10_ordered_agreement": (
                sum(
                    a == b
                    for a, b in zip(cpu_top10, mps_top10)
                ) / 10
            ),
            "exact_top10_match": (
                cpu_top10 == mps_top10
            ),
            "kendall_tau_full": kendall_tau(
                cpu_ids,
                mps_ids,
            ),
            "cpu_top10": cpu_top10,
            "mps_top10": mps_top10,
        }

        results.append(result)

        print(
            f"{item['query'][:38]:38s} "
            f"set={result['top10_set_agreement']:.3f} "
            f"ordered={result['top10_ordered_agreement']:.3f} "
            f"tau={result['kendall_tau_full']:.6f} "
            f"max_diff={result['max_abs_score_diff']:.6f}"
        )

    output = BASE / "parity_results.json"

    output.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\nSUMMARY")
    print("Queries:", len(results))

    print(
        "Exact Top-10:",
        sum(row["exact_top10_match"] for row in results),
        "/",
        len(results),
    )

    print(
        "Top-1 Agreement:",
        sum(row["top1_agreement"] for row in results),
        "/",
        len(results),
    )

    print(
        "Worst Max Absolute Score Difference:",
        max(row["max_abs_score_diff"] for row in results),
    )

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
