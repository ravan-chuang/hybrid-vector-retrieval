"""Analyze real Cross-Encoder input lengths from benchmark candidates."""

import json
import statistics
from pathlib import Path

import numpy as np
import requests

from backend.api.services.cross_encoder import get_cross_encoder

BASE_URL = "http://127.0.0.1:8000"

OUTPUT = Path(
    "results/msmarco_full/latency/api_v077"
)
OUTPUT.mkdir(parents=True, exist_ok=True)

QUERIES = [
    "what is information retrieval",
    "how does a search engine work",
    "what causes climate change",
    "what causes high blood pressure",
    "what is machine learning",
    "how do neural networks work",
    "what is the function of mitochondria",
    "how does photosynthesis work",
    "what causes diabetes",
    "what is the difference between DNA and RNA",
]


def main():
    model = get_cross_encoder()
    tokenizer = model.tokenizer
    max_length = model.max_seq_length

    records = []
    exported = []

    for query in QUERIES:
        payload = {
            "query": query,
            "corpus": "benchmark",
            "lexical_backend": "cpp_bm25",
            "stopword_filter": True,
            "top_k": 10,
            "candidate_k": 50,
            "ef_search": 160,
        }

        response = requests.post(
            BASE_URL + "/internal/benchmark/candidates",
            json=payload,
            timeout=60,
        )
        response.raise_for_status()

        data = response.json()
        candidates = data["candidates"]

        exported.append({
            "query": query,
            "candidates": candidates,
        })

        for candidate in candidates:
            content = candidate["content"]

            raw_tokens = tokenizer(
                query,
                content,
                truncation=False,
                add_special_tokens=True,
            )["input_ids"]

            raw_length = len(raw_tokens)
            effective_length = min(raw_length, max_length)

            records.append({
                "query": query,
                "document_id": candidate["document_id"],
                "raw_tokens": raw_length,
                "effective_tokens": effective_length,
                "truncated": raw_length > max_length,
            })

        print(
            f"Query: {query[:35]:35s} "
            f"Candidates: {len(candidates)}",
            flush=True,
        )

    raw = np.array([
        row["raw_tokens"] for row in records
    ])

    effective = np.array([
        row["effective_tokens"] for row in records
    ])

    truncated = sum(
        row["truncated"] for row in records
    )

    summary = {
        "queries": len(QUERIES),
        "pairs": len(records),
        "max_seq_length": max_length,
        "raw_mean": float(np.mean(raw)),
        "raw_p50": float(np.percentile(raw, 50)),
        "raw_p90": float(np.percentile(raw, 90)),
        "raw_p95": float(np.percentile(raw, 95)),
        "raw_p99": float(np.percentile(raw, 99)),
        "raw_max": int(np.max(raw)),
        "effective_mean": float(np.mean(effective)),
        "effective_p50": float(np.percentile(effective, 50)),
        "effective_p95": float(np.percentile(effective, 95)),
        "truncated_count": truncated,
        "truncation_rate": truncated / len(records),
    }

    (OUTPUT / "real_candidates.json").write_text(
        json.dumps(exported, indent=2)
    )

    (OUTPUT / "real_token_records.json").write_text(
        json.dumps(records, indent=2)
    )

    (OUTPUT / "real_token_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print("\n=== REAL TOKEN LENGTH SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
