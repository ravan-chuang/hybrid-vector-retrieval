"""Cross-Encoder model loading and passage reranking."""

from functools import lru_cache
from threading import Lock

import torch
from sentence_transformers import CrossEncoder


CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_inference_lock = Lock()


@lru_cache(maxsize=1)
def get_cross_encoder() -> CrossEncoder:
    """Load the reranker once per process, on first use."""

    device = "cpu"

    model = CrossEncoder(
        CROSS_ENCODER_MODEL,
        device=device,
    )

    return model


def rerank_candidates(
    query: str,
    candidates: list[dict],
    top_k: int = 10,
    batch_size: int = 16,
) -> list[dict]:
    """Score candidate passages and return the highest-ranked documents."""

    if not candidates:
        return []

    model = get_cross_encoder()

    pairs = [
        (query, candidate["content"])
        for candidate in candidates
    ]

    with _inference_lock:
        scores = model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    ranked = []

    for candidate, score in zip(candidates, scores):
        ranked.append({
            **candidate,
            "cross_encoder_score": float(score),
        })

    ranked.sort(
        key=lambda item: (
            -item["cross_encoder_score"],
            item["document_id"],
        )
    )

    return ranked[:top_k]


def union_candidates(
    lexical_rows: list[tuple],
    vector_rows: list[tuple],
) -> list[dict]:
    """Union lexical and dense candidates by document ID."""

    candidates = {}

    for rank, row in enumerate(lexical_rows, start=1):
        document_id, external_id, content = row[:3]

        candidates[document_id] = {
            "document_id": document_id,
            "external_id": external_id,
            "content": content,
            "lexical_rank": rank,
            "vector_rank": None,
        }

    for rank, row in enumerate(vector_rows, start=1):
        document_id, external_id, content = row[:3]

        if document_id in candidates:
            candidates[document_id]["vector_rank"] = rank
        else:
            candidates[document_id] = {
                "document_id": document_id,
                "external_id": external_id,
                "content": content,
                "lexical_rank": None,
                "vector_rank": rank,
            }

    return list(candidates.values())
