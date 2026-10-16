"""Cross-Encoder model loading and passage reranking."""

import os
import time
from functools import lru_cache
from threading import Lock

import torch
from sentence_transformers import CrossEncoder


CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_inference_lock = Lock()


def resolve_device() -> str:
    """Resolve CPU/MPS device from environment configuration."""

    requested = os.getenv(
        "CROSS_ENCODER_DEVICE",
        "cpu",
    ).strip().lower()

    if requested not in {"cpu", "mps", "auto"}:
        raise ValueError(
            "CROSS_ENCODER_DEVICE must be cpu, mps, or auto"
        )

    if requested == "cpu":
        return "cpu"

    available = torch.backends.mps.is_available()

    if requested == "mps" and not available:
        print(
            "[CrossEncoder] MPS unavailable; "
            "falling back to CPU."
        )
        return "cpu"

    if requested == "auto":
        return "mps" if available else "cpu"

    return "mps"


def resolve_batch_size() -> int:
    """Read and validate Cross-Encoder batch size."""

    value = int(
        os.getenv("CROSS_ENCODER_BATCH_SIZE", "32")
    )

    if not 1 <= value <= 256:
        raise ValueError(
            "CROSS_ENCODER_BATCH_SIZE must be between 1 and 256"
        )

    return value


@lru_cache(maxsize=1)
def get_cross_encoder() -> CrossEncoder:
    """Load the reranker once per process, on first use."""

    device = resolve_device()

    model = CrossEncoder(
        CROSS_ENCODER_MODEL,
        device=device,
        activation_fn=torch.nn.Identity(),
    )

    print(
        f"[CrossEncoder] Loaded on device={device}"
    )

    return model



def rerank_candidates_profiled(
    query: str,
    candidates: list[dict],
    top_k: int = 10,
    batch_size: int | None = None,
) -> tuple[list[dict], dict[str, float]]:
    """Rerank candidates and profile lock contention."""

    total_start = time.perf_counter()

    if not candidates:
        return [], {
            "rerank_lock_wait_ms": 0.0,
            "rerank_inference_ms": 0.0,
            "rerank_other_ms": 0.0,
        }

    if batch_size is None:
        batch_size = resolve_batch_size()

    model = get_cross_encoder()

    pairs = [
        (query, candidate["content"])
        for candidate in candidates
    ]

    lock_start = time.perf_counter()

    with _inference_lock:
        lock_acquired = time.perf_counter()

        scores = model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        if resolve_device() == "mps":
            torch.mps.synchronize()

        inference_end = time.perf_counter()

    ranked = [
        {
            **candidate,
            "cross_encoder_score": float(score),
        }
        for candidate, score in zip(candidates, scores)
    ]

    ranked.sort(
        key=lambda item: (
            -item["cross_encoder_score"],
            item["document_id"],
        )
    )

    ranked = ranked[:top_k]

    total_end = time.perf_counter()

    lock_wait_ms = (
        lock_acquired - lock_start
    ) * 1000.0

    inference_ms = (
        inference_end - lock_acquired
    ) * 1000.0

    total_ms = (
        total_end - total_start
    ) * 1000.0

    other_ms = max(
        0.0,
        total_ms - lock_wait_ms - inference_ms,
    )

    return ranked, {
        "rerank_lock_wait_ms": round(lock_wait_ms, 3),
        "rerank_inference_ms": round(inference_ms, 3),
        "rerank_other_ms": round(other_ms, 3),
    }


def rerank_candidates(
    query: str,
    candidates: list[dict],
    top_k: int = 10,
    batch_size: int | None = None,
) -> list[dict]:
    """Backward-compatible reranking interface."""

    ranked, _ = rerank_candidates_profiled(
        query=query,
        candidates=candidates,
        top_k=top_k,
        batch_size=batch_size,
    )

    return ranked


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
