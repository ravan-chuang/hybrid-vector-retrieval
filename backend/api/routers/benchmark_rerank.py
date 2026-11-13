"""Local-only frozen-candidate Cross-Encoder benchmark endpoints."""

import os
import time

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, model_validator

from backend.api.services.cross_encoder import (
    rerank_candidates_profiled,
)
from backend.api.services.dynamic_batch import (
    BatchQueueFullError,
    BatchTimeoutError,
    get_dynamic_batch_reranker,
)


router = APIRouter(
    prefix="/internal/benchmark",
    tags=["Internal Benchmark"],
)


class FrozenCandidate(BaseModel):
    document_id: int
    external_id: str
    content: str = Field(min_length=1, max_length=10000)
    lexical_rank: int | None = None
    vector_rank: int | None = None


class FrozenRerankRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=10, ge=1, le=100)
    candidates: list[FrozenCandidate] = Field(
        min_length=1,
        max_length=100,
    )

    @model_validator(mode="after")
    def validate_candidates(self):
        if self.top_k > len(self.candidates):
            raise ValueError(
                "top_k cannot exceed candidate count"
            )

        ids = [
            candidate.document_id
            for candidate in self.candidates
        ]

        if len(ids) != len(set(ids)):
            raise ValueError(
                "Duplicate document IDs are not allowed"
            )

        return self


def require_local_benchmark(request: Request):
    enabled = (
        os.getenv("ENABLE_FROZEN_BENCHMARK", "0") == "1"
    )

    if not enabled:
        raise HTTPException(
            status_code=404,
            detail="Benchmark endpoint disabled",
        )

    host = (
        request.client.host
        if request.client is not None
        else None
    )

    if host not in {"127.0.0.1", "::1"}:
        raise HTTPException(
            status_code=403,
            detail="Local access only",
        )


@router.post("/rerank")
def benchmark_rerank(
    request: Request,
    payload: FrozenRerankRequest,
):
    require_local_benchmark(request)

    candidates = [
        candidate.model_dump()
        for candidate in payload.candidates
    ]

    start = time.perf_counter()

    ranked, profile = rerank_candidates_profiled(
        query=payload.query,
        candidates=candidates,
        top_k=payload.top_k,
    )

    total_ms = (
        time.perf_counter() - start
    ) * 1000

    return {
        "pipeline": "baseline",
        "query": payload.query,
        "candidate_count": len(candidates),
        "total_ms": round(total_ms, 3),
        **profile,
        "results": ranked,
    }


@router.post("/rerank/batched")
def benchmark_rerank_batched(
    request: Request,
    payload: FrozenRerankRequest,
):
    require_local_benchmark(request)

    candidates = [
        candidate.model_dump()
        for candidate in payload.candidates
    ]

    start = time.perf_counter()

    try:
        ranked, profile = (
            get_dynamic_batch_reranker().submit(
                query=payload.query,
                candidates=candidates,
                top_k=payload.top_k,
            )
        )

    except BatchQueueFullError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except BatchTimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail=str(exc),
        ) from exc

    total_ms = (
        time.perf_counter() - start
    ) * 1000

    return {
        "pipeline": "batched",
        "query": payload.query,
        "candidate_count": len(candidates),
        "total_ms": round(total_ms, 3),
        **profile,
        "results": ranked,
    }
