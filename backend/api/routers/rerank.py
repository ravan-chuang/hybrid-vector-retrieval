"""Live Cross-Encoder reranking endpoint."""

import time

from fastapi import APIRouter, HTTPException, Request

from backend.api.schemas.search import (
    RerankSearchRequest,
    RerankSearchResponse,
    BatchedRerankSearchResponse,
    RerankSearchResult,
)
from backend.api.routers.search import (
    retrieve_hybrid_candidates,
    log_search_history,
)
from backend.api.services.cross_encoder import (
    rerank_candidates_profiled,
    union_candidates,
)


router = APIRouter()



@router.post(
    "/internal/benchmark/candidates",
    include_in_schema=False,
)
def export_benchmark_candidates(
    request: Request,
    payload: RerankSearchRequest,
):
    """Export full retrieval candidates for local parity testing."""

    client_host = (
        request.client.host
        if request.client is not None
        else None
    )

    if client_host not in {"127.0.0.1", "::1"}:
        raise HTTPException(
            status_code=403,
            detail="Local benchmark access only.",
        )

    if payload.corpus != "benchmark":
        raise HTTPException(
            status_code=422,
            detail="Benchmark corpus required.",
        )

    candidates_data = retrieve_hybrid_candidates(payload)

    candidates = union_candidates(
        candidates_data["lexical_rows"],
        candidates_data["vector_rows"],
    )

    return {
        "query": payload.query,
        "candidate_k": payload.candidate_k,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


@router.post(
    "/search/rerank",
    response_model=RerankSearchResponse,
    tags=["Search"],
)
def rerank_search(request: RerankSearchRequest):
    total_start = time.perf_counter()

    candidates_data = retrieve_hybrid_candidates(request)

    candidates = union_candidates(
        candidates_data["lexical_rows"],
        candidates_data["vector_rows"],
    )

    rerank_start = time.perf_counter()

    try:
        ranked, profile = rerank_candidates_profiled(
            query=request.query,
            candidates=candidates,
            top_k=request.top_k,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Cross-Encoder reranking failed: {exc}",
        ) from exc

    rerank_ms = (
        time.perf_counter() - rerank_start
    ) * 1000.0

    total_ms = (
        time.perf_counter() - total_start
    ) * 1000.0

    results = [
        RerankSearchResult(**item)
        for item in ranked
    ]

    log_search_history(
        query_text=request.query,
        search_type="rerank",
        result_count=len(results),
        latency_ms=total_ms,
    )

    return RerankSearchResponse(
        query=request.query,
        method="cross_encoder_union_rerank",
        top_k=request.top_k,
        candidate_k=request.candidate_k,
        ef_search=request.ef_search,
        effective_ef_search=candidates_data["effective_ef_search"],
        candidate_count=len(candidates),
        embedding_ms=round(candidates_data["embedding_ms"], 3),
        lexical_ms=round(candidates_data["lexical_ms"], 3),
        vector_ms=round(candidates_data["vector_ms"], 3),
        rerank_ms=round(rerank_ms, 3),
        rerank_lock_wait_ms=profile["rerank_lock_wait_ms"],
        rerank_inference_ms=profile["rerank_inference_ms"],
        rerank_other_ms=profile["rerank_other_ms"],
        total_ms=round(total_ms, 3),
        results=results,
    )


@router.post(
    "/search/rerank/batched",
    response_model=BatchedRerankSearchResponse,
    tags=["Search"],
)
def rerank_search_batched(request: RerankSearchRequest):
    """Hybrid retrieval with bounded cross-request batching."""

    from backend.api.services.dynamic_batch import (
        BatchQueueFullError,
        BatchTimeoutError,
        get_dynamic_batch_reranker,
    )

    total_start = time.perf_counter()

    candidates_data = retrieve_hybrid_candidates(request)

    candidates = union_candidates(
        candidates_data["lexical_rows"],
        candidates_data["vector_rows"],
    )

    rerank_start = time.perf_counter()

    try:
        ranked, batch_profile = (
            get_dynamic_batch_reranker().submit(
                query=request.query,
                candidates=candidates,
                top_k=request.top_k,
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

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Dynamic Cross-Encoder reranking failed.",
        ) from exc

    rerank_ms = (
        time.perf_counter() - rerank_start
    ) * 1000.0

    total_ms = (
        time.perf_counter() - total_start
    ) * 1000.0

    results = [
        RerankSearchResult(**item)
        for item in ranked
    ]

    log_search_history(
        query_text=request.query,
        search_type="rerank",
        result_count=len(results),
        latency_ms=total_ms,
    )

    return BatchedRerankSearchResponse(
        query=request.query,
        method="cross_encoder_dynamic_batch",
        top_k=request.top_k,
        candidate_k=request.candidate_k,
        ef_search=request.ef_search,
        effective_ef_search=candidates_data["effective_ef_search"],
        candidate_count=len(candidates),
        embedding_ms=round(candidates_data["embedding_ms"], 3),
        lexical_ms=round(candidates_data["lexical_ms"], 3),
        vector_ms=round(candidates_data["vector_ms"], 3),
        rerank_ms=round(rerank_ms, 3),
        rerank_lock_wait_ms=0.0,
        rerank_inference_ms=batch_profile["batch_inference_ms"],
        rerank_other_ms=0.0,
        total_ms=round(total_ms, 3),
        batch_queue_wait_ms=batch_profile["batch_queue_wait_ms"],
        batch_inference_ms=batch_profile["batch_inference_ms"],
        batch_request_count=batch_profile["batch_request_count"],
        batch_pair_count=batch_profile["batch_pair_count"],
        results=results,
    )
