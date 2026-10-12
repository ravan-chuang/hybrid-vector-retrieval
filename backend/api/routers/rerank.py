"""Live Cross-Encoder reranking endpoint."""

import time

from fastapi import APIRouter, HTTPException

from backend.api.schemas.search import (
    RerankSearchRequest,
    RerankSearchResponse,
    RerankSearchResult,
)
from backend.api.routers.search import (
    retrieve_hybrid_candidates,
    log_search_history,
)
from backend.api.services.cross_encoder import (
    rerank_candidates,
    union_candidates,
)


router = APIRouter()


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
        ranked = rerank_candidates(
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
        total_ms=round(total_ms, 3),
        results=results,
    )
