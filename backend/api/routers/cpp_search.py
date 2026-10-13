import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.api import runtime


router = APIRouter()


class CppSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)


class CppSearchResult(BaseModel):
    document_id: int
    score: float


class CppSearchResponse(BaseModel):
    query: str
    method: str
    top_k: int
    retrieval_ms: float
    results: list[CppSearchResult]


@router.post(
    "/search/cpp",
    response_model=CppSearchResponse,
    tags=["Search"],
)
def cpp_search(request: CppSearchRequest):
    engine = runtime.get_cpp_engine()

    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="C++ BM25 engine is not loaded.",
        )

    start = time.perf_counter()

    try:
        matches = engine.search_dense(
            request.query,
            request.top_k,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"C++ BM25 search failed: {exc}",
        ) from exc

    retrieval_ms = (
        time.perf_counter() - start
    ) * 1000.0

    results = [
        CppSearchResult(
            document_id=match.doc_id,
            score=float(match.score),
        )
        for match in matches
    ]

    return CppSearchResponse(
        query=request.query,
        method="cpp_bm25",
        top_k=request.top_k,
        retrieval_ms=round(retrieval_ms, 3),
        results=results,
    )
