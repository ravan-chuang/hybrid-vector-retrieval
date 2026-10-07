import time

from fastapi import APIRouter, HTTPException

from backend.api import runtime
from backend.api.config import SEARCH_TABLE
from backend.api.services.embedding import vector_to_pg
from backend.api.services.application_retrieval import (
    lexical_search as application_lexical_search,
    vector_search as application_vector_search,
)
from backend.api.schemas.search import (
    VectorSearchRequest,
    SearchResult,
    VectorSearchResponse,
    LexicalSearchRequest,
    LexicalSearchResult,
    LexicalSearchResponse,
    HybridSearchRequest,
    HybridSearchResult,
    HybridSearchResponse,
    SearchHistoryItem,
)


router = APIRouter()


@router.post(
    "/search/vector",
    response_model=VectorSearchResponse,
    tags=["Search"],
)
def vector_search(request: VectorSearchRequest):
    model = runtime.get_model()
    db_pool = runtime.get_db_pool()

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Embedding model is not loaded.",
        )

    if db_pool is None:
        raise HTTPException(
            status_code=503,
            detail="Database connection pool is not ready.",
        )

    total_start = time.perf_counter()

    # -------------------------
    # Query embedding
    # -------------------------
    embedding_start = time.perf_counter()

    query_embedding = model.encode(
        request.query,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embedding_ms = (
        time.perf_counter() - embedding_start
    ) * 1000.0

    vector = vector_to_pg(query_embedding)

    # -------------------------
    # PostgreSQL HNSW search
    # -------------------------
    retrieval_start = time.perf_counter()

    try:
        with db_pool.connection() as conn:
            with conn.cursor() as cur:
                if request.corpus == "application":
                    rows = application_vector_search(
                        cur,
                        vector=vector,
                        limit=request.top_k,
                        ef_search=max(
                            request.ef_search,
                            request.top_k,
                        ),
                    )
                else:
                    # Benchmark corpus: explicitly use HNSW.
                    cur.execute("SET LOCAL enable_seqscan = off")

                    ef = max(
                        int(request.ef_search),
                        int(request.top_k),
                    )
                    cur.execute(
                        f"SET LOCAL hnsw.ef_search = {ef}"
                    )

                    cur.execute(
                        f"""
                        SELECT
                            document_id,
                            external_id,
                            content,
                            embedding <=> %s::vector AS distance
                        FROM {SEARCH_TABLE}
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (
                            vector,
                            vector,
                            request.top_k,
                        ),
                    )

                    rows = cur.fetchall()

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Vector search failed: {exc}",
        )

    retrieval_ms = (
        time.perf_counter() - retrieval_start
    ) * 1000.0

    results = []

    for document_id, external_id, content, distance in rows:
        distance = float(distance)

        results.append(
            SearchResult(
                document_id=document_id,
                external_id=external_id,
                content=content,
                distance=distance,
                similarity=1.0 - distance,
            )
        )

    total_ms = (
        time.perf_counter() - total_start
    ) * 1000.0

    log_search_history(
        query_text=request.query,
        search_type="vector",
        result_count=len(results),
        latency_ms=total_ms,
    )

    return VectorSearchResponse(
        query=request.query,
        method="hnsw_cosine",
        top_k=request.top_k,
        ef_search=request.ef_search,
        embedding_ms=round(embedding_ms, 3),
        retrieval_ms=round(retrieval_ms, 3),
        total_ms=round(total_ms, 3),
        results=results,
    )

@router.post(
    "/search/lexical",
    response_model=LexicalSearchResponse,
    tags=["Search"],
)
def lexical_search(request: LexicalSearchRequest):
    db_pool = runtime.get_db_pool()

    if db_pool is None:
        raise HTTPException(
            status_code=503,
            detail="Database connection pool is not ready.",
        )

    total_start = time.perf_counter()
    retrieval_start = time.perf_counter()

    try:
        with db_pool.connection() as conn:
            with conn.cursor() as cur:
                if request.corpus == "application":
                    rows = application_lexical_search(
                        cur,
                        query=request.query,
                        limit=request.top_k,
                    )
                else:
                    cur.execute(
                        f"""
                        WITH query AS (
                            SELECT websearch_to_tsquery(
                                'english',
                                %s
                            ) AS q
                        )
                        SELECT
                            d.document_id,
                            d.external_id,
                            d.content,
                            ts_rank_cd(
                                d.search_vector,
                                query.q
                            ) AS score
                        FROM {SEARCH_TABLE} AS d
                        CROSS JOIN query
                        WHERE d.search_vector @@ query.q
                        ORDER BY score DESC, d.document_id
                        LIMIT %s
                        """,
                        (
                            request.query,
                            request.top_k,
                        ),
                    )

                    rows = cur.fetchall()

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Lexical search failed: {exc}",
        )

    retrieval_ms = (
        time.perf_counter() - retrieval_start
    ) * 1000.0

    results = [
        LexicalSearchResult(
            document_id=document_id,
            external_id=external_id,
            content=content,
            score=float(score),
        )
        for document_id, external_id, content, score in rows
    ]

    total_ms = (
        time.perf_counter() - total_start
    ) * 1000.0

    log_search_history(
        query_text=request.query,
        search_type="keyword",
        result_count=len(results),
        latency_ms=total_ms,
    )

    return LexicalSearchResponse(
        query=request.query,
        method="postgresql_fts_gin",
        top_k=request.top_k,
        retrieval_ms=round(retrieval_ms, 3),
        total_ms=round(total_ms, 3),
        results=results,
    )

@router.post(
    "/search/hybrid",
    response_model=HybridSearchResponse,
    tags=["Search"],
)
def hybrid_search(request: HybridSearchRequest):
    model = runtime.get_model()
    db_pool = runtime.get_db_pool()

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Embedding model is not loaded.",
        )

    if db_pool is None:
        raise HTTPException(
            status_code=503,
            detail="Database connection pool is not ready.",
        )

    total_start = time.perf_counter()

    # -----------------------------------
    # Query embedding
    # -----------------------------------
    embedding_start = time.perf_counter()

    query_embedding = model.encode(
        request.query,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    vector = vector_to_pg(query_embedding)

    embedding_ms = (
        time.perf_counter() - embedding_start
    ) * 1000.0

    try:
        with db_pool.connection() as conn:
            with conn.cursor() as cur:

                # -------------------------
                # Lexical candidates
                # -------------------------
                lexical_start = time.perf_counter()

                if request.corpus == "application":
                    lexical_scored_rows = application_lexical_search(
                        cur,
                        query=request.query,
                        limit=request.candidate_k,
                    )
                    lexical_rows = [
                        row[:3]
                        for row in lexical_scored_rows
                    ]
                else:
                    cur.execute(
                        f"""
                        WITH query AS (
                            SELECT websearch_to_tsquery(
                                'english',
                                %s
                            ) AS q
                        )
                        SELECT
                            d.document_id,
                            d.external_id,
                            d.content
                        FROM {SEARCH_TABLE} AS d
                        CROSS JOIN query
                        WHERE d.search_vector @@ query.q
                        ORDER BY
                            ts_rank_cd(
                                d.search_vector,
                                query.q
                            ) DESC,
                            d.document_id
                        LIMIT %s
                        """,
                        (
                            request.query,
                            request.candidate_k,
                        ),
                    )

                    lexical_rows = cur.fetchall()

                lexical_ms = (
                    time.perf_counter() - lexical_start
                ) * 1000.0

                # -------------------------
                # Vector candidates
                # -------------------------
                vector_start = time.perf_counter()

                effective_ef_search = max(
                    int(request.ef_search),
                    int(request.candidate_k),
                )

                if request.corpus == "application":
                    vector_scored_rows = application_vector_search(
                        cur,
                        vector=vector,
                        limit=request.candidate_k,
                        ef_search=effective_ef_search,
                    )
                    vector_rows = [
                        row[:3]
                        for row in vector_scored_rows
                    ]
                else:
                    cur.execute("SET LOCAL enable_seqscan = off")
                    cur.execute(
                        f"SET LOCAL hnsw.ef_search = {effective_ef_search}"
                    )

                    cur.execute(
                        f"""
                        SELECT
                            document_id,
                            external_id,
                            content
                        FROM {SEARCH_TABLE}
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (
                            vector,
                            request.candidate_k,
                        ),
                    )

                    vector_rows = cur.fetchall()

                vector_ms = (
                    time.perf_counter() - vector_start
                ) * 1000.0

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Hybrid retrieval failed: {exc}",
        )

    # -----------------------------------
    # Reciprocal Rank Fusion
    # -----------------------------------
    fusion_start = time.perf_counter()

    fused = {}

    for rank, row in enumerate(
        lexical_rows,
        start=1,
    ):
        document_id, external_id, content = row

        fused[document_id] = {
            "document_id": document_id,
            "external_id": external_id,
            "content": content,
            "rrf_score": (
                1.0 / (request.rrf_k + rank)
            ),
            "lexical_rank": rank,
            "vector_rank": None,
        }

    for rank, row in enumerate(
        vector_rows,
        start=1,
    ):
        document_id, external_id, content = row

        contribution = (
            1.0 / (request.rrf_k + rank)
        )

        if document_id in fused:
            fused[document_id]["rrf_score"] += contribution
            fused[document_id]["vector_rank"] = rank
        else:
            fused[document_id] = {
                "document_id": document_id,
                "external_id": external_id,
                "content": content,
                "rrf_score": contribution,
                "lexical_rank": None,
                "vector_rank": rank,
            }

    ranked = sorted(
        fused.values(),
        key=lambda x: (
            -x["rrf_score"],
            x["document_id"],
        ),
    )[:request.top_k]

    fusion_ms = (
        time.perf_counter() - fusion_start
    ) * 1000.0

    results = [
        HybridSearchResult(
            document_id=item["document_id"],
            external_id=item["external_id"],
            content=item["content"],
            rrf_score=item["rrf_score"],
            lexical_rank=item["lexical_rank"],
            vector_rank=item["vector_rank"],
        )
        for item in ranked
    ]

    total_ms = (
        time.perf_counter() - total_start
    ) * 1000.0

    log_search_history(
        query_text=request.query,
        search_type="hybrid",
        result_count=len(results),
        latency_ms=total_ms,
    )

    return HybridSearchResponse(
        query=request.query,
        method="rrf_lexical_hnsw",
        top_k=request.top_k,
        candidate_k=request.candidate_k,
        ef_search=request.ef_search,
        rrf_k=request.rrf_k,
        embedding_ms=round(embedding_ms, 3),
        lexical_ms=round(lexical_ms, 3),
        vector_ms=round(vector_ms, 3),
        fusion_ms=round(fusion_ms, 3),
        total_ms=round(total_ms, 3),
        results=results,
    )

def log_search_history(
    query_text: str,
    search_type: str,
    result_count: int,
    latency_ms: float,
) -> None:
    db_pool = runtime.get_db_pool()

    """
    Persist one successful search event.

    Logging failure must not cause the search request itself to fail.
    """
    if db_pool is None:
        return

    try:
        with db_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO search_history (
                        user_id,
                        query_text,
                        search_type,
                        result_count,
                        latency_ms
                    )
                    VALUES (
                        NULL,
                        %s,
                        %s,
                        %s,
                        %s
                    )
                    """,
                    (
                        query_text,
                        search_type,
                        result_count,
                        latency_ms,
                    ),
                )
    except Exception as exc:
        print(f"Search history logging failed: {exc}")

@router.get(
    "/search/history",
    response_model=list[SearchHistoryItem],
    tags=["Search History"],
)
def get_search_history(
    limit: int = 20,
    offset: int = 0,
):
    db_pool = runtime.get_db_pool()

    if db_pool is None:
        raise HTTPException(
            status_code=503,
            detail="Database connection pool is not ready.",
        )

    limit = max(1, min(limit, 100))
    offset = max(0, offset)

    try:
        with db_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        search_id,
                        query_text,
                        search_type,
                        result_count,
                        latency_ms,
                        created_at
                    FROM search_history
                    ORDER BY search_id DESC
                    LIMIT %s
                    OFFSET %s
                    """,
                    (limit, offset),
                )

                rows = cur.fetchall()

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load search history: {exc}",
        )

    return [
        SearchHistoryItem(
            search_id=row[0],
            query_text=row[1],
            search_type=row[2],
            result_count=row[3],
            latency_ms=row[4],
            created_at=row[5].isoformat(),
        )
        for row in rows
    ]

