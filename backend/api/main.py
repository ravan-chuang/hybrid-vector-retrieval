import os
import time
from contextlib import asynccontextmanager

import psycopg
from psycopg_pool import ConnectionPool
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer


DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://retrieval:retrieval@127.0.0.1:5433/retrieval_db",
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SEARCH_TABLE = "scalability_large_documents"

model = None
db_pool = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, db_pool

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # Warm up model so the first API request does not pay
    # the full PyTorch initialization cost.
    model.encode(
        "warmup",
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    print("Embedding model loaded and warmed up.")

    db_pool = ConnectionPool(
        conninfo=DB_URL,
        min_size=2,
        max_size=10,
        open=True,
    )

    db_pool.wait()
    print("PostgreSQL connection pool ready.")

    yield

    db_pool.close()
    db_pool = None
    model = None


app = FastAPI(
    title="Hybrid Vector Retrieval API",
    description=(
        "PostgreSQL + pgvector hybrid retrieval system "
        "supporting lexical, vector, and hybrid search."
    ),
    version="0.6.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class VectorSearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Natural-language search query",
    )

    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of results to return",
    )

    ef_search: int = Field(
        default=40,
        ge=10,
        le=500,
        description="HNSW search breadth",
    )


class SearchResult(BaseModel):
    document_id: int
    external_id: str
    content: str
    distance: float
    similarity: float


class VectorSearchResponse(BaseModel):
    query: str
    method: str
    top_k: int
    ef_search: int
    embedding_ms: float
    retrieval_ms: float
    total_ms: float
    results: list[SearchResult]


def vector_to_pg(vector) -> str:
    return "[" + ",".join(map(str, vector.tolist())) + "]"


@app.get("/")
def root():
    return {
        "name": "Hybrid Vector Retrieval API",
        "version": "0.6.0",
        "status": "running",
    }


@app.get("/health")
def health():
    try:
        with psycopg.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                result = cur.fetchone()[0]

        return {
            "status": "healthy",
            "database": "connected",
            "database_check": result == 1,
            "embedding_model": MODEL_NAME,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Database unavailable: {exc}",
        )


@app.post(
    "/search/vector",
    response_model=VectorSearchResponse,
    tags=["Search"],
)
def vector_search(request: VectorSearchRequest):
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
                # This endpoint explicitly uses HNSW.
                cur.execute("SET LOCAL enable_seqscan = off")

                # SET does not support bind parameters here.
                ef = int(request.ef_search)
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


class LexicalSearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Natural-language or keyword search query",
    )

    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of results to return",
    )


class LexicalSearchResult(BaseModel):
    document_id: int
    external_id: str
    content: str
    score: float


class LexicalSearchResponse(BaseModel):
    query: str
    method: str
    top_k: int
    retrieval_ms: float
    total_ms: float
    results: list[LexicalSearchResult]


@app.post(
    "/search/lexical",
    response_model=LexicalSearchResponse,
    tags=["Search"],
)
def lexical_search(request: LexicalSearchRequest):
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


class HybridSearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Natural-language hybrid search query",
    )

    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of final results",
    )

    candidate_k: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Candidates retrieved from each method",
    )

    ef_search: int = Field(
        default=40,
        ge=10,
        le=500,
        description="HNSW search breadth",
    )

    rrf_k: int = Field(
        default=60,
        ge=1,
        le=200,
        description="RRF rank constant",
    )


class HybridSearchResult(BaseModel):
    document_id: int
    external_id: str
    content: str
    rrf_score: float
    lexical_rank: int | None = None
    vector_rank: int | None = None


class HybridSearchResponse(BaseModel):
    query: str
    method: str
    top_k: int
    candidate_k: int
    ef_search: int
    rrf_k: int
    embedding_ms: float
    lexical_ms: float
    vector_ms: float
    fusion_ms: float
    total_ms: float
    results: list[HybridSearchResult]


@app.post(
    "/search/hybrid",
    response_model=HybridSearchResponse,
    tags=["Search"],
)
def hybrid_search(request: HybridSearchRequest):
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

                cur.execute("SET LOCAL enable_seqscan = off")

                ef = int(request.ef_search)
                cur.execute(
                    f"SET LOCAL hnsw.ef_search = {ef}"
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


class IndexStats(BaseModel):
    name: str
    size: str


class SystemStatsResponse(BaseModel):
    documents: int
    embedding_dimension: int
    embedding_model: str
    database: str
    vector_extension: str
    table_size: str
    indexes_size: str
    total_size: str
    indexes: list[IndexStats]


@app.get(
    "/stats",
    response_model=SystemStatsResponse,
    tags=["System"],
)
def system_stats():
    if db_pool is None:
        raise HTTPException(
            status_code=503,
            detail="Database connection pool is not ready.",
        )

    try:
        with db_pool.connection() as conn:
            with conn.cursor() as cur:

                cur.execute(
                    f"SELECT COUNT(*) FROM {SEARCH_TABLE}"
                )
                documents = cur.fetchone()[0]

                cur.execute(
                    f"""
                    SELECT
                        pg_size_pretty(
                            pg_relation_size('{SEARCH_TABLE}')
                        ),
                        pg_size_pretty(
                            pg_indexes_size('{SEARCH_TABLE}')
                        ),
                        pg_size_pretty(
                            pg_total_relation_size('{SEARCH_TABLE}')
                        )
                    """
                )

                table_size, indexes_size, total_size = (
                    cur.fetchone()
                )

                cur.execute(
                    """
                    SELECT
                        indexname,
                        pg_size_pretty(
                            pg_relation_size(indexname::regclass)
                        )
                    FROM pg_indexes
                    WHERE tablename = %s
                    ORDER BY
                        pg_relation_size(indexname::regclass)
                        DESC
                    """,
                    (SEARCH_TABLE,),
                )

                index_rows = cur.fetchall()

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load system statistics: {exc}",
        )

    indexes = [
        IndexStats(
            name=name,
            size=size,
        )
        for name, size in index_rows
    ]

    return SystemStatsResponse(
        documents=documents,
        embedding_dimension=384,
        embedding_model=MODEL_NAME,
        database="PostgreSQL 17",
        vector_extension="pgvector",
        table_size=table_size,
        indexes_size=indexes_size,
        total_size=total_size,
        indexes=indexes,
    )


# ============================================================
# Document CRUD
# ============================================================

class DocumentCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=1)
    author: str | None = Field(default=None, max_length=255)
    source: str | None = Field(default=None, max_length=500)
    publication_year: int | None = Field(
        default=None,
        ge=0,
        le=2100,
    )


class DocumentUpdate(BaseModel):
    title: str | None = Field(
        default=None,
        min_length=1,
        max_length=500,
    )
    content: str | None = Field(
        default=None,
        min_length=1,
    )
    author: str | None = Field(default=None, max_length=255)
    source: str | None = Field(default=None, max_length=500)
    publication_year: int | None = Field(
        default=None,
        ge=0,
        le=2100,
    )


class DocumentResponse(BaseModel):
    document_id: int
    title: str
    content: str
    author: str | None
    source: str | None
    publication_year: int | None
    created_at: str


@app.post(
    "/documents",
    response_model=DocumentResponse,
    status_code=201,
    tags=["Documents"],
)
def create_document(request: DocumentCreate):
    if model is None or db_pool is None:
        raise HTTPException(
            status_code=503,
            detail="Service is not ready.",
        )

    embedding = model.encode(
        request.content,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    vector = vector_to_pg(embedding)

    try:
        # Both INSERTs are in one transaction.
        with db_pool.connection() as conn:
            with conn.cursor() as cur:

                cur.execute(
                    """
                    INSERT INTO documents (
                        title,
                        author,
                        source,
                        publication_year
                    )
                    VALUES (%s, %s, %s, %s)
                    RETURNING
                        document_id,
                        created_at
                    """,
                    (
                        request.title,
                        request.author,
                        request.source,
                        request.publication_year,
                    ),
                )

                document_id, created_at = cur.fetchone()

                cur.execute(
                    """
                    INSERT INTO document_chunks (
                        document_id,
                        chunk_index,
                        content,
                        embedding
                    )
                    VALUES (%s, 0, %s, %s::vector)
                    """,
                    (
                        document_id,
                        request.content,
                        vector,
                    ),
                )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create document: {exc}",
        )

    return DocumentResponse(
        document_id=document_id,
        title=request.title,
        content=request.content,
        author=request.author,
        source=request.source,
        publication_year=request.publication_year,
        created_at=created_at.isoformat(),
    )


@app.get(
    "/documents",
    response_model=list[DocumentResponse],
    tags=["Documents"],
)
def list_documents(
    limit: int = 20,
    offset: int = 0,
):
    if db_pool is None:
        raise HTTPException(
            status_code=503,
            detail="Database connection pool is not ready.",
        )

    limit = max(1, min(limit, 100))
    offset = max(0, offset)

    with db_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    d.document_id,
                    d.title,
                    c.content,
                    d.author,
                    d.source,
                    d.publication_year,
                    d.created_at
                FROM documents AS d
                LEFT JOIN document_chunks AS c
                  ON c.document_id = d.document_id
                 AND c.chunk_index = 0
                ORDER BY d.document_id DESC
                LIMIT %s
                OFFSET %s
                """,
                (limit, offset),
            )

            rows = cur.fetchall()

    return [
        DocumentResponse(
            document_id=row[0],
            title=row[1],
            content=row[2] or "",
            author=row[3],
            source=row[4],
            publication_year=row[5],
            created_at=row[6].isoformat(),
        )
        for row in rows
    ]


@app.get(
    "/documents/{document_id}",
    response_model=DocumentResponse,
    tags=["Documents"],
)
def get_document(document_id: int):
    if db_pool is None:
        raise HTTPException(
            status_code=503,
            detail="Database connection pool is not ready.",
        )

    with db_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    d.document_id,
                    d.title,
                    c.content,
                    d.author,
                    d.source,
                    d.publication_year,
                    d.created_at
                FROM documents AS d
                LEFT JOIN document_chunks AS c
                  ON c.document_id = d.document_id
                 AND c.chunk_index = 0
                WHERE d.document_id = %s
                """,
                (document_id,),
            )

            row = cur.fetchone()

    if row is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    return DocumentResponse(
        document_id=row[0],
        title=row[1],
        content=row[2] or "",
        author=row[3],
        source=row[4],
        publication_year=row[5],
        created_at=row[6].isoformat(),
    )


@app.put(
    "/documents/{document_id}",
    response_model=DocumentResponse,
    tags=["Documents"],
)
def update_document(
    document_id: int,
    request: DocumentUpdate,
):
    if model is None or db_pool is None:
        raise HTTPException(
            status_code=503,
            detail="Service is not ready.",
        )

    with db_pool.connection() as conn:
        with conn.cursor() as cur:

            cur.execute(
                """
                SELECT
                    d.title,
                    c.content,
                    d.author,
                    d.source,
                    d.publication_year,
                    d.created_at
                FROM documents AS d
                LEFT JOIN document_chunks AS c
                  ON c.document_id = d.document_id
                 AND c.chunk_index = 0
                WHERE d.document_id = %s
                """,
                (document_id,),
            )

            row = cur.fetchone()

            if row is None:
                raise HTTPException(
                    status_code=404,
                    detail="Document not found.",
                )

            (
                old_title,
                old_content,
                old_author,
                old_source,
                old_year,
                created_at,
            ) = row

            title = (
                request.title
                if request.title is not None
                else old_title
            )
            content = (
                request.content
                if request.content is not None
                else old_content
            )
            author = (
                request.author
                if request.author is not None
                else old_author
            )
            source = (
                request.source
                if request.source is not None
                else old_source
            )
            year = (
                request.publication_year
                if request.publication_year is not None
                else old_year
            )

            cur.execute(
                """
                UPDATE documents
                SET
                    title = %s,
                    author = %s,
                    source = %s,
                    publication_year = %s
                WHERE document_id = %s
                """,
                (
                    title,
                    author,
                    source,
                    year,
                    document_id,
                ),
            )

            if request.content is not None:
                embedding = model.encode(
                    content,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )

                vector = vector_to_pg(embedding)

                cur.execute(
                    """
                    UPDATE document_chunks
                    SET
                        content = %s,
                        embedding = %s::vector
                    WHERE
                        document_id = %s
                        AND chunk_index = 0
                    """,
                    (
                        content,
                        vector,
                        document_id,
                    ),
                )

    return DocumentResponse(
        document_id=document_id,
        title=title,
        content=content or "",
        author=author,
        source=source,
        publication_year=year,
        created_at=created_at.isoformat(),
    )


@app.delete(
    "/documents/{document_id}",
    tags=["Documents"],
)
def delete_document(document_id: int):
    if db_pool is None:
        raise HTTPException(
            status_code=503,
            detail="Database connection pool is not ready.",
        )

    with db_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM documents
                WHERE document_id = %s
                RETURNING document_id
                """,
                (document_id,),
            )

            deleted = cur.fetchone()

    if deleted is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    return {
        "deleted": True,
        "document_id": document_id,
    }


# ============================================================
# Search History
# ============================================================

class SearchHistoryItem(BaseModel):
    search_id: int
    query_text: str
    search_type: str
    result_count: int | None
    latency_ms: float | None
    created_at: str


def log_search_history(
    query_text: str,
    search_type: str,
    result_count: int,
    latency_ms: float,
) -> None:
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


@app.get(
    "/search/history",
    response_model=list[SearchHistoryItem],
    tags=["Search History"],
)
def get_search_history(
    limit: int = 20,
    offset: int = 0,
):
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
