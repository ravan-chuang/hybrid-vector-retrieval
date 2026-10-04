from fastapi import APIRouter, HTTPException

from backend.api import runtime
from backend.api.config import MODEL_NAME, SEARCH_TABLE
from backend.api.schemas.system import (
    IndexStats,
    SystemStatsResponse,
)


router = APIRouter()


@router.get("/")
def root():
    return {
        "name": "Hybrid Vector Retrieval API",
        "version": "0.6.0",
        "status": "running",
    }


@router.get("/health", tags=["System"])
def health():
    db_pool = runtime.get_db_pool()

    if db_pool is None:
        raise HTTPException(
            status_code=503,
            detail="Database connection pool is not ready.",
        )

    try:
        with db_pool.connection() as conn:
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


@router.get(
    "/stats",
    response_model=SystemStatsResponse,
    tags=["System"],
)
def system_stats():
    db_pool = runtime.get_db_pool()

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
