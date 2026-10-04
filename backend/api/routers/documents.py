from fastapi import APIRouter, HTTPException

from backend.api import runtime
from backend.api.schemas.documents import (
    DocumentCreate,
    DocumentUpdate,
    DocumentResponse,
)
from backend.api.services.embedding import vector_to_pg


router = APIRouter()


@router.post(
    "/documents",
    response_model=DocumentResponse,
    status_code=201,
    tags=["Documents"],
)
def create_document(request: DocumentCreate):
    model = runtime.get_model()
    db_pool = runtime.get_db_pool()

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

@router.get(
    "/documents",
    response_model=list[DocumentResponse],
    tags=["Documents"],
)
def list_documents(
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

@router.get(
    "/documents/{document_id}",
    response_model=DocumentResponse,
    tags=["Documents"],
)
def get_document(document_id: int):
    db_pool = runtime.get_db_pool()

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

@router.put(
    "/documents/{document_id}",
    response_model=DocumentResponse,
    tags=["Documents"],
)
def update_document(
    document_id: int,
    request: DocumentUpdate,
):
    model = runtime.get_model()
    db_pool = runtime.get_db_pool()

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

@router.delete(
    "/documents/{document_id}",
    tags=["Documents"],
)
def delete_document(document_id: int):
    db_pool = runtime.get_db_pool()

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

