from fastapi import APIRouter, HTTPException

from backend.api import runtime
from backend.api.schemas.tags import (
    TagCreate,
    TagResponse,
)


router = APIRouter()


@router.post(
    "/documents/{document_id}/tags",
    response_model=TagResponse,
    tags=["Tags"],
)
def add_document_tag(
    document_id: int,
    request: TagCreate,
):
    db_pool = runtime.get_db_pool()

    if db_pool is None:
        raise HTTPException(
            status_code=503,
            detail="Database connection pool is not ready.",
        )

    tag_name = request.name.strip()

    if not tag_name:
        raise HTTPException(
            status_code=422,
            detail="Tag name cannot be empty.",
        )

    try:
        with db_pool.connection() as conn:
            with conn.cursor() as cur:
                # Ensure the document exists.
                cur.execute(
                    """
                    SELECT 1
                    FROM documents
                    WHERE document_id = %s
                    """,
                    (document_id,),
                )

                if cur.fetchone() is None:
                    raise HTTPException(
                        status_code=404,
                        detail="Document not found.",
                    )

                # Reuse an existing tag or create a new one.
                cur.execute(
                    """
                    INSERT INTO tags (name)
                    VALUES (%s)
                    ON CONFLICT (name)
                    DO UPDATE SET name = EXCLUDED.name
                    RETURNING tag_id, name
                    """,
                    (tag_name,),
                )

                tag_id, name = cur.fetchone()

                # Create the many-to-many association.
                cur.execute(
                    """
                    INSERT INTO document_tags (
                        document_id,
                        tag_id
                    )
                    VALUES (%s, %s)
                    ON CONFLICT (document_id, tag_id)
                    DO NOTHING
                    """,
                    (document_id, tag_id),
                )

        return TagResponse(
            tag_id=tag_id,
            name=name,
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to add tag: {exc}",
        )


@router.get(
    "/documents/{document_id}/tags",
    response_model=list[TagResponse],
    tags=["Tags"],
)
def list_document_tags(document_id: int):
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
                    """
                    SELECT 1
                    FROM documents
                    WHERE document_id = %s
                    """,
                    (document_id,),
                )

                if cur.fetchone() is None:
                    raise HTTPException(
                        status_code=404,
                        detail="Document not found.",
                    )

                cur.execute(
                    """
                    SELECT
                        t.tag_id,
                        t.name
                    FROM tags AS t
                    JOIN document_tags AS dt
                        ON dt.tag_id = t.tag_id
                    WHERE dt.document_id = %s
                    ORDER BY t.name
                    """,
                    (document_id,),
                )

                rows = cur.fetchall()

        return [
            TagResponse(
                tag_id=tag_id,
                name=name,
            )
            for tag_id, name in rows
        ]

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load tags: {exc}",
        )


@router.delete(
    "/documents/{document_id}/tags/{tag_id}",
    tags=["Tags"],
)
def remove_document_tag(
    document_id: int,
    tag_id: int,
):
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
                    """
                    DELETE FROM document_tags
                    WHERE document_id = %s
                      AND tag_id = %s
                    RETURNING document_id
                    """,
                    (document_id, tag_id),
                )

                deleted = cur.fetchone()

                if deleted is None:
                    raise HTTPException(
                        status_code=404,
                        detail="Document-tag association not found.",
                    )

        return {
            "document_id": document_id,
            "tag_id": tag_id,
            "deleted": True,
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to remove tag: {exc}",
        )
