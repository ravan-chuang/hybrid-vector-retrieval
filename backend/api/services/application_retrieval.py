from typing import Any


def lexical_search(
    cur: Any,
    query: str,
    limit: int,
):
    """
    Search the normalized application corpus with PostgreSQL FTS.

    Multiple matching chunks from the same document are aggregated into
    one document-level result using the best chunk score.
    """
    cur.execute(
        """
        WITH ranked_chunks AS (
            SELECT
                d.document_id,
                d.title,
                c.content,
                ts_rank_cd(
                    c.search_vector,
                    websearch_to_tsquery('english', %s)
                ) AS score,
                ROW_NUMBER() OVER (
                    PARTITION BY d.document_id
                    ORDER BY
                        ts_rank_cd(
                            c.search_vector,
                            websearch_to_tsquery('english', %s)
                        ) DESC,
                        c.chunk_id
                ) AS rn
            FROM document_chunks AS c
            JOIN documents AS d
              ON d.document_id = c.document_id
            WHERE c.search_vector @@
                  websearch_to_tsquery('english', %s)
        )
        SELECT
            document_id,
            'app-' || document_id::text AS external_id,
            content,
            score
        FROM ranked_chunks
        WHERE rn = 1
        ORDER BY score DESC, document_id
        LIMIT %s
        """,
        (
            query,
            query,
            query,
            limit,
        ),
    )

    return cur.fetchall()


def vector_search(
    cur: Any,
    vector: str,
    limit: int,
    ef_search: int,
):
    """
    Search application document chunks with HNSW cosine distance.

    HNSW operates at chunk level. The candidate set is then collapsed to
    one best matching chunk per document.
    """
    cur.execute("SET LOCAL enable_seqscan = off")
    cur.execute(
        f"SET LOCAL hnsw.ef_search = {int(ef_search)}"
    )

    # Retrieve extra chunks because one document can own multiple chunks.
    chunk_limit = max(limit * 4, limit)

    cur.execute(
        """
        WITH chunk_candidates AS MATERIALIZED (
            SELECT
                c.chunk_id,
                c.document_id,
                c.content,
                c.embedding <=> %s::vector AS distance
            FROM document_chunks AS c
            WHERE c.embedding IS NOT NULL
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
        ),
        best_per_document AS (
            SELECT DISTINCT ON (document_id)
                document_id,
                content,
                distance
            FROM chunk_candidates
            ORDER BY document_id, distance, chunk_id
        )
        SELECT
            document_id,
            'app-' || document_id::text AS external_id,
            content,
            distance
        FROM best_per_document
        ORDER BY distance, document_id
        LIMIT %s
        """,
        (
            vector,
            vector,
            chunk_limit,
            limit,
        ),
    )

    return cur.fetchall()
