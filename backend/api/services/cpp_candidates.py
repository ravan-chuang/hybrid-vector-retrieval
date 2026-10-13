from backend.api import runtime
from backend.api.config import SEARCH_TABLE


def retrieve_cpp_candidates(
    cur,
    query: str,
    limit: int,
):
    """
    Retrieve BM25-ranked candidates using C++ and
    fetch their metadata from PostgreSQL.

    Returns:
        list[(document_id, external_id, content)]
    """

    engine = runtime.get_cpp_engine()

    if engine is None:
        raise RuntimeError("C++ BM25 engine is not loaded.")

    matches = engine.search_dense(query, limit)

    if not matches:
        return []

    ranked_ids = [
        int(match.doc_id)
        for match in matches
    ]

    cur.execute(
        f"""
        SELECT
            document_id,
            external_id,
            content
        FROM {SEARCH_TABLE}
        WHERE document_id = ANY(%s)
        """,
        (ranked_ids,),
    )

    rows = cur.fetchall()

    rows_by_id = {
        int(row[0]): row
        for row in rows
    }

    missing_ids = [
        doc_id
        for doc_id in ranked_ids
        if doc_id not in rows_by_id
    ]

    if missing_ids:
        raise RuntimeError(
            "C++/PostgreSQL corpus mismatch: "
            f"{len(missing_ids)} missing document IDs; "
            f"examples={missing_ids[:5]}"
        )

    # Preserve C++ BM25 ranking for RRF.
    return [
        rows_by_id[doc_id]
        for doc_id in ranked_ids
    ]
