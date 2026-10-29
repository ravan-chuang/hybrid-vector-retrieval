import time

from backend.api import runtime
from backend.api.services.query_processing import filter_stopwords
from backend.api.config import SEARCH_TABLE


def retrieve_cpp_candidates(
    cur,
    query: str,
    limit: int,
    stopword_filter: bool = False,
    profile: dict | None = None,
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

    lexical_query = query

    if stopword_filter:
        filtered_query = filter_stopwords(query)

        if filtered_query:
            lexical_query = filtered_query

    bm25_start = time.perf_counter()

    matches = engine.search_dense(lexical_query, limit)

    bm25_search_ms = (
        time.perf_counter() - bm25_start
    ) * 1000.0

    if profile is not None:
        profile["bm25_search_ms"] = bm25_search_ms
        profile["metadata_fetch_ms"] = 0.0

    if not matches:
        return []

    ranked_ids = [
        int(match.doc_id)
        for match in matches
    ]

    metadata_start = time.perf_counter()

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

    metadata_fetch_ms = (
        time.perf_counter() - metadata_start
    ) * 1000.0

    if profile is not None:
        profile["metadata_fetch_ms"] = metadata_fetch_ms

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
