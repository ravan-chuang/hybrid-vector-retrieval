from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from backend.api.services.cpp_candidates import retrieve_cpp_candidates


def make_match(doc_id, score):
    return SimpleNamespace(doc_id=doc_id, score=score)


def test_preserves_cpp_bm25_ranking():
    engine = MagicMock()

    engine.search_dense.return_value = [
        make_match(30, 9.5),
        make_match(10, 8.2),
        make_match(20, 7.1),
    ]

    cursor = MagicMock()

    # PostgreSQL deliberately returns a different order.
    cursor.fetchall.return_value = [
        (10, "doc-10", "Content 10"),
        (20, "doc-20", "Content 20"),
        (30, "doc-30", "Content 30"),
    ]

    with patch(
        "backend.api.services.cpp_candidates.runtime.get_cpp_engine",
        return_value=engine,
    ):
        results = retrieve_cpp_candidates(
            cursor,
            query="machine learning",
            limit=3,
        )

    assert [row[0] for row in results] == [30, 10, 20]

    engine.search_dense.assert_called_once_with(
        "machine learning",
        3,
    )

    assert cursor.execute.call_count == 1

    sql, params = cursor.execute.call_args.args

    assert "document_id = ANY(%s)" in sql
    assert params == ([30, 10, 20],)


def test_missing_document_raises_error():
    engine = MagicMock()

    engine.search_dense.return_value = [
        make_match(10, 9.5),
        make_match(20, 8.2),
    ]

    cursor = MagicMock()

    cursor.fetchall.return_value = [
        (10, "doc-10", "Content 10"),
    ]

    with patch(
        "backend.api.services.cpp_candidates.runtime.get_cpp_engine",
        return_value=engine,
    ):
        with pytest.raises(
            RuntimeError,
            match="corpus mismatch",
        ):
            retrieve_cpp_candidates(
                cursor,
                query="machine learning",
                limit=2,
            )


def test_empty_results_skip_database_query():
    engine = MagicMock()
    engine.search_dense.return_value = []

    cursor = MagicMock()

    with patch(
        "backend.api.services.cpp_candidates.runtime.get_cpp_engine",
        return_value=engine,
    ):
        results = retrieve_cpp_candidates(
            cursor,
            query="nonexistent query",
            limit=10,
        )

    assert results == []

    cursor.execute.assert_not_called()


def test_missing_engine_raises_error():
    cursor = MagicMock()

    with patch(
        "backend.api.services.cpp_candidates.runtime.get_cpp_engine",
        return_value=None,
    ):
        with pytest.raises(
            RuntimeError,
            match="not loaded",
        ):
            retrieve_cpp_candidates(
                cursor,
                query="machine learning",
                limit=10,
            )

    cursor.execute.assert_not_called()
