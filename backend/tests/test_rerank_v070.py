import pytest
from pydantic import ValidationError

from backend.api.schemas.search import RerankSearchRequest


def test_cpp_backend_accepted():
    request = RerankSearchRequest(
        query="information retrieval",
        lexical_backend="cpp_bm25",
        stopword_filter=True,
    )

    assert request.lexical_backend == "cpp_bm25"
    assert request.stopword_filter is True


def test_default_backend_preserved():
    request = RerankSearchRequest(
        query="information retrieval"
    )

    assert request.lexical_backend == "postgres_fts"
    assert request.stopword_filter is False


def test_invalid_backend_rejected():
    with pytest.raises(ValidationError):
        RerankSearchRequest(
            query="information retrieval",
            lexical_backend="unknown_backend",
        )
