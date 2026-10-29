import pytest
from pydantic import ValidationError

from backend.api.schemas.search import HybridSearchRequest


def test_default_backend():
    request = HybridSearchRequest(
        query="machine learning",
    )

    assert request.lexical_backend == "postgresql_fts"


def test_cpp_backend():
    request = HybridSearchRequest(
        query="machine learning",
        corpus="benchmark",
        lexical_backend="cpp_bm25",
        top_k=5,
        candidate_k=50,
    )

    assert request.lexical_backend == "cpp_bm25"


def test_cpp_application_rejected():
    with pytest.raises(ValidationError):
        HybridSearchRequest(
            query="machine learning",
            corpus="application",
            lexical_backend="cpp_bm25",
        )


def test_invalid_backend_rejected():
    with pytest.raises(ValidationError):
        HybridSearchRequest(
            query="machine learning",
            lexical_backend="invalid_backend",
        )


def test_candidate_k_less_than_top_k_rejected():
    with pytest.raises(ValidationError):
        HybridSearchRequest(
            query="machine learning",
            lexical_backend="cpp_bm25",
            top_k=100,
            candidate_k=50,
        )
