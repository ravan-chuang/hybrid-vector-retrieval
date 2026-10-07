import pytest
from pydantic import ValidationError

from backend.api.schemas.search import (
    HybridSearchRequest,
    LexicalSearchRequest,
    VectorSearchRequest,
)


def test_default_corpus_is_benchmark():
    assert VectorSearchRequest(
        query="test"
    ).corpus == "benchmark"

    assert LexicalSearchRequest(
        query="test"
    ).corpus == "benchmark"

    assert HybridSearchRequest(
        query="test"
    ).corpus == "benchmark"


def test_application_corpus_is_accepted():
    assert VectorSearchRequest(
        query="test",
        corpus="application",
    ).corpus == "application"


def test_invalid_corpus_is_rejected():
    with pytest.raises(ValidationError):
        VectorSearchRequest(
            query="test",
            corpus="invalid",
        )


def test_hybrid_candidate_k_must_cover_top_k():
    with pytest.raises(
        ValidationError,
        match=(
            "candidate_k must be greater than "
            "or equal to top_k"
        ),
    ):
        HybridSearchRequest(
            query="test",
            top_k=100,
            candidate_k=10,
        )


def test_hybrid_valid_candidate_k():
    request = HybridSearchRequest(
        query="test",
        top_k=10,
        candidate_k=50,
        ef_search=40,
    )

    assert request.top_k == 10
    assert request.candidate_k == 50
    assert request.ef_search == 40
