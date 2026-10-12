import pytest
from pydantic import ValidationError

from backend.api.schemas.search import HybridSearchRequest


def test_default_weights_are_balanced():
    request = HybridSearchRequest(query="Manhattan Project")

    assert request.lexical_weight == 0.5
    assert request.dense_weight == 0.5


def test_dense_weighted_configuration():
    request = HybridSearchRequest(
        query="Manhattan Project",
        lexical_weight=0.1,
        dense_weight=0.9,
    )

    assert request.lexical_weight == 0.1
    assert request.dense_weight == 0.9


@pytest.mark.parametrize(
    "lexical_weight,dense_weight",
    [
        (-0.1, 1.1),
        (1.1, -0.1),
        (0.5, 0.6),
        (0.0, 0.0),
    ],
)
def test_invalid_weights_are_rejected(
    lexical_weight,
    dense_weight,
):
    with pytest.raises(ValidationError):
        HybridSearchRequest(
            query="test",
            lexical_weight=lexical_weight,
            dense_weight=dense_weight,
        )


def test_boundary_weights_are_valid():
    lexical_only = HybridSearchRequest(
        query="test",
        lexical_weight=1.0,
        dense_weight=0.0,
    )
    dense_only = HybridSearchRequest(
        query="test",
        lexical_weight=0.0,
        dense_weight=1.0,
    )

    assert lexical_only.lexical_weight == 1.0
    assert dense_only.dense_weight == 1.0
