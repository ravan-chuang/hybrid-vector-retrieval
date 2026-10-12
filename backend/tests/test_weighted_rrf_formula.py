import pytest

from backend.api.services.weighted_rrf import weighted_rrf_score


def test_balanced_rrf():
    score = weighted_rrf_score(
        lexical_rank=1,
        vector_rank=2,
        lexical_weight=0.5,
        dense_weight=0.5,
    )

    assert score == pytest.approx(0.5 / 61 + 0.5 / 62)


def test_dense_weighted_rrf():
    score = weighted_rrf_score(
        lexical_rank=1,
        vector_rank=2,
        lexical_weight=0.1,
        dense_weight=0.9,
    )

    assert score == pytest.approx(0.1 / 61 + 0.9 / 62)


def test_lexical_only():
    score = weighted_rrf_score(
        lexical_rank=1,
        vector_rank=None,
        lexical_weight=0.1,
        dense_weight=0.9,
    )

    assert score == pytest.approx(0.1 / 61)


def test_dense_only():
    score = weighted_rrf_score(
        lexical_rank=None,
        vector_rank=1,
        lexical_weight=0.1,
        dense_weight=0.9,
    )

    assert score == pytest.approx(0.9 / 61)


def test_no_ranks():
    assert weighted_rrf_score(None, None) == 0.0


def test_dense_weight_changes_ranking():
    lexical_favored = weighted_rrf_score(
        lexical_rank=1,
        vector_rank=50,
        lexical_weight=0.5,
        dense_weight=0.5,
    )

    dense_favored = weighted_rrf_score(
        lexical_rank=50,
        vector_rank=1,
        lexical_weight=0.1,
        dense_weight=0.9,
    )

    assert dense_favored > lexical_favored
