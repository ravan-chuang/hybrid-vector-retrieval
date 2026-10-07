import math

import pytest

from backend.evaluate_relevance import (
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)


def test_reciprocal_rank_first_relevant_at_rank_two():
    ranked = ["30", "10", "40", "20"]
    relevant = {"10", "20"}

    assert reciprocal_rank_at_k(
        ranked, relevant, k=10
    ) == pytest.approx(0.5)


def test_reciprocal_rank_no_relevant_result():
    ranked = ["30", "40", "50"]
    relevant = {"10", "20"}

    assert reciprocal_rank_at_k(
        ranked, relevant, k=10
    ) == 0.0


def test_recall_at_k():
    ranked = ["30", "10", "40", "20"]
    relevant = {"10", "20"}

    assert recall_at_k(
        ranked, relevant, k=10
    ) == pytest.approx(1.0)

    assert recall_at_k(
        ranked, relevant, k=2
    ) == pytest.approx(0.5)


def test_ndcg_at_k():
    ranked = ["30", "10", "40", "20"]
    relevant = {"10", "20"}

    dcg = (
        1.0 / math.log2(2 + 1)
        + 1.0 / math.log2(4 + 1)
    )

    idcg = (
        1.0 / math.log2(1 + 1)
        + 1.0 / math.log2(2 + 1)
    )

    expected = dcg / idcg

    assert ndcg_at_k(
        ranked, relevant, k=10
    ) == pytest.approx(expected)


def test_metrics_respect_cutoff():
    ranked = ["30", "40", "50", "10"]
    relevant = {"10"}

    assert reciprocal_rank_at_k(
        ranked, relevant, k=3
    ) == 0.0

    assert recall_at_k(
        ranked, relevant, k=3
    ) == 0.0

    assert ndcg_at_k(
        ranked, relevant, k=3
    ) == 0.0


def test_empty_relevance_set():
    ranked = ["10", "20"]

    assert reciprocal_rank_at_k(
        ranked, set(), k=10
    ) == 0.0

    assert recall_at_k(
        ranked, set(), k=10
    ) == 0.0

    assert ndcg_at_k(
        ranked, set(), k=10
    ) == 0.0
