"""Weighted Reciprocal Rank Fusion utilities."""


def weighted_rrf_score(
    lexical_rank: int | None,
    vector_rank: int | None,
    lexical_weight: float = 0.5,
    dense_weight: float = 0.5,
    rrf_k: int = 60,
) -> float:
    """Calculate weighted RRF score from available ranks."""

    score = 0.0

    if lexical_rank is not None:
        score += lexical_weight / (rrf_k + lexical_rank)

    if vector_rank is not None:
        score += dense_weight / (rrf_k + vector_rank)

    return score
