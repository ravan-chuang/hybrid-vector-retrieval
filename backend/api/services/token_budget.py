"""Token counting utilities for Cross-Encoder scheduling."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenBudgetEstimate:
    pair_count: int
    total_tokens: int
    max_pair_tokens: int


def estimate_request_tokens(
    tokenizer,
    query: str,
    candidates: list[dict],
    max_length: int,
) -> TokenBudgetEstimate:
    """Count encoded tokens for query-document pairs."""

    if max_length < 1:
        raise ValueError("max_length must be positive.")

    if not candidates:
        return TokenBudgetEstimate(
            pair_count=0,
            total_tokens=0,
            max_pair_tokens=0,
        )

    queries = [query] * len(candidates)
    documents = [
        candidate["content"]
        for candidate in candidates
    ]

    encoded = tokenizer(
        queries,
        documents,
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=True,
        return_attention_mask=False,
    )

    lengths = [
        len(input_ids)
        for input_ids in encoded["input_ids"]
    ]

    return TokenBudgetEstimate(
        pair_count=len(lengths),
        total_tokens=sum(lengths),
        max_pair_tokens=max(lengths),
    )
