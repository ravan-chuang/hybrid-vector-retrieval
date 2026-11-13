"""Tests for Cross-Encoder token budget estimation."""

import pytest

from backend.api.services.token_budget import (
    estimate_request_tokens,
)


class FakeTokenizer:
    """Deterministic tokenizer for scheduler unit tests."""

    def __call__(
        self,
        queries,
        documents,
        truncation,
        max_length,
        padding,
        add_special_tokens,
        return_attention_mask,
    ):
        encoded = []

        for query, document in zip(queries, documents):
            query_tokens = query.split()
            document_tokens = document.split()

            token_count = (
                len(query_tokens)
                + len(document_tokens)
                + 3
            )

            if truncation:
                token_count = min(
                    token_count,
                    max_length,
                )

            encoded.append(list(range(token_count)))

        return {"input_ids": encoded}


def test_token_count_basic():
    tokenizer = FakeTokenizer()

    candidates = [
        {"content": "one two three"},
        {"content": "one two three four five"},
    ]

    result = estimate_request_tokens(
        tokenizer=tokenizer,
        query="hello world",
        candidates=candidates,
        max_length=128,
    )

    assert result.pair_count == 2
    assert result.total_tokens == 18
    assert result.max_pair_tokens == 10


def test_token_count_truncation():
    tokenizer = FakeTokenizer()

    result = estimate_request_tokens(
        tokenizer=tokenizer,
        query="hello world",
        candidates=[
            {
                "content": " ".join(
                    ["token"] * 100
                )
            }
        ],
        max_length=32,
    )

    assert result.pair_count == 1
    assert result.total_tokens == 32
    assert result.max_pair_tokens == 32


def test_empty_candidates():
    result = estimate_request_tokens(
        tokenizer=FakeTokenizer(),
        query="hello",
        candidates=[],
        max_length=128,
    )

    assert result.pair_count == 0
    assert result.total_tokens == 0
    assert result.max_pair_tokens == 0


def test_invalid_max_length():
    with pytest.raises(ValueError):
        estimate_request_tokens(
            tokenizer=FakeTokenizer(),
            query="hello",
            candidates=[
                {"content": "document"}
            ],
            max_length=0,
        )
