"""Batched reranking profile contract tests."""

from backend.api.schemas.search import BatchedRerankSearchResponse
from backend.api.services.dynamic_batch import DynamicBatchReranker


EXPECTED_FIELDS = {
    "batch_queue_wait_ms",
    "batch_backlog_wait_ms",
    "batch_formation_wait_ms",
    "batch_inference_ms",
    "batch_request_count",
    "batch_pair_count",
    "token_counting_ms",
    "batch_token_count",
    "batch_token_utilization",
}


def test_empty_token_profile_contract():
    scheduler = DynamicBatchReranker(batch_policy="token")

    try:
        ranked, profile = scheduler.submit(
            query="test",
            candidates=[],
            top_k=10,
        )

        assert ranked == []
        assert EXPECTED_FIELDS <= profile.keys()
        assert profile["batch_token_count"] == 0
        assert profile["batch_token_utilization"] == 0.0
        assert profile["token_counting_ms"] == 0.0

    finally:
        scheduler.close()


def test_empty_pair_profile_contract():
    scheduler = DynamicBatchReranker(batch_policy="pair")

    try:
        ranked, profile = scheduler.submit(
            query="test",
            candidates=[],
            top_k=10,
        )

        assert ranked == []
        assert EXPECTED_FIELDS <= profile.keys()
        assert profile["batch_token_count"] is None
        assert profile["batch_token_utilization"] is None

    finally:
        scheduler.close()


def test_http_schema_declares_profile_fields():
    fields = set(BatchedRerankSearchResponse.model_fields)

    assert EXPECTED_FIELDS <= fields

    assert (
        BatchedRerankSearchResponse
        .model_fields["batch_token_count"]
        .default is None
    )
