"""HTTP integration tests for the production batched rerank endpoint."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.api.services.dynamic_batch import (
    BatchQueueFullError,
    BatchTimeoutError,
)


client = TestClient(app)

ENDPOINT = "/search/rerank/batched"

PAYLOAD = {
    "query": "What is dynamic batching?",
    "corpus": "benchmark",
    "top_k": 1,
    "candidate_k": 10,
    "ef_search": 160,
}

CANDIDATE = {
    "document_id": 1,
    "external_id": "doc-1",
    "content": "Dynamic batching combines inference requests.",
    "lexical_rank": 1,
    "vector_rank": None,
}


def fake_retrieval(request):
    return {
        "lexical_rows": [CANDIDATE],
        "vector_rows": [],
        "effective_ef_search": request.ef_search,
        "embedding_ms": 1.0,
        "lexical_ms": 2.0,
        "vector_ms": 3.0,
    }


def make_profile(policy):
    return {
        "batch_id": f"test-scheduler-{policy}-batch-1",
        "batch_queue_wait_ms": 1.5,
        "batch_backlog_wait_ms": 0.5,
        "batch_formation_wait_ms": 1.0,
        "batch_inference_ms": 3.0,
        "batch_request_count": 1,
        "batch_pair_count": 1,
        "token_counting_ms": 0.8 if policy == "token" else 0.0,
        "batch_token_count": 24 if policy == "token" else None,
        "batch_token_utilization": 0.24 if policy == "token" else None,
    }


class FakeScheduler:
    def __init__(self, policy="pair", error=None):
        self.policy = policy
        self.error = error

    def submit(self, query, candidates, top_k):
        if self.error is not None:
            raise self.error

        ranked = [
            {
                **candidate,
                "cross_encoder_score": 0.95,
            }
            for candidate in candidates[:top_k]
        ]

        return ranked, make_profile(self.policy)


def request_with_scheduler(scheduler):
    with (
        patch(
            "backend.api.routers.rerank.retrieve_hybrid_candidates",
            side_effect=fake_retrieval,
        ),
        patch(
            "backend.api.routers.rerank.union_candidates",
            return_value=[CANDIDATE],
        ),
        patch(
            "backend.api.routers.rerank.log_search_history",
        ),
        patch(
            "backend.api.services.dynamic_batch.get_dynamic_batch_reranker",
            return_value=scheduler,
        ),
    ):
        return client.post(ENDPOINT, json=PAYLOAD)


@pytest.mark.parametrize("policy", ["pair", "token"])
def test_production_batched_http_profile(policy):
    response = request_with_scheduler(FakeScheduler(policy))

    assert response.status_code == 200, response.text

    data = response.json()

    assert data["method"] == "cross_encoder_dynamic_batch"
    assert data["candidate_count"] == 1
    assert data["batch_id"] == f"test-scheduler-{policy}-batch-1"
    assert data["batch_request_count"] == 1
    assert data["batch_pair_count"] == 1

    assert data["batch_queue_wait_ms"] == 1.5
    assert data["batch_backlog_wait_ms"] == 0.5
    assert data["batch_formation_wait_ms"] == 1.0
    assert data["batch_inference_ms"] == 3.0

    if policy == "token":
        assert data["token_counting_ms"] == 0.8
        assert data["batch_token_count"] == 24
        assert data["batch_token_utilization"] == 0.24
    else:
        assert data["token_counting_ms"] == 0.0
        assert data["batch_token_count"] is None
        assert data["batch_token_utilization"] is None

    assert data["results"][0]["document_id"] == 1
    assert data["results"][0]["cross_encoder_score"] == 0.95


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (BatchQueueFullError("Queue full"), 503),
        (BatchTimeoutError("Batch timed out"), 504),
    ],
)
def test_production_batched_http_errors(error, expected_status):
    response = request_with_scheduler(
        FakeScheduler(error=error)
    )

    assert response.status_code == expected_status
