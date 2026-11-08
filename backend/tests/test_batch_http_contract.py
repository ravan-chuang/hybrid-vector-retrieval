"""HTTP contract tests for dynamic Cross-Encoder batching."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.api.main import app


class LocalhostASGI:
    """Simulate a loopback client in HTTP integration tests."""

    def __init__(self, application):
        self.application = application

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            scope = dict(scope)
            scope["client"] = ("127.0.0.1", 54321)

        await self.application(scope, receive, send)


client = TestClient(LocalhostASGI(app))


CANDIDATE = {
    "document_id": 1,
    "external_id": "doc-1",
    "content": "Dynamic batching combines inference requests.",
    "lexical_rank": 1,
    "vector_rank": None,
}


def make_profile(policy):
    return {
        "batch_queue_wait_ms": 1.5,
        "batch_backlog_wait_ms": 0.5,
        "batch_formation_wait_ms": 1.0,
        "batch_inference_ms": 3.0,
        "batch_request_count": 1,
        "batch_pair_count": 1,
        "token_counting_ms": (
            0.8 if policy == "token" else 0.0
        ),
        "batch_token_count": (
            24 if policy == "token" else None
        ),
        "batch_token_utilization": (
            0.24 if policy == "token" else None
        ),
    }


class FakeScheduler:
    def __init__(self, policy):
        self.policy = policy

    def submit(self, query, candidates, top_k):
        ranked = [
            {
                **candidate,
                "cross_encoder_score": 0.95,
            }
            for candidate in candidates[:top_k]
        ]

        return ranked, make_profile(self.policy)


@pytest.mark.parametrize("policy", ["pair", "token"])
def test_frozen_batched_http_contract(monkeypatch, policy):
    monkeypatch.setenv("ENABLE_FROZEN_BENCHMARK", "1")

    with patch(
        "backend.api.routers.benchmark_rerank.get_dynamic_batch_reranker",
        return_value=FakeScheduler(policy),
    ):
        response = client.post(
            "/internal/benchmark/rerank/batched",
            json={
                "query": "What is dynamic batching?",
                "top_k": 1,
                "candidates": [CANDIDATE],
            },
        )

    assert response.status_code == 200, response.text

    data = response.json()

    assert data["pipeline"] == "batched"
    assert data["batch_request_count"] == 1
    assert data["batch_pair_count"] == 1

    if policy == "token":
        assert data["token_counting_ms"] == 0.8
        assert data["batch_token_count"] == 24
        assert data["batch_token_utilization"] == 0.24
    else:
        assert data["token_counting_ms"] == 0.0
        assert data["batch_token_count"] is None
        assert data["batch_token_utilization"] is None

    assert data["results"][0]["document_id"] == 1
