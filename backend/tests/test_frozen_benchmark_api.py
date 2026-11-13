"""Tests for local frozen-candidate benchmark endpoints."""

from fastapi.testclient import TestClient

from backend.api.main import app


client = TestClient(app)


def payload():
    return {
        "query": "What is information retrieval?",
        "top_k": 1,
        "candidates": [
            {
                "document_id": 1,
                "external_id": "doc-1",
                "content": "Information retrieval finds relevant documents.",
                "lexical_rank": 1,
                "vector_rank": None,
            }
        ],
    }


def test_benchmark_disabled_by_default(monkeypatch):
    monkeypatch.delenv(
        "ENABLE_FROZEN_BENCHMARK",
        raising=False,
    )

    response = client.post(
        "/internal/benchmark/rerank",
        json=payload(),
    )

    assert response.status_code == 404


def test_duplicate_document_ids_rejected():
    data = payload()
    data["candidates"].append(
        dict(data["candidates"][0])
    )

    response = client.post(
        "/internal/benchmark/rerank",
        json=data,
    )

    assert response.status_code == 422


def test_top_k_cannot_exceed_candidate_count():
    data = payload()
    data["top_k"] = 2

    response = client.post(
        "/internal/benchmark/rerank",
        json=data,
    )

    assert response.status_code == 422
