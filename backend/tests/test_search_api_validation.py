from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.routers.search import router


app = FastAPI()
app.include_router(router)

client = TestClient(app)


def test_hybrid_rejects_candidate_k_below_top_k():
    response = client.post(
        "/search/hybrid",
        json={
            "query": "test",
            "corpus": "application",
            "top_k": 100,
            "candidate_k": 10,
            "ef_search": 40,
        },
    )

    assert response.status_code == 422

    body = response.json()

    assert any(
        "candidate_k must be greater than or equal to top_k"
        in item["msg"]
        for item in body["detail"]
    )


def test_vector_rejects_invalid_corpus():
    response = client.post(
        "/search/vector",
        json={
            "query": "test",
            "corpus": "invalid",
        },
    )

    assert response.status_code == 422


def test_lexical_rejects_invalid_corpus():
    response = client.post(
        "/search/lexical",
        json={
            "query": "test",
            "corpus": "invalid",
        },
    )

    assert response.status_code == 422


def test_hybrid_rejects_invalid_corpus():
    response = client.post(
        "/search/hybrid",
        json={
            "query": "test",
            "corpus": "invalid",
        },
    )

    assert response.status_code == 422
