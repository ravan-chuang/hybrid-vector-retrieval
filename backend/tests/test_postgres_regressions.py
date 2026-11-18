"""Run against the disposable database created by the CI lifecycle job."""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg_pool import ConnectionPool

from backend.api import runtime
from backend.api.main import app
from backend.api.services.application_bm25 import ApplicationBM25


DATABASE_URL = os.getenv("REGRESSION_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DATABASE_URL, reason="Requires a disposable pgvector database")


@pytest.fixture
def postgres_app(monkeypatch):
    application_name = f"audit_{uuid4().hex}"
    with ConnectionPool(
        DATABASE_URL, min_size=2, max_size=2,
        kwargs={"application_name": application_name},
    ) as pool:
        pool.wait()
        monkeypatch.setattr(runtime, "get_db_pool", lambda: pool)
        monkeypatch.setattr(runtime, "get_model", object)
        client = TestClient(app)
        try:
            yield client, application_name
        finally:
            client.close()


def test_concurrent_partial_updates_and_explicit_null(postgres_app):
    client, application_name = postgres_app
    with psycopg.connect(DATABASE_URL) as locker, psycopg.connect(DATABASE_URL, autocommit=True) as observer:
        document_id = locker.execute(
            """
            INSERT INTO documents (title, author, source, publication_year)
            VALUES ('audit', 'old author', 'old source', 0)
            RETURNING document_id
            """
        ).fetchone()[0]
        locker.commit()
        route = f"/documents/{document_id}"

        try:
            locker.execute("SELECT document_id FROM documents WHERE document_id = %s FOR UPDATE", (document_id,))
            with ThreadPoolExecutor(max_workers=2) as executor:
                updates = [
                    executor.submit(client.put, route, json=payload)
                    for payload in ({"author": "new author"}, {"source": "new source"})
                ]
                try:
                    deadline = time.monotonic() + 8
                    while time.monotonic() < deadline:
                        waiting = observer.execute(
                            """
                            SELECT count(*) FROM pg_stat_activity
                            WHERE application_name = %s AND wait_event_type = 'Lock'
                            """,
                            (application_name,),
                        ).fetchone()[0]
                        if waiting == 2:
                            break
                        time.sleep(0.05)
                    else:
                        pytest.fail("Both requests did not reach the held document lock")
                finally:
                    locker.commit()
                assert [update.result(timeout=8).status_code for update in updates] == [200, 200]

            document = client.get(route).json()
            assert (document["author"], document["source"], document["publication_year"]) == (
                "new author", "new source", 0,
            )
            cleared = client.put(route, json={"author": None, "publication_year": None})
            assert cleared.status_code == 200
            assert cleared.json()["author"] is None
            assert cleared.json()["publication_year"] is None
            assert cleared.json()["source"] == "new source"
        finally:
            observer.execute("DELETE FROM documents WHERE document_id = %s", (document_id,))


def test_fresh_and_migrated_schema_persist_rerank_history(postgres_app):
    client, _ = postgres_app
    query = f"audit rerank {uuid4().hex}"
    migration = Path("database/15_search_history_rerank.sql").read_text()

    def assert_rerank_history():
        with patch("backend.api.routers.rerank.retrieve_hybrid_candidates", return_value={
            "lexical_rows": [], "vector_rows": [], "effective_ef_search": 160,
            "embedding_ms": 0.0, "lexical_ms": 0.0, "vector_ms": 0.0,
        }):
            for path in ("/search/rerank", "/search/rerank/batched"):
                assert client.post(path, json={"query": query, "corpus": "application"}).status_code == 200

        history = client.get("/search/history").json()
        assert [row["search_type"] for row in history if row["query_text"] == query] == ["rerank", "rerank"]

    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        try:
            assert_rerank_history()
            conn.execute("DELETE FROM search_history WHERE query_text = %s", (query,))
            conn.execute("""
                ALTER TABLE search_history
                    DROP CONSTRAINT search_history_search_type_check,
                    ADD CONSTRAINT search_history_search_type_check
                        CHECK (search_type IN ('keyword', 'vector', 'hybrid'))
            """)
            conn.execute(migration)
            conn.execute(migration)
            assert_rerank_history()
        finally:
            conn.execute("DELETE FROM search_history WHERE query_text = %s", (query,))


def test_empty_benchmark_table_reports_zero_documents(postgres_app):
    client, _ = postgres_app
    stats = client.get("/stats")
    assert stats.status_code == 200
    assert stats.json()["documents"] == 0
    assert stats.json()["benchmark_has_documents"] is False


def test_cpp_lexical_search_persists_keyword_history(postgres_app, monkeypatch):
    client, _ = postgres_app
    query = f"audit BM25 {uuid4().hex}"
    monkeypatch.setattr(runtime, "get_cpp_engine", object)
    try:
        with patch("backend.api.services.cpp_candidates.retrieve_cpp_candidates",
                   return_value=[(1, "msmarco-0", "first passage")]):
            response = client.post("/search/cpp/lexical", json={"query": query})
        assert response.status_code == 200
        history = client.get("/search/history").json()
        assert [(row["search_type"], row["result_count"]) for row in history
                if row["query_text"] == query] == [("keyword", 1)]
    finally:
        with psycopg.connect(DATABASE_URL) as conn:
            conn.execute("DELETE FROM search_history WHERE query_text = %s", (query,))


def test_application_bm25_refreshes_after_document_changes(postgres_app, monkeypatch):
    from backend import retrieval_cpp

    class EmbeddingModel:
        def encode(self, *_args, **_kwargs):
            return np.ones(384)

    client, _ = postgres_app
    monkeypatch.setattr(runtime, "get_model", EmbeddingModel)
    monkeypatch.setattr(runtime, "application_bm25", ApplicationBM25(retrieval_cpp.SearchEngine))
    old_term = f"orchidledger{uuid4().hex}"
    new_term = f"updatedledger{uuid4().hex}"
    documents = []

    def search(query, **options):
        response = client.post("/search/lexical", json={
            "corpus": "application", "lexical_backend": "cpp_bm25",
            "query": query, "top_k": 2, **options,
        })
        assert response.status_code == 200, response.text
        return response.json()

    def create(content):
        response = client.post("/documents", json={"title": "BM25 test", "content": content})
        assert response.status_code == 201, response.text
        doc_id = response.json()["document_id"]
        documents.append(doc_id)
        return doc_id

    try:
        assert search(old_term)["results"] == []
        first = create(old_term)
        second = create(f"Additional background words {old_term}")
        with psycopg.connect(DATABASE_URL) as conn:
            conn.execute(
                "INSERT INTO document_chunks (document_id, chunk_index, content) VALUES (%s, 1, %s)",
                (first, old_term),
            )
        found = search(old_term)
        assert found["method"] == "cpp_bm25"
        assert [hit["document_id"] for hit in found["results"]] == [first, second]

        assert client.put(f"/documents/{second}", json={"content": new_term}).status_code == 200
        assert [hit["document_id"] for hit in search(old_term)["results"]] == [first]
        assert [(hit["external_id"], hit["content"]) for hit in search(new_term)["results"]] == [
            (f"app-{second}", new_term),
        ]
        assert search(new_term, lexical_backend="postgresql_fts")["method"] == "postgresql_fts_gin"

        monkeypatch.setattr(runtime, "application_bm25", ApplicationBM25(retrieval_cpp.SearchEngine))
        assert [hit["document_id"] for hit in search(new_term)["results"]] == [second]
        assert client.delete(f"/documents/{second}").status_code == 200
        assert search(new_term)["results"] == []
    finally:
        with psycopg.connect(DATABASE_URL) as conn:
            conn.execute("DELETE FROM documents WHERE document_id = ANY(%s)", (documents,))
            conn.execute("DELETE FROM search_history WHERE query_text = ANY(%s)", ([old_term, new_term],))
