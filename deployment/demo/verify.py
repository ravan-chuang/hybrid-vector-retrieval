"""Verify the portable demo and its database across container recreation."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[2]
COMPOSE = [
    "docker", "compose", "-f", str(ROOT / "deployment/demo/compose.yaml"),
    "--project-name", "hybrid-demo-verification",
]
ORIGIN = "http://127.0.0.1:" + os.environ.get("DEMO_PORT", "5176")


def compose(*args):
    subprocess.run([*COMPOSE, *args], cwd=ROOT, check=True)


def request(path, method="GET", body=None):
    data = json.dumps(body).encode() if body is not None else None
    query = Request(
        ORIGIN + path, data=data, method=method,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    with urlopen(query, timeout=30) as response:
        return json.load(response)


def verify():
    with urlopen(ORIGIN, timeout=10) as response:
        page = response.read().decode()
    bundle_path = re.search(r'src="(/assets/[^"]+\.js)"', page).group(1)
    with urlopen(ORIGIN + bundle_path, timeout=10) as response:
        bundle = response.read().decode()
    assert re.search(r'''["'`]/api["'`]''', bundle)
    assert request("/api/health")["database_check"]
    assert not request("/api/stats")["benchmark_has_documents"]
    print("Verified the built frontend and its same-origin API.", flush=True)

    subprocess.run(
        [sys.executable, str(ROOT / "backend/tests/document_search_lifecycle.py")],
        env={**os.environ, "DOCUMENT_SEARCH_BASE_URL": ORIGIN + "/api"},
        cwd=ROOT, check=True,
    )

    document = request("/api/documents", "POST", {
        "title": "Container persistence test",
        "content": "The silverbeech marker is stored in the demonstration database.",
    })
    document_id = document["document_id"]
    query = {"query": "silverbeech", "corpus": "application"}

    def verify_search():
        results = request("/api/search/hybrid", "POST", query)["results"]
        assert any(row["document_id"] == document_id for row in results)

    verify_search()
    [entry] = request("/api/search/history?limit=1")
    assert entry["query_text"] == query["query"] and entry["search_type"] == "hybrid"

    compose("down")
    compose("up", "--wait", "--wait-timeout", "300")

    assert request(f"/api/documents/{document_id}")["content"] == document["content"]
    assert request("/api/search/history?limit=1") == [entry]
    verify_search()
    print("Containers recreated. The document, search history and retrieval persisted.", flush=True)


if __name__ == "__main__":
    verify()
