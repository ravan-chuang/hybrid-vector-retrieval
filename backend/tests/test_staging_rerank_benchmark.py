from io import BytesIO
import json
import time
from types import SimpleNamespace

from deployment.staging.benchmark_rerank import read_pool, request_sender


def test_staging_replay_pool_matches_manifest():
    pool, payloads = read_pool()
    assert len(payloads) == len(set(pool["query_ids"])) == 120
    example = json.loads(payloads[0]["body"])
    assert example["corpus"] == "benchmark"
    assert example["candidate_k"] == 50
    assert example["ef_search"] == 160


def test_staging_result_without_request_id_counts_as_failure():
    class Response(BytesIO):
        status = 200
        headers = {}

    def open_request(request, timeout):
        return Response(json.dumps({
            "results": [{"document_id": doc_id} for doc_id in range(10)],
            "effective_ef_search": 160,
            "batch_id": "one",
        }).encode())

    now = time.perf_counter()
    send = request_sender(SimpleNamespace(open=open_request), "Basic example", "batched")
    row = send("https://example.test", "/api/search/rerank/batched",
               {"query_id": "42", "body": b"{}"}, 0, now, now, 1)
    assert row["status_code"] == 200
    assert not row["success"]
    assert row["error"] == "ValueError"
