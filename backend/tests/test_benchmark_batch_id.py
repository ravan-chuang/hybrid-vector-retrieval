"""Verify benchmark client preserves batch identity."""

import sys
from pathlib import Path
from unittest.mock import patch
import io
import json
import time

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "scripts"),
)

from benchmark_open_loop_http import send_request


class FakeHTTPResponse:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def test_benchmark_client_preserves_batch_id():
    response = {
        "results": [],
        "batch_id": "scheduler-123",
        "batch_request_count": 2,
        "batch_pair_count": 100,
        "batch_queue_wait_ms": 5.0,
        "batch_inference_ms": 80.0,
    }

    def fake_json_load(_):
        return response

    payload = {
        "query": "test",
        "candidates": [
            {"content": "test document"}
        ],
    }

    now = time.perf_counter()

    with (
        patch(
            "benchmark_open_loop_http.urllib.request.urlopen",
            return_value=FakeHTTPResponse(),
        ),
        patch(
            "benchmark_open_loop_http.json.load",
            side_effect=fake_json_load,
        ),
    ):
        record = send_request(
            base_url="http://127.0.0.1:8001",
            endpoint="/internal/benchmark/rerank/batched",
            payload=payload,
            request_id=1,
            scheduled_at=now,
            experiment_start=now,
            timeout=5,
        )

    assert record["success"] is True

    assert (
        record["profile"]["batch_id"]
        == "scheduler-123"
    )
