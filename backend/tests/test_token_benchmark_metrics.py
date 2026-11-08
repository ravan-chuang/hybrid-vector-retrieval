"""Regression tests for token-aware benchmark aggregation."""

import importlib.util
from pathlib import Path
from unittest.mock import patch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "benchmark_frozen_http.py"
)

spec = importlib.util.spec_from_file_location(
    "benchmark_frozen_http",
    SCRIPT,
)

benchmark_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark_module)


def fake_request_once(base_url, endpoint, payload):
    is_batched = endpoint.endswith("/batched")

    result = {
        "batch_queue_wait_ms": 3.0,
        "batch_backlog_wait_ms": 1.0,
        "batch_formation_wait_ms": 2.0,
        "batch_inference_ms": 5.0,
        "batch_request_count": 2,
        "batch_pair_count": 20,
        "token_counting_ms": 0.5,
        "batch_token_count": 1000,
        "batch_token_utilization": 0.25,
    }

    return {
        "latency_ms": 10.0,
        "result": result if is_batched else {},
        "candidate_count": len(payload["candidates"]),
    }


def test_token_benchmark_metrics():
    payloads = [
        {
            "query": "test",
            "candidates": [
                {"document_id": 1, "content": "test"}
            ],
        }
        for _ in range(4)
    ]

    with patch.object(
        benchmark_module,
        "request_once",
        side_effect=fake_request_once,
    ):
        summary = benchmark_module.benchmark(
            base_url="http://127.0.0.1:8000",
            endpoint=benchmark_module.ENDPOINTS["batched"],
            payloads=payloads,
            concurrency=2,
        )

    assert summary["requests"] == 4
    assert summary["mean_token_counting_ms"] == 0.5
    assert summary["p95_token_counting_ms"] == 0.5
    assert summary["mean_batch_tokens"] == 1000
    assert summary["mean_token_utilization"] == 0.25
    assert summary["p95_queue_wait_ms"] == 3.0
    assert summary["queue_decomposition_max_error_ms"] == 0.0
