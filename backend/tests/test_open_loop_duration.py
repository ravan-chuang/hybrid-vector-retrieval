"""Regression tests for capacity accounting and response validation."""

import io
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import benchmark_open_loop_http
from benchmark_open_loop_duration import calculate_capacity_metrics, summarize_measurement


def test_capacity_counts_late_dispatch_separately():
    records = [
        {"scheduled_offset_ms": 0, "dispatch_offset_ms": 100, "completed_offset_ms": 200, "success": True},
        {"scheduled_offset_ms": 100, "dispatch_offset_ms": 200, "completed_offset_ms": 300, "success": False},
        {"scheduled_offset_ms": 700, "dispatch_offset_ms": 800, "completed_offset_ms": 1000, "success": True},
        {"scheduled_offset_ms": 900, "dispatch_offset_ms": 1000, "completed_offset_ms": 1200, "success": True},
    ]

    assert calculate_capacity_metrics(records, 1) == {
        "max_inflight": 1,
        "inflight_at_window_end": 1,
        "queued_at_window_end": 1,
        "completed_in_window": 2,
        "completed_in_drain": 2,
        "completion_throughput_rps": 2,
        "window_goodput_rps": 1,
    }


@pytest.mark.parametrize("body", [b"bad json", b"{}", b'{"results":null}'])
def test_http_200_requires_a_valid_rerank_response(monkeypatch, body):
    class Response(io.BytesIO):
        status = 200

    monkeypatch.setattr(benchmark_open_loop_http.urllib.request, "urlopen", lambda *a, **kw: Response(body))
    now = time.perf_counter()
    record = benchmark_open_loop_http.send_request(
        "http://127.0.0.1", "/rerank", {"candidates": []}, 1, now, now, 1,
    )
    assert record["status_code"] == 200
    assert record["success"] is False
    assert record["error"]
    assert record["profile"] is None


def test_batch_summary_counts_unique_batch_ids():
    profiles = [
        {"batch_id": "one", "batch_inference_ms": 10},
        {"batch_id": "one", "batch_inference_ms": 10},
        {"batch_id": "two", "batch_inference_ms": 40},
    ]
    records = [
        dict(success=True, status_code=200, profile=profile, http_latency_ms=1,
             scheduled_latency_ms=1, dispatch_lag_ms=0)
        for profile in profiles
    ]
    summary = summarize_measurement({
        "records": records, "duration_seconds": 1, "drain_seconds": 0,
    }, 3)

    assert summary["server_profile"]["batch_inference_ms"]["mean"] == 20
    assert summary["server_batch_profile"]["unique_batch_count"] == 2
    assert summary["server_batch_profile"]["batch_inference_ms"]["mean"] == 25
