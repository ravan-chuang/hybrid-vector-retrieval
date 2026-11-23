from types import SimpleNamespace

import pytest

from deployment.staging import benchmark_rerank


@pytest.mark.parametrize(
    ("group_size", "interval", "offsets"),
    [(None, None, [0, 0, 0, 0, 0]), (2, 1.5, [0, 0, 1.5, 1.5, 3])],
)
def test_burst_release_schedule(group_size, interval, offsets):
    args = SimpleNamespace(
        burst=5, burst_group_size=group_size, burst_interval=interval,
        gateway="https://localhost", route="batched", timeout=15,
    )

    def send(origin, route, payload, request_id, scheduled, start, timeout):
        assert payload == request_id
        return {"success": True, "status_code": 200, "offset": scheduled - start}

    response = benchmark_rerank.burst(args, list(range(7)), send)
    assert [r["offset"] for r in response["records"]] == pytest.approx(offsets)
    assert response["summary"]["status_counts"] == {"200": 5}
