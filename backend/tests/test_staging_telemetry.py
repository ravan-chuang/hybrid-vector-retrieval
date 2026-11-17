from contextlib import asynccontextmanager
from io import StringIO
import json

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from deployment.staging.telemetry import lifespan_with_telemetry


def test_staging_telemetry_open_failure_prevents_application_startup(tmp_path):
    @asynccontextmanager
    async def lifespan(app):
        pytest.fail("The application should not start with an invalid telemetry path")
        yield

    path = tmp_path / "missing" / "telemetry.jsonl"
    app = FastAPI(lifespan=lifespan_with_telemetry(lifespan, path))

    with pytest.raises(RuntimeError, match="Cannot open STAGING_RERANK_TELEMETRY output"):
        with TestClient(app):
            pytest.fail("Startup succeeded with an invalid telemetry path")


def test_staging_telemetry_write_failure_prevents_readiness():
    class FullLog(StringIO):
        def write(self, text):
            raise OSError("No space left on device")

    class FullLogPath:
        def open(self, *args, **kwargs):
            return FullLog()

    app = FastAPI()
    app.router.lifespan_context = lifespan_with_telemetry(app.router.lifespan_context, FullLogPath())

    with pytest.raises(RuntimeError, match="Cannot write STAGING_RERANK_TELEMETRY output"):
        with TestClient(app):
            pytest.fail("Startup succeeded without writing an initial sample")


def test_staging_telemetry_appends_initial_sample_before_readiness(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    path.write_text('{"previous": true}\n')
    app = FastAPI()
    app.router.lifespan_context = lifespan_with_telemetry(app.router.lifespan_context, path)

    with TestClient(app):
        previous, first_sample, *_ = map(json.loads, path.read_text().splitlines())
        assert previous == {"previous": True}
        assert first_sample["pid"] > 0
        assert first_sample["monotonic"] > 0
        assert first_sample["queue_depth"] >= 0
