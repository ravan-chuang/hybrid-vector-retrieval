"""Regression coverage for abandoning requests before inference."""

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event
import time
from unittest.mock import Mock

import pytest

from backend.api.services import cross_encoder, dynamic_batch, rerank_cancellation
from deployment.staging.serve_api import serve_rerank


def test_standard_request_exits_while_waiting_for_inference(monkeypatch):
    model = Mock()
    waiting = Event()
    signal = Future()

    def get_model():
        waiting.set()
        return model

    def run():
        token = rerank_cancellation.disconnected.set(signal)
        try:
            return cross_encoder.rerank_candidates_profiled("query", [{"content": "passage"}])
        finally:
            rerank_cancellation.disconnected.reset(token)

    monkeypatch.setattr(cross_encoder, "get_cross_encoder", get_model)
    with ThreadPoolExecutor(1) as executor:
        with cross_encoder._inference_lock:
            pending = executor.submit(run)
            assert waiting.wait(2)
            signal.set_result(None)
            with pytest.raises(rerank_cancellation.RerankDisconnected):
                pending.result(2)

    model.predict.assert_not_called()


def test_batch_excludes_disconnected_job_after_shared_lock_wait(monkeypatch):
    model = Mock()
    model.predict.return_value = [0.5]
    waiting = Event()
    signal = Future()

    def get_model():
        waiting.set()
        return model

    monkeypatch.setattr(dynamic_batch, "get_cross_encoder", get_model)
    monkeypatch.setattr(dynamic_batch, "resolve_device", lambda: "cpu")
    abandoned = dynamic_batch.BatchJob(
        "abandoned", [{"document_id": 1, "content": "first"}], 1, Future(),
        time.perf_counter(), disconnect=signal,
    )
    retained = dynamic_batch.BatchJob(
        "retained", [{"document_id": 2, "content": "second"}], 1, Future(),
        time.perf_counter(),
    )
    scheduler = dynamic_batch.DynamicBatchReranker()
    try:
        with ThreadPoolExecutor(1) as executor:
            with cross_encoder._inference_lock:
                pending = executor.submit(scheduler._process_batch, [abandoned, retained])
                assert waiting.wait(2)
                signal.set_result(None)
            pending.result(2)
        assert model.predict.call_args.args[0] == [("retained", "second")]
        result, profile = retained.future.result(2)
        assert result[0]["document_id"] == 2
        assert profile["batch_request_count"] == profile["batch_pair_count"] == 1
    finally:
        scheduler.close()


@pytest.mark.parametrize("disconnect_before_response", [True, False])
def test_staging_observes_disconnect_after_multiframe_body(disconnect_before_response):
    async def exercise():
        messages = asyncio.Queue()
        for body, more_body in ((b"first", True), (b"second", False)):
            messages.put_nowait({
                "type": "http.request", "body": body, "more_body": more_body,
            })

        async def send(message):
            messages.put_nowait({"type": "http.disconnect"})
            await asyncio.sleep(0)

        async def app(scope, receive, wrapped_send):
            assert (await receive())["body"] == b"first"
            assert (await receive())["body"] == b"second"
            signal = rerank_cancellation.disconnected.get()
            if disconnect_before_response:
                messages.put_nowait({"type": "http.disconnect"})
                observed = await asyncio.to_thread(signal.result, 2)
                assert observed is None
            else:
                await wrapped_send({"type": "http.response.start", "status": 200})
                assert (await receive())["type"] == "http.disconnect"
                assert not signal.done()

        await serve_rerank(app, {"path": "/search/rerank"}, messages.get, send, None)
        assert rerank_cancellation.disconnected.get() is None

    asyncio.run(exercise())
