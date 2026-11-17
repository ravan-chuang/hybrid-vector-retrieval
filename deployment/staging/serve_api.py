"""Run the API on loopback and accept requests from the staging gateway."""

import argparse
import asyncio
from concurrent.futures import Future
from contextlib import suppress
import hmac
import os
import re
from pathlib import Path

import uvicorn
from starlette.responses import JSONResponse

from backend.api.services import rerank_cancellation, rerank_trace


RERANK_PATHS = {"/search/rerank", "/search/rerank/batched"}
REQUEST_ID = re.compile(rb"[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}")


async def serve_rerank(app, scope, receive, send, trace_id):
    signal = Future()
    cancellation_token = rerank_cancellation.disconnected.set(signal)
    trace_token = rerank_trace.request_id.set(trace_id)
    rerank_trace.record("api_received", trace_id, path=scope["path"])
    status = None
    watcher = None

    def mark_disconnected():
        if status is None and not signal.done():
            rerank_trace.record("api_disconnected", trace_id)
            signal.set_result(None)

    async def watch_disconnect():
        message = await receive()
        if message["type"] == "http.disconnect":
            mark_disconnected()
        return message

    async def receive_request():
        nonlocal watcher
        if watcher is not None:
            return await asyncio.shield(watcher)
        message = await receive()
        if message["type"] == "http.disconnect":
            mark_disconnected()
        elif not message.get("more_body", False):
            # FastAPI has read the body. Observe connection loss while its
            # synchronous endpoint is running in the thread pool.
            watcher = asyncio.create_task(watch_disconnect())
        return message

    async def trace_send(message):
        nonlocal status
        if message["type"] == "http.response.start":
            status = message["status"]
        await send(message)

    try:
        await app(scope, receive_request, trace_send)
    finally:
        try:
            if watcher is not None:
                watcher.cancel()
                with suppress(asyncio.CancelledError):
                    await watcher
        finally:
            rerank_trace.record("api_finished", trace_id, status=status)
            rerank_trace.request_id.reset(trace_token)
            rerank_cancellation.disconnected.reset(cancellation_token)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, type=int)
    args = parser.parse_args()
    expected = os.environ.get("STAGING_PROXY_TOKEN", "").encode()
    if not expected:
        parser.error("STAGING_PROXY_TOKEN is required")

    from backend.api.main import app

    telemetry_file = os.environ.get("STAGING_RERANK_TELEMETRY")
    if telemetry_file:
        from deployment.staging.telemetry import lifespan_with_telemetry

        app.router.lifespan_context = lifespan_with_telemetry(
            app.router.lifespan_context, Path(telemetry_file)
        )

    async def protected_app(scope, receive, send):
        if scope["type"] == "http":
            tokens = [
                value for key, value in scope["headers"] if key == b"x-staging-proxy-token"
            ]
            if len(tokens) != 1 or not hmac.compare_digest(expected, tokens[0]):
                response = JSONResponse({"detail": "Access denied"}, status_code=403)
                await response(scope, receive, send)
                return

            if scope["path"] in RERANK_PATHS and scope["method"] == "POST":
                ids = [
                    value for key, value in scope["headers"] if key == b"x-request-id"
                ]
                trace_id = (
                    ids[0].decode("ascii")
                    if len(ids) == 1 and REQUEST_ID.fullmatch(ids[0])
                    else None
                )
                await serve_rerank(app, scope, receive, send, trace_id)
                return
        await app(scope, receive, send)

    uvicorn.run(protected_app, host="127.0.0.1", port=args.port, proxy_headers=False)


if __name__ == "__main__":
    main()
