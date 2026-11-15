"""Run the API on loopback and accept requests from the staging gateway."""

import argparse
import hmac
import os
import re

import uvicorn
from starlette.responses import JSONResponse

from backend.api.services import rerank_trace


RERANK_PATHS = {"/search/rerank", "/search/rerank/batched"}
REQUEST_ID = re.compile(rb"[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}")


async def trace_rerank(app, scope, receive, send, trace_id):
    token = rerank_trace.request_id.set(trace_id)
    rerank_trace.record("api_received", trace_id, path=scope["path"])
    status = None

    async def trace_send(message):
        nonlocal status
        if message["type"] == "http.response.start":
            status = message["status"]
        await send(message)

    try:
        await app(scope, receive, trace_send)
    finally:
        rerank_trace.record("api_finished", trace_id, status=status)
        rerank_trace.request_id.reset(token)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, type=int)
    args = parser.parse_args()
    expected = os.environ.get("STAGING_PROXY_TOKEN", "").encode()
    if not expected:
        parser.error("STAGING_PROXY_TOKEN is required")

    from backend.api.main import app

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
                if len(ids) == 1 and REQUEST_ID.fullmatch(ids[0]):
                    await trace_rerank(app, scope, receive, send, ids[0].decode("ascii"))
                    return
        await app(scope, receive, send)

    uvicorn.run(protected_app, host="127.0.0.1", port=args.port, proxy_headers=False)


if __name__ == "__main__":
    main()
