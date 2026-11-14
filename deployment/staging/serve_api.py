"""Run the API on loopback and accept requests from the staging gateway."""

import argparse
import hmac
import os

import uvicorn
from starlette.responses import JSONResponse


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
        await app(scope, receive, send)

    uvicorn.run(protected_app, host="127.0.0.1", port=args.port, proxy_headers=False)


if __name__ == "__main__":
    main()
