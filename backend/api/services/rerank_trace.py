"""Log rerank stages for requests identified by the staging gateway."""

from contextvars import ContextVar
import json
import logging
import time


request_id: ContextVar[str | None] = ContextVar("rerank_request_id", default=None)
logger = logging.getLogger("uvicorn.error.rerank")


def record(event: str, trace_id: str | None, **fields) -> None:
    if trace_id is not None:
        logger.info("rerank_trace %s", json.dumps({
            "time": time.time(),
            "request_id": trace_id,
            "event": event,
            **fields,
        }))
