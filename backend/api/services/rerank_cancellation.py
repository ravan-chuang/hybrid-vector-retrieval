"""Pass the serving connection's disconnect signal to the reranking workers."""

from concurrent.futures import Future
from contextvars import ContextVar


disconnected: ContextVar[Future[None] | None] = ContextVar(
    "rerank_disconnected",
    default=None,
)


class RerankDisconnected(Exception):
    """The connection closed while reranking was in progress."""


def check_disconnected(signal: Future[None] | None) -> None:
    if signal is not None and signal.done():
        raise RerankDisconnected()
