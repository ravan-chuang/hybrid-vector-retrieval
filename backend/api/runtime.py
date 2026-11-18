from psycopg_pool import ConnectionPool
from sentence_transformers import SentenceTransformer

from backend.api.services.application_bm25 import ApplicationBM25


model: SentenceTransformer | None = None
db_pool: ConnectionPool | None = None


def set_model(value: SentenceTransformer | None) -> None:
    global model
    model = value


def get_model() -> SentenceTransformer | None:
    return model


def set_db_pool(value: ConnectionPool | None) -> None:
    global db_pool
    db_pool = value


def get_db_pool() -> ConnectionPool | None:
    return db_pool



# C++ BM25 retrieval engine
cpp_engine = None
application_bm25: ApplicationBM25 | None = None


def set_cpp_engine(value) -> None:
    global cpp_engine
    cpp_engine = value


def get_cpp_engine():
    return cpp_engine


def set_application_bm25(value: ApplicationBM25 | None) -> None:
    global application_bm25
    application_bm25 = value


def get_application_bm25() -> ApplicationBM25 | None:
    return application_bm25
