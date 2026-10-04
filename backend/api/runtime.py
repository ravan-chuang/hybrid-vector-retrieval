from psycopg_pool import ConnectionPool
from sentence_transformers import SentenceTransformer


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
