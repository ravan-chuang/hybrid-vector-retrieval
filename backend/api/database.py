from psycopg_pool import ConnectionPool

from backend.api.config import (
    DB_POOL_MAX_SIZE,
    DB_POOL_MIN_SIZE,
    DB_URL,
)


def create_connection_pool() -> ConnectionPool:
    pool = ConnectionPool(
        conninfo=DB_URL,
        min_size=DB_POOL_MIN_SIZE,
        max_size=DB_POOL_MAX_SIZE,
        open=True,
    )
    pool.wait()
    return pool
