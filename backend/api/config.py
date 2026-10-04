import os


DB_URL = os.getenv("DATABASE_URL")

if not DB_URL:
    raise RuntimeError(
        "DATABASE_URL is not set. Load the project .env before starting the application."
    )


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SEARCH_TABLE = "scalability_large_documents"

EMBEDDING_DIMENSION = 384

DB_POOL_MIN_SIZE = 2
DB_POOL_MAX_SIZE = 10
