from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer

from backend.api import runtime
from backend.api.config import MODEL_NAME
from backend.api.database import create_connection_pool
from backend.api.routers.documents import router as documents_router
from backend.api.routers.search import router as search_router
from backend.api.routers.system import router as system_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)
    model.encode(
        "warmup",
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    runtime.set_model(model)

    print("Embedding model loaded and warmed up.")

    db_pool = create_connection_pool()
    runtime.set_db_pool(db_pool)

    print("PostgreSQL connection pool ready.")

    yield

    db_pool.close()
    runtime.set_db_pool(None)
    runtime.set_model(None)


app = FastAPI(
    title="Hybrid Vector Retrieval API",
    description=(
        "PostgreSQL + pgvector hybrid retrieval system "
        "supporting lexical, vector, and hybrid search."
    ),
    version="0.6.0",
    lifespan=lifespan,
)

app.include_router(search_router)
app.include_router(documents_router)
app.include_router(system_router)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
