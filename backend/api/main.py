from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer

from backend.api import runtime
from backend.api.config import MODEL_NAME
from backend.api.database import create_connection_pool
from backend.api.services.cpp_retrieval import load_cpp_engine
from backend.api.services.dynamic_batch import (
    shutdown_dynamic_batch_reranker,
)
from backend.api.routers.documents import router as documents_router
from backend.api.routers.search import router as search_router
from backend.api.routers.cpp_search import router as cpp_search_router
from backend.api.routers.rerank import router as rerank_router
from backend.api.routers.system import router as system_router
from backend.api.routers.tags import router as tags_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME, device="cpu")
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

    cpp_engine = load_cpp_engine()
    runtime.set_cpp_engine(cpp_engine)

    print("C++ BM25 engine ready.")

    yield

    shutdown_dynamic_batch_reranker()

    runtime.set_cpp_engine(None)

    db_pool.close()
    runtime.set_db_pool(None)
    runtime.set_model(None)


app = FastAPI(
    title="Hybrid Vector Retrieval API",
    description=(
        "PostgreSQL + pgvector hybrid retrieval system "
        "supporting lexical, vector, and hybrid search."
    ),
    version="0.7.0",
    lifespan=lifespan,
)

app.include_router(search_router)
app.include_router(cpp_search_router)
app.include_router(rerank_router)
app.include_router(documents_router)
app.include_router(system_router)
app.include_router(tags_router)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
