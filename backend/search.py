import os
import sys
import time
import psycopg
from sentence_transformers import SentenceTransformer

DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://retrieval:retrieval@127.0.0.1:5433/retrieval_db",
)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


def to_vector_string(vector):
    return "[" + ",".join(str(float(x)) for x in vector) + "]"


def semantic_search(query, k=5):
    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    query_vector = to_vector_string(query_embedding)

    start = time.perf_counter()

    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    dc.chunk_id,
                    d.title,
                    dc.content,
                    dc.embedding <=> %s::vector AS cosine_distance
                FROM document_chunks dc
                JOIN documents d
                    ON d.document_id = dc.document_id
                WHERE dc.embedding IS NOT NULL
                ORDER BY dc.embedding <=> %s::vector
                LIMIT %s
                """,
                (query_vector, query_vector, k)
            )

            results = cur.fetchall()

    latency_ms = (time.perf_counter() - start) * 1000

    print("\n========================================")
    print("Semantic Vector Search")
    print("========================================")
    print(f"Query: {query}")
    print(f"Top-K: {k}")
    print(f"DB latency: {latency_ms:.3f} ms")
    print("========================================\n")

    for rank, (chunk_id, title, content, distance) in enumerate(results, 1):
        similarity = 1.0 - float(distance)

        print(f"[{rank}] {title}")
        print(f"Chunk ID:   {chunk_id}")
        print(f"Similarity: {similarity:.4f}")
        print(f"Text:       {content}")
        print()


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]).strip()

    if not query:
        query = "How do search engines find relevant documents?"

    semantic_search(query)
