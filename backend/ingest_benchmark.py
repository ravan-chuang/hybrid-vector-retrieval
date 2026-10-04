import os
import time
import psycopg
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://retrieval:retrieval@127.0.0.1:5433/retrieval_db",
)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CORPUS_SIZE = 10_000
BATCH_SIZE = 128


def to_vector_string(vector):
    return "[" + ",".join(str(float(x)) for x in vector) + "]"


print("=" * 60)
print("Benchmark Corpus Ingestion")
print("=" * 60)

# 1. Load corpus
print("\n[1/5] Loading AG News corpus...")

dataset = load_dataset(
    "fancyzhx/ag_news",
    split=f"train[:{CORPUS_SIZE}]"
)

texts = [row["text"].strip() for row in dataset]

print(f"Loaded: {len(texts):,} documents")


# 2. Load embedding model
print("\n[2/5] Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print(f"Model: {MODEL_NAME}")


# 3. Generate embeddings
print("\n[3/5] Generating 384-d embeddings...")

embedding_start = time.perf_counter()

embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    normalize_embeddings=True,
    show_progress_bar=True
)

embedding_seconds = time.perf_counter() - embedding_start

print(f"Embedding shape: {embeddings.shape}")
print(f"Embedding time: {embedding_seconds:.2f} s")
print(
    f"Embedding throughput: "
    f"{len(texts) / embedding_seconds:.2f} docs/s"
)


# 4. Insert into PostgreSQL
print("\n[4/5] Inserting into PostgreSQL...")

insert_start = time.perf_counter()

with psycopg.connect(DB_URL) as conn:
    with conn.cursor() as cur:

        cur.execute(
            "TRUNCATE TABLE benchmark_documents RESTART IDENTITY"
        )

        sql = """
            INSERT INTO benchmark_documents
                (external_id, content, embedding)
            VALUES
                (%s, %s, %s::vector)
        """

        rows = []

        for i, (text, embedding) in enumerate(
            zip(texts, embeddings)
        ):
            rows.append(
                (
                    f"ag_news_{i}",
                    text,
                    to_vector_string(embedding)
                )
            )

        cur.executemany(sql, rows)

    conn.commit()

insert_seconds = time.perf_counter() - insert_start

print(f"Inserted: {len(texts):,} rows")
print(f"Insert time: {insert_seconds:.2f} s")
print(
    f"Insert throughput: "
    f"{len(texts) / insert_seconds:.2f} rows/s"
)


# 5. Verify
print("\n[5/5] Verifying database...")

with psycopg.connect(DB_URL) as conn:
    with conn.cursor() as cur:

        cur.execute("""
            SELECT
                COUNT(*),
                COUNT(embedding),
                MIN(vector_dims(embedding)),
                MAX(vector_dims(embedding))
            FROM benchmark_documents
        """)

        total, vectors, min_dims, max_dims = cur.fetchone()

        cur.execute("""
            SELECT pg_size_pretty(
                pg_total_relation_size('benchmark_documents')
            )
        """)

        table_size = cur.fetchone()[0]


print("\n" + "=" * 60)
print("Ingestion Complete")
print("=" * 60)
print(f"Documents:       {total:,}")
print(f"Embeddings:      {vectors:,}")
print(f"Vector dims:     {min_dims} - {max_dims}")
print(f"Table size:      {table_size}")
print(f"Embedding time:  {embedding_seconds:.2f} s")
print(f"Insert time:     {insert_seconds:.2f} s")
print("=" * 60)
