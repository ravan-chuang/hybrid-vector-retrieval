CREATE INDEX idx_benchmark_embedding_hnsw
ON benchmark_documents
USING hnsw (embedding vector_cosine_ops)
WITH (
    m = 16,
    ef_construction = 64
);

ANALYZE benchmark_documents;
