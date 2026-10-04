CREATE INDEX idx_benchmark_embedding_ivfflat
ON benchmark_documents
USING ivfflat (embedding vector_cosine_ops)
WITH (
    lists = 100
);

ANALYZE benchmark_documents;
