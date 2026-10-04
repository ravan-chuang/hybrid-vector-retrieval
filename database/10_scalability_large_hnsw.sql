SET maintenance_work_mem = '1GB';

SHOW maintenance_work_mem;

CREATE INDEX idx_scalability_large_embedding_hnsw
ON scalability_large_documents
USING hnsw (embedding vector_cosine_ops)
WITH (
    m = 16,
    ef_construction = 64
);

ANALYZE scalability_large_documents;
