SET maintenance_work_mem = '1GB';

SHOW maintenance_work_mem;

CREATE INDEX idx_scalability_embedding_hnsw
ON scalability_documents
USING hnsw (embedding vector_cosine_ops)
WITH (
    m = 16,
    ef_construction = 64
);

ANALYZE scalability_documents;
