-- Full MS MARCO retrieval indexes
-- Corpus: 8,841,823 passages
-- Embedding: all-MiniLM-L6-v2, 384 dimensions, L2 normalized

-- Lexical retrieval
CREATE INDEX IF NOT EXISTS idx_msmarco_full_fts_gin
ON msmarco_full_documents
USING gin (to_tsvector('english', content));

-- Dense ANN retrieval
CREATE INDEX IF NOT EXISTS idx_msmarco_full_embedding_hnsw
ON msmarco_full_documents
USING hnsw (embedding vector_cosine_ops)
WITH (
    m = 16,
    ef_construction = 64
);

ANALYZE msmarco_full_documents;
