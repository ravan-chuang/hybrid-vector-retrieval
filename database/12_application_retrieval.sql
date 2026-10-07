-- Application retrieval indexes
-- Enables document_chunks to serve as a searchable application corpus.

ALTER TABLE document_chunks
ADD COLUMN IF NOT EXISTS search_vector TSVECTOR
GENERATED ALWAYS AS (
    to_tsvector('english', content)
) STORED;

CREATE INDEX IF NOT EXISTS idx_document_chunks_search_vector_gin
ON document_chunks
USING GIN (search_vector);

CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding_hnsw
ON document_chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

ANALYZE document_chunks;
