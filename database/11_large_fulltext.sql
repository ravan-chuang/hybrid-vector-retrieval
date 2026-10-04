ALTER TABLE scalability_large_documents
ADD COLUMN IF NOT EXISTS search_vector tsvector
GENERATED ALWAYS AS (
    to_tsvector('english', coalesce(content, ''))
) STORED;

CREATE INDEX IF NOT EXISTS idx_scalability_large_search_vector_gin
ON scalability_large_documents
USING gin (search_vector);

ANALYZE scalability_large_documents;
