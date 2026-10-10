CREATE TABLE IF NOT EXISTS msmarco_full_documents (
    document_id BIGINT PRIMARY KEY,
    external_id TEXT NOT NULL UNIQUE,
    content TEXT NOT NULL,
    embedding VECTOR(384)
);

COMMENT ON TABLE msmarco_full_documents IS
'Full MS MARCO passage corpus for large-scale retrieval evaluation.';

ANALYZE msmarco_full_documents;
