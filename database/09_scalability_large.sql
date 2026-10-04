CREATE TABLE IF NOT EXISTS scalability_large_documents (
    document_id BIGSERIAL PRIMARY KEY,
    external_id VARCHAR(255) NOT NULL UNIQUE,
    content TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL
);

ANALYZE scalability_large_documents;
