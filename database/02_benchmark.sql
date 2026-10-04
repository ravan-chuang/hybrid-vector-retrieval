CREATE TABLE IF NOT EXISTS benchmark_documents (
    benchmark_id BIGSERIAL PRIMARY KEY,
    external_id VARCHAR(255) UNIQUE,
    content TEXT NOT NULL,
    embedding VECTOR(384),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_benchmark_external_id
ON benchmark_documents(external_id);
