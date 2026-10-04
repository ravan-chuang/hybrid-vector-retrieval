ALTER TABLE benchmark_documents
ADD COLUMN IF NOT EXISTS search_vector tsvector;

UPDATE benchmark_documents
SET search_vector =
    to_tsvector('english', content);

CREATE INDEX IF NOT EXISTS idx_benchmark_search_vector
ON benchmark_documents
USING GIN (search_vector);

CREATE OR REPLACE FUNCTION benchmark_search_vector_update()
RETURNS trigger AS $$
BEGIN
    NEW.search_vector :=
        to_tsvector('english', NEW.content);

    RETURN NEW;
END
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_benchmark_search_vector
ON benchmark_documents;

CREATE TRIGGER trg_benchmark_search_vector
BEFORE INSERT OR UPDATE OF content
ON benchmark_documents
FOR EACH ROW
EXECUTE FUNCTION benchmark_search_vector_update();

ANALYZE benchmark_documents;
