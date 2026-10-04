DROP TABLE IF EXISTS scalability_documents CASCADE;

CREATE TABLE scalability_documents (
    document_id BIGSERIAL PRIMARY KEY,
    external_id VARCHAR(255) NOT NULL UNIQUE,
    content TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL,
    search_vector tsvector
);

CREATE OR REPLACE FUNCTION scalability_search_vector_update()
RETURNS trigger AS $$
BEGIN
    NEW.search_vector :=
        to_tsvector('english', NEW.content);
    RETURN NEW;
END
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_scalability_search_vector
BEFORE INSERT OR UPDATE OF content
ON scalability_documents
FOR EACH ROW
EXECUTE FUNCTION scalability_search_vector_update();
