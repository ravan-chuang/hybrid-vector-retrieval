BEGIN;

ALTER TABLE search_history
    DROP CONSTRAINT IF EXISTS search_history_search_type_check,
    ADD CONSTRAINT search_history_search_type_check
        CHECK (search_type IN ('keyword', 'vector', 'hybrid', 'rerank'));

COMMIT;
