from threading import Lock


class ApplicationBM25:
    """Keep a C++ index over the committed chunks of the small application corpus."""

    def __init__(self, engine_factory):
        self._engine_factory = engine_factory
        self._snapshot = (None, [])
        self._lock = Lock()

    def _current_snapshot(self, cursor):
        # Serialize refreshes so an older read cannot replace a newer index.
        with self._lock:
            cursor.execute(
                """
                SELECT document_id, content
                FROM document_chunks
                ORDER BY document_id, chunk_id
                """
            )
            chunks = cursor.fetchall()
            if chunks != self._snapshot[1]:
                engine = self._engine_factory()
                for chunk_id, (_, content) in enumerate(chunks):
                    engine.add_document(chunk_id, content)
                self._snapshot = (engine, chunks)
            return self._snapshot

    def search(self, cursor, query: str, limit: int):
        engine, chunks = self._current_snapshot(cursor)
        if not chunks:
            return []

        # Rank chunks before choosing one result per document. Asking for just
        # limit chunks could fill the entire result set with one document.
        matches = engine.search(query, len(chunks))
        results = []
        seen = set()
        for match in matches:
            document_id, content = chunks[match.doc_id]
            if document_id in seen:
                continue
            seen.add(document_id)
            results.append((document_id, f"app-{document_id}", content, match.score))
            if len(results) == limit:
                break
        return results
