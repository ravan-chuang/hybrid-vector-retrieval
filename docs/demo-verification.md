# Clean Checkout Verification

The November 14, 2026 verification started from [`0d1f1e2`](https://github.com/ravan-chuang/hybrid-vector-retrieval/commit/0d1f1e25bd8db530c39683e4588d9f020461482f) with this handoff's lifecycle assertion applied. It used a fresh clone, a fresh Python environment, a Release build of the C++ extension, the checked-in one-passage startup fixture, and a disposable PostgreSQL instance with fresh application and benchmark schemas.

| Component | Verified version |
| --- | --- |
| Host | macOS 27.0.1 arm64 |
| Frontend | Node.js 24.14.1 · npm 11.11.0 · React 19.3.0 · Vite 8.3.3 |
| Backend | Python 3.13.13 · FastAPI 0.162.7 · Uvicorn 0.54.0 · psycopg 3.3.9 |
| Models | sentence-transformers 6.3.0 · transformers 5.23.0 · Torch 2.15.0 (CPU) |
| C++ | CMake 4.4.4 · AppleClang 21.0.0 · pybind11 3.1.0 |
| Database | PostgreSQL 17.11 · pgvector 0.8.6 |

The extension was built from the verification checkout. Its SHA-256 was `9301d8e0f765a7cd97ce80e2be841e47dd1857002b1503322262246130851d7f` for this macOS arm64 build. The [full-corpus experiments](evaluation/README.md) record the separately loaded v0.7.9 binary.

## Checks completed

- `npm ci`, lint, the document selection regression, and the Vite build passed. Lint reported one existing unused-parameter warning.
- All 105 Python tests passed with the PostgreSQL regressions enabled. The three C++ tests and the Python binding smoke test passed.
- The API reported a healthy database and an empty benchmark corpus.
- The live lifecycle created a document, replaced its content and embedding, verified lexical, vector, and hybrid results, deleted the document, and checked the new search history. The database was empty of application documents and chunks afterward.
- Both live rerank routes retrieved the expected application document and persisted their search history. The batched response also supplied a batch ID.

## Re-run the lifecycle

Set up a disposable database and start the API with the fixture using the [Quickstart](../README.md#quickstart). Run from the repository root in the backend Python environment.

```bash
DOCUMENT_SEARCH_BASE_URL=http://127.0.0.1:8000 \
  python backend/tests/document_search_lifecycle.py
```

The lifecycle writes search events and deletes the document that it creates. For the automated build, database regression, and lifecycle commands, see the [CI workflow](../.github/workflows/ci.yml).
