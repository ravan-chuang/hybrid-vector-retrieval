# Hybrid Vector Retrieval System

A PostgreSQL-based hybrid information retrieval system combining
relational database design, full-text search, dense vector retrieval,
approximate nearest-neighbor (ANN) indexing, and Reciprocal Rank Fusion
(RRF).

> **Status:** Core database, retrieval, 500K scalability evaluation,
> REST API, CRUD, search history, and many-to-many tag management are
> complete. Web UI/dashboard, ER diagram, and final report/demo
> materials remain in progress.

## Overview

This project integrates traditional relational database functionality
with modern semantic retrieval using PostgreSQL and pgvector.

Current capabilities include:

-   Relational document and metadata storage
-   384-dimensional dense embeddings
-   Exact cosine vector search
-   PostgreSQL full-text search with GIN
-   HNSW approximate nearest-neighbor retrieval
-   IVFFlat approximate nearest-neighbor retrieval
-   Hybrid lexical + semantic retrieval using Reciprocal Rank Fusion
-   ANN recall, latency, throughput, index-size, and build-time
    evaluation
-   PostgreSQL/Docker resource tuning
-   10K, 100K, and 500K scalability experiments
-   FastAPI REST API with PostgreSQL connection pooling
-   Document CRUD and persisted search history
-   Many-to-many document/tag API with FK cascade validation

The project is designed as both a **database systems project** and a
**retrieval engineering project**, with emphasis on schema design,
indexing, query execution, performance measurement, and scalability.

## Architecture

``` text
                         ┌─────────────────┐
                         │   User Query    │
                         └────────┬────────┘
                                  │
                  ┌───────────────┴───────────────┐
                  │                               │
                  ▼                               ▼
        ┌──────────────────┐           ┌──────────────────┐
        │ PostgreSQL FTS   │           │ MiniLM Encoder   │
        │ lexical search   │           │ 384-d embedding  │
        └────────┬─────────┘           └────────┬─────────┘
                 │                              │
                 ▼                              ▼
        ┌──────────────────┐           ┌──────────────────┐
        │    GIN Index     │           │     pgvector     │
        └────────┬─────────┘           └────────┬─────────┘
                 │                     ┌─────────┴─────────┐
                 │                     │                   │
                 │                     ▼                   ▼
                 │               ┌──────────┐        ┌──────────┐
                 │               │   HNSW   │        │ IVFFlat │
                 │               └────┬─────┘        └────┬─────┘
                 │                    │                   │
                 └────────────────────┴─────────┬─────────┘
                                                ▼
                                   ┌────────────────────────┐
                                   │ Reciprocal Rank Fusion │
                                   │        (RRF)           │
                                   └───────────┬────────────┘
                                               ▼
                                       ┌──────────────┐
                                       │ Top-K Results│
                                       └──────────────┘
```

## Technology Stack

**Database** - PostgreSQL 17 - pgvector - PostgreSQL Full-Text Search -
GIN, HNSW, and IVFFlat indexes

**Backend / Evaluation** - Python 3 - psycopg - NumPy - Sentence
Transformers - Hugging Face Datasets

**Embedding Model** - `sentence-transformers/all-MiniLM-L6-v2` - 384
dimensions - normalized embeddings

**Infrastructure** - Docker - Docker Compose

## Database Design

The primary relational schema contains:

  Table               Purpose
  ------------------- ------------------------------------------------
  `users`             User information
  `documents`         Document-level metadata
  `document_chunks`   Text chunks and vector embeddings
  `tags`              Unique tags
  `document_tags`     Many-to-many document/tag relationship
  `search_history`    Query type, result count, latency, and history

The schema demonstrates primary/foreign keys, referential integrity,
unique and check constraints, many-to-many relationships, JOIN-based
retrieval, and relational metadata + vector data integration.

Separate benchmark tables isolate performance and scalability
experiments from the primary application schema.

## Retrieval Methods

### Exact Vector Search

Queries and documents are represented as normalized 384-dimensional
embeddings. Exact cosine-distance retrieval uses pgvector:

``` sql
SELECT document_id,
       content,
       embedding <=> %s::vector AS distance
FROM benchmark_documents
ORDER BY embedding <=> %s::vector
LIMIT 10;
```

For normalized vectors:

``` text
cosine_distance = 1 - cosine_similarity
```

Exact retrieval is also used as ground truth for ANN Recall@10.

### HNSW

HNSW provides low-latency approximate nearest-neighbor retrieval.

``` sql
CREATE INDEX idx_benchmark_embedding_hnsw
ON benchmark_documents
USING hnsw (embedding vector_cosine_ops)
WITH (
    m = 16,
    ef_construction = 64
);
```

Experiments vary `hnsw.ef_search` to evaluate recall/latency trade-offs.

### IVFFlat

IVFFlat partitions vector space into lists and searches selected
partitions.

``` sql
CREATE INDEX idx_benchmark_embedding_ivfflat
ON benchmark_documents
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

Experiments vary `ivfflat.probes`.

### Full-Text Search

Lexical retrieval uses PostgreSQL `tsvector` and `tsquery`, accelerated
by a GIN index.

``` sql
CREATE INDEX idx_benchmark_search_vector
ON benchmark_documents
USING GIN (search_vector);
```

A trigger keeps `search_vector` synchronized with document content.

### Hybrid Retrieval

Hybrid retrieval combines lexical and semantic candidates using
Reciprocal Rank Fusion:

``` text
RRF(d) = Σ 1 / (k + rank(d))
```

The current implementation uses `k = 60`. RRF avoids requiring lexical
and vector scores to share the same numerical scale.

## Benchmark Methodology

Database retrieval latency is measured separately from query-embedding
generation. Benchmarks use warm-ups, persistent database connections,
repeated measurements, and report P50/P95/P99 latency and approximate
QPS.

ANN Recall@10 is measured against exact vector Top-K:

``` text
Recall@10 = |Top10_ANN ∩ Top10_Exact| / 10
```

This measures agreement with exact vector retrieval, **not**
human-judged end-to-end relevance.

## Experimental Results

### 10K Documents --- Exact vs HNSW

  Method            Recall@10   Mean (ms)   P50 (ms)   P95 (ms)       QPS
  --------------- ----------- ----------- ---------- ---------- ---------
  Exact                 1.000       3.335      3.219      4.192    299.86
  HNSW `ef=10`          0.820       0.565      0.545      0.756   1770.54
  HNSW `ef=20`          0.910       0.656      0.621      0.822   1524.45
  HNSW `ef=40`          0.980       0.808      0.785      1.049   1237.50
  HNSW `ef=80`          1.000       1.073      1.078      1.366    932.07
  HNSW `ef=160`         1.000       1.516      1.416      2.048    659.81

### 10K Documents --- Exact vs IVFFlat

  Method                  Recall@10   Mean (ms)   P50 (ms)   P95 (ms)       QPS
  --------------------- ----------- ----------- ---------- ---------- ---------
  Exact                       1.000       3.233      3.106      4.154    309.33
  IVFFlat `probes=1`          0.410       0.419      0.406      0.529   2383.99
  IVFFlat `probes=2`          0.560       0.443      0.416      0.547   2259.06
  IVFFlat `probes=5`          0.740       0.452      0.445      0.581   2214.38
  IVFFlat `probes=10`         0.850       0.633      0.627      0.758   1580.53
  IVFFlat `probes=20`         0.950       0.937      0.903      1.313   1067.26
  IVFFlat `probes=50`         1.000       1.834      1.841      2.218    545.31

On this workload, HNSW provided the stronger high-recall latency
trade-off, while IVFFlat offered different build/storage
characteristics.

## 100K Scalability Experiment

### Ingestion

  Metric                            Result
  ---------------------- -----------------
  Documents                        100,000
  Embedding dimension                  384
  Embedding time                   47.22 s
  Embedding throughput     2,117.62 docs/s
  Insert time                      32.04 s
  Insert throughput        3,121.44 rows/s
  Heap size                          87 MB

### Exact Baseline

An independent exact-search run produced:

  Metric             Result
  ------------- -----------
  Mean            77.657 ms
  P50             77.357 ms
  P95             83.813 ms
  P99             85.045 ms
  Approx. QPS         12.88

`EXPLAIN ANALYZE` verified sequential scanning for the exact baseline.

For direct Exact-vs-HNSW speed comparisons below, the exact result from
the **same benchmark run** is used to reduce differences caused by query
sets and system state.

### Exact vs HNSW at 100K

  -----------------------------------------------------------------------------
  Method       Recall@10  Mean (ms)   P50 (ms)   P95 (ms)   P99 (ms)        QPS
  ---------- ----------- ---------- ---------- ---------- ---------- ----------
  Exact            1.000     66.280     65.117     75.072     77.879      15.09

  HNSW             0.942      0.948      0.890      1.429      1.628    1054.70
  `ef=10`

  HNSW             0.965      1.090      1.062      1.742      1.898     917.83
  `ef=20`

  HNSW             0.992      1.387      1.405      1.978      2.449     720.74
  `ef=40`

  HNSW             0.994      2.039      1.946      3.246      3.489     490.52
  `ef=80`

  HNSW             0.997      3.047      2.973      4.655      5.337     328.24
  `ef=160`
  -----------------------------------------------------------------------------

At `ef_search=40`:

-   Recall@10: **0.992**
-   P50 latency: **1.405 ms**
-   Same-run exact P50: **65.117 ms**
-   P50 speedup: approximately **46.3×**

Increasing `ef_search` from 80 to 160 improved Recall@10 from 0.994 to
0.997 while P50 latency increased from 1.946 ms to 2.973 ms, showing
diminishing returns at higher search effort.

## 500K Scalability Experiment

The final large-scale experiment uses **500,000 passages** streamed from
the Hugging Face `sentence-transformers/msmarco-corpus` dataset.

### 500K Ingestion

  Metric                                  Result
  --------------------------- ------------------
  Documents                              500,000
  Embedding dimension                        384
  Embedding time                        849.79 s
  Embedding throughput             588.38 docs/s
  Insert time                            42.36 s
  Insert throughput             11,803.81 rows/s
  Total ingestion time                  971.75 s
  Overall throughput               514.54 docs/s
  Relation size before HNSW             1,019 MB
  Heap size                               888 MB
  Indexes before HNSW                      37 MB

Embedding generation dominates ingestion time at this scale; PostgreSQL
insertion is comparatively inexpensive.

### Exact vs HNSW at 500K

The final validated benchmark uses 100 queries. Exact and HNSW retrieval
use separate database connections, prepared statements are disabled for
the benchmark, and `EXPLAIN` checks verify the intended physical plan.

  -----------------------------------------------------------------------------
  Method       Recall@10  Mean (ms)   P50 (ms)   P95 (ms)   P99 (ms)        QPS
  ---------- ----------- ---------- ---------- ---------- ---------- ----------
  Exact            1.000    103.348    102.918    108.737    111.612       9.68

  HNSW             0.919      1.544      1.534      2.354      2.893     647.73
  `ef=10`

  HNSW             0.949      1.639      1.579      2.539      2.902     610.28
  `ef=20`

  HNSW             0.981      2.157      2.215      3.248      4.339     463.55
  `ef=40`

  HNSW             0.984      3.427      3.526      5.404      6.310     291.83
  `ef=80`

  HNSW             0.995      5.853      5.695      9.134     10.057     170.84
  `ef=160`
  -----------------------------------------------------------------------------

At `ef_search=40`:

-   Recall@10: **0.981**
-   P50 latency: **2.215 ms**
-   Exact P50: **102.918 ms**
-   P50 speedup: approximately **46.47×**
-   Approximate QPS: **463.55**

Increasing `ef_search` from 40 to 80 improves Recall@10 by only 0.003
while P50 latency rises from 2.215 ms to 3.526 ms, illustrating
diminishing returns.

### 500K Index and Storage Observations

  -----------------------------------------------------------------------
  Metric                                                           Result
  ------------------------------ ----------------------------------------
  HNSW build time with                                            83.22 s
  `maintenance_work_mem=1GB`

  HNSW index size                                                  976 MB

  Tuples fitting in HNSW build                          482,217 / 500,000
  memory before overflow

  Fraction fitting before                                          96.44%
  overflow

  Current serving table size                                       451 MB

  Current indexes size                                           1,071 MB

  Current total relation                                         2,256 MB
  footprint

  GIN full-text index                                               69 MB
  -----------------------------------------------------------------------

A representative 500K full-text query for `"Manhattan Project"` used the
GIN-backed bitmap plan and executed in approximately **0.428 ms**.

The stored-`tsvector` migration also exposed an operational database
cost: the table rewrite caused HNSW to be rebuilt. Under lower/default
maintenance memory, this made the migration substantially more
expensive.

## Scalability Summary

The experiments use different corpora at different stages, so
cross-scale comparisons should be treated as engineering observations
rather than a perfectly controlled corpus-scaling study.

  Scale      Exact P50   HNSW `ef=40` P50   HNSW Recall@10
  ------- ------------ ------------------ ----------------
  10K         3.219 ms           0.785 ms            0.980
  100K       65.117 ms           1.405 ms            0.992
  500K      102.918 ms           2.215 ms            0.981

From 10K to 500K, corpus size increases by 50× while HNSW `ef=40` P50
rises from 0.785 ms to 2.215 ms (about 2.82×). Because the stages do not
use identical corpora, this is not presented as a strict complexity
measurement.

## Scalability Observation

Corpus size increased 10× from 10K to 100K.

Using comparable benchmark results:

-   Exact P50: approximately 3.219 ms → 65.117 ms
-   HNSW (`ef=40`) P50: approximately 0.785 ms → 1.405 ms

This illustrates the different scaling behavior of brute-force exact
search and graph-based ANN retrieval on the tested workload.

## HNSW Index Build Tuning

The initial 100K HNSW build used PostgreSQL's default:

``` text
maintenance_work_mem = 64MB
```

PostgreSQL reported that the HNSW graph stopped fitting in
`maintenance_work_mem` after approximately 28K tuples.

  Configuration                     Build Time   HNSW Size
  ------------------------------- ------------ -----------
  `maintenance_work_mem = 64MB`       37.848 s      195 MB
  `maintenance_work_mem = 1GB`        12.038 s      195 MB

The Docker container's shared memory was increased to:

``` yaml
shm_size: '2gb'
```

and the tuned build used:

``` sql
SET maintenance_work_mem = '1GB';
```

This reduced HNSW build time by approximately **68.2%**, corresponding
to a **3.14× speedup**, while final index size remained unchanged.

The experiment highlights the distinction between Docker memory, POSIX
shared memory (`/dev/shm`), PostgreSQL `maintenance_work_mem`,
`shared_buffers`, and `work_mem`.

## REST API

The FastAPI backend exposes retrieval, system, CRUD, history, and
many-to-many tag operations. The application uses a PostgreSQL
connection pool and warms the MiniLM embedding model during startup.

Run the API:

``` bash
set -a
source .env
set +a

uvicorn backend.api.main:app --reload --host 127.0.0.1 --port 8000
```

Interactive OpenAPI documentation is available at
`http://127.0.0.1:8000/docs`.

  ------------------------------------------------------------------------------------------
  Method                  Endpoint                                   Purpose
  ----------------------- ------------------------------------------ -----------------------
  `GET`                   `/`                                        API information

  `GET`                   `/health`                                  Health/database
                                                                     connectivity

  `GET`                   `/stats`                                   Corpus, model,
                                                                     database, and index
                                                                     statistics

  `POST`                  `/search/vector`                           HNSW semantic retrieval

  `POST`                  `/search/lexical`                          PostgreSQL FTS/GIN
                                                                     retrieval

  `POST`                  `/search/hybrid`                           Lexical + vector
                                                                     retrieval with RRF

  `GET`                   `/search/history`                          Persisted recent
                                                                     searches

  `GET`                   `/documents`                               List application
                                                                     documents

  `POST`                  `/documents`                               Create a document and
                                                                     embedding

  `GET`                   `/documents/{document_id}`                 Retrieve one document

  `PUT`                   `/documents/{document_id}`                 Update a document

  `DELETE`                `/documents/{document_id}`                 Delete a document with
                                                                     FK cascades

  `GET`                   `/documents/{document_id}/tags`            List document tags

  `POST`                  `/documents/{document_id}/tags`            Create/reuse and attach
                                                                     a tag

  `DELETE`                `/documents/{document_id}/tags/{tag_id}`   Remove a document/tag
                                                                     association
  ------------------------------------------------------------------------------------------

### API Examples

Vector search:

``` bash
curl -X POST http://127.0.0.1:8000/search/vector \
  -H "Content-Type: application/json" \
  -d '{"query":"Manhattan Project","top_k":5,"ef_search":40}'
```

Hybrid search:

``` bash
curl -X POST http://127.0.0.1:8000/search/hybrid \
  -H "Content-Type: application/json" \
  -d '{"query":"atomic bomb World War II","top_k":5,"candidate_k":50,"ef_search":40,"rrf_k":60}'
```

Attach a tag:

``` bash
curl -X POST http://127.0.0.1:8000/documents/1/tags \
  -H "Content-Type: application/json" \
  -d '{"name":"database"}'
```

The tag endpoint reuses unique tags and creates the
`(document_id, tag_id)` association safely. Tests verified that the same
tag can be shared by multiple documents, removing an association does
not delete the tag, and deleting a document cascades to its
`document_chunks` and `document_tags`.

### Backend Architecture

``` text
backend/api/
├── main.py
├── config.py
├── database.py
├── runtime.py
├── routers/
│   ├── documents.py
│   ├── search.py
│   ├── system.py
│   └── tags.py
├── schemas/
│   ├── documents.py
│   ├── search.py
│   ├── system.py
│   └── tags.py
└── services/
    └── embedding.py
```

The API was refactored from a monolithic implementation into modular
routers, schemas, services, configuration, runtime state, and
database-pool management. Search-specific PostgreSQL settings use
transaction-local state where appropriate so settings do not leak across
pooled connections.

Search-history logging is performed after the measured retrieval timing,
so the history insert itself is not included in reported search latency.

## Project Structure

``` text
.
├── backend/
│   ├── api/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── database.py
│   │   ├── runtime.py
│   │   ├── routers/
│   │   ├── schemas/
│   │   └── services/
│   ├── benchmark_exact.py
│   ├── benchmark_hnsw.py
│   ├── benchmark_ivfflat.py
│   ├── benchmark_scalability_exact.py
│   ├── benchmark_scalability_hnsw.py
│   ├── benchmark_scalability_large_exact.py
│   ├── benchmark_scalability_large_hnsw.py
│   ├── hybrid_search.py
│   ├── ingest_benchmark.py
│   ├── ingest_scalability.py
│   ├── ingest_scalability_large.py
│   ├── search.py
│   ├── seed_documents.py
│   └── test_embedding.py
├── database/
│   ├── 01_schema.sql
│   ├── 02_benchmark.sql
│   ├── 03_hnsw.sql
│   ├── 04_ivfflat.sql
│   ├── 05_fulltext.sql
│   ├── 06_scalability.sql
│   ├── 07_scalability_hnsw.sql
│   ├── 08_scalability_hnsw_tuned.sql
│   ├── 09_scalability_large.sql
│   ├── 10_scalability_large_hnsw.sql
│   └── 11_large_fulltext.sql
├── frontend/
├── docs/
├── .env.example
├── .gitignore
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Setup

### 1. Clone

``` bash
git clone https://github.com/ravan-chuang/hybrid-vector-retrieval.git
cd hybrid-vector-retrieval
```

### 2. Configure Environment Variables

``` bash
cp .env.example .env
```

Edit `.env` and replace the example password with a local development
password.

``` dotenv
POSTGRES_DB=retrieval_db
POSTGRES_USER=retrieval
POSTGRES_PASSWORD=change_me
POSTGRES_PORT=5433
DATABASE_URL=postgresql://retrieval:change_me@127.0.0.1:5433/retrieval_db
```

Never commit the real `.env`.

### 3. Start PostgreSQL + pgvector

``` bash
docker compose up -d
```

Verify:

``` bash
docker exec hybrid-retrieval-db pg_isready -U retrieval -d retrieval_db
```

### 4. Create Python Environment

``` bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 5. Load Environment Variables

``` bash
set -a
source .env
set +a
```

### 6. Initialize Schema

``` bash
docker exec -i hybrid-retrieval-db psql -U retrieval -d retrieval_db < database/01_schema.sql
```

### 7. Run the API

``` bash
uvicorn backend.api.main:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the interactive API documentation.

For the small normalized-schema demo, `backend/seed_documents.py` can be
used to seed application documents. The 500K corpus itself is
intentionally not committed to Git.

## Main Scripts

  ----------------------------------------------------------------------------
  Script                                   Purpose
  ---------------------------------------- -----------------------------------
  `benchmark_exact.py`                     10K exact vector-search baseline

  `benchmark_hnsw.py`                      10K HNSW recall/latency evaluation

  `benchmark_ivfflat.py`                   10K IVFFlat recall/latency
                                           evaluation

  `benchmark_scalability_exact.py`         100K exact baseline

  `benchmark_scalability_hnsw.py`          100K HNSW evaluation

  `benchmark_scalability_large_exact.py`   500K validated exact baseline

  `benchmark_scalability_large_hnsw.py`    500K validated HNSW evaluation

  `hybrid_search.py`                       Full-text + dense retrieval + RRF

  `ingest_benchmark.py`                    Small benchmark ingestion

  `ingest_scalability.py`                  100K scalability ingestion

  `ingest_scalability_large.py`            Streaming 500K MS MARCO ingestion

  `seed_documents.py`                      Seed normalized application
                                           documents
  ----------------------------------------------------------------------------

## SQL Files

  File                              Purpose
  --------------------------------- --------------------------------------------
  `01_schema.sql`                   Primary normalized relational schema
  `02_benchmark.sql`                Benchmark table
  `03_hnsw.sql`                     HNSW index
  `04_ivfflat.sql`                  IVFFlat index
  `05_fulltext.sql`                 Full-text search and GIN index
  `06_scalability.sql`              100K scalability schema
  `07_scalability_hnsw.sql`         100K HNSW index
  `08_scalability_hnsw_tuned.sql`   Tuned HNSW build
  `09_scalability_large.sql`        500K scalability schema
  `10_scalability_large_hnsw.sql`   500K HNSW index
  `11_large_fulltext.sql`           500K stored full-text vector and GIN index

## Current Progress

-   [x] PostgreSQL 17 + pgvector
-   [x] Docker Compose environment
-   [x] Relational schema
-   [x] Dense embedding pipeline
-   [x] Exact vector retrieval
-   [x] HNSW ANN retrieval
-   [x] IVFFlat ANN evaluation
-   [x] PostgreSQL full-text search
-   [x] GIN indexing
-   [x] Hybrid retrieval with RRF
-   [x] 10K benchmark
-   [x] 100K scalability benchmark
-   [x] 500K MS MARCO scalability benchmark
-   [x] ANN recall/latency evaluation
-   [x] HNSW build-memory tuning
-   [x] FastAPI REST API
-   [x] PostgreSQL connection pooling
-   [x] Search-history persistence
-   [x] Document CRUD
-   [x] Many-to-many document tags
-   [x] FK cascade validation
-   [x] Modular backend architecture
-   [x] OpenAPI/API regression validation
-   [ ] Web interface/dashboard
-   [ ] Scalability plots for final presentation
-   [ ] ER diagram
-   [ ] Final report and live demo

A 1M-document benchmark is **not required for the current project
scope**. The completed 500K experiment is the final planned large-scale
benchmark unless additional scale is needed for a later research
question.

## Key Engineering Findings

1.  **HNSW provides a large speedup at high recall.** At 500K,
    `ef_search=40` achieves 0.981 Recall@10 with 2.215 ms P50 latency,
    compared with 102.918 ms for exact retrieval.
2.  **Embedding generation dominates large-scale ingestion.** At 500K,
    embedding generation took 849.79 s while database insertion took
    42.36 s.
3.  **Index-build memory materially affects build time.** At 100K,
    increasing `maintenance_work_mem` from 64 MB to 1 GB reduced HNSW
    build time from 37.848 s to 12.038 s.
4.  **Physical query-plan validation matters.** The final benchmark
    explicitly verifies sequential scan for exact retrieval and HNSW use
    for ANN retrieval.
5.  **Schema changes have operational costs at scale.** Stored full-text
    migration can trigger table rewrites and index rebuilds.
6.  **Connection pooling requires careful session-state handling.**
    Search-specific PostgreSQL settings must not leak across pooled
    connections.
7.  **Relational and vector workloads can coexist in one database.**
    CRUD, M:N tags, FK cascades, FTS, ANN, and search history are
    integrated in PostgreSQL.

## Limitations

-   ANN Recall@10 measures agreement with exact vector retrieval, not
    human relevance.
-   Small/limited query sets may not represent every workload.
-   10K/100K and 500K experiments use different corpora, so cross-scale
    comparisons are not strict controlled scaling experiments.
-   Latency depends on hardware, cache state, PostgreSQL configuration,
    query distribution, and workload.
-   Formal database retrieval benchmarks exclude embedding-model
    inference, while API vector/hybrid latency includes it.
-   The current hybrid method uses RRF rather than a learned fusion
    model.
-   The serving system uses HNSW; IVFFlat is retained as a benchmark
    comparison.
-   The Web UI/dashboard, ER diagram, and final report artifacts are not
    yet complete.

Future relevance evaluation could use real queries and qrels to report
MRR@10, nDCG@10, and relevance-oriented Recall@10. These should remain
conceptually separate from ANN recall against exact vector search.

## Roadmap

``` text
Core relational database                 DONE
        ↓
Exact / HNSW / IVFFlat retrieval         DONE
        ↓
PostgreSQL FTS + GIN                     DONE
        ↓
Hybrid RRF retrieval                     DONE
        ↓
500K scalability evaluation              DONE
        ↓
FastAPI + CRUD + history + tags          DONE
        ↓
Web interface / evaluation dashboard     NEXT
        ↓
ER diagram + final figures
        ↓
Final report + live demo
```

## Author

**Ravan Chuang**

Computer Science · Information Retrieval · Backend & Systems Engineering
