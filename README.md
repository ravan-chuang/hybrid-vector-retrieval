# Hybrid Vector Retrieval System

A PostgreSQL-based multi-stage information retrieval system combining
relational database design, full-text search, dense vector retrieval,
approximate nearest-neighbor (ANN) indexing, hybrid candidate
generation, and Cross-Encoder reranking.

> **Status:** Core database, application retrieval lifecycle, 500K
> scalability evaluation, qrels-based relevance evaluation, multi-stage
> reranking, REST API, relational CRUD, search history, many-to-many tag
> management, regression tests, and the React/Vite dashboard are
> complete. The remaining work is final presentation/report polish and
> the live demo.

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
-   Dense + lexical candidate union with Cross-Encoder reranking
-   Qrels-based MRR@10, nDCG@10, and relevance Recall@10 evaluation
-   Paired bootstrap confidence intervals and lexical-rescue analysis
-   ANN recall, latency, throughput, index-size, and build-time
    evaluation
-   PostgreSQL/Docker resource tuning
-   10K, 100K, and 500K scalability experiments
-   FastAPI REST API with PostgreSQL connection pooling
-   Document CRUD and persisted search history
-   Many-to-many document/tag API with FK cascade validation
-   React/Vite retrieval and relational database dashboard
-   End-to-end Lexical, Vector, and Hybrid search UI
-   Browser-based document CRUD and tag-association management

The project is designed as both a **database systems project** and a
**retrieval engineering project**, with emphasis on schema design,
indexing, query execution, performance measurement, and scalability.

## Architecture

The system separates the normalized application schema from the large
retrieval-serving corpus while keeping both workloads in PostgreSQL.

![System Architecture](docs/system-architecture.png)

**Figure 1. System Architecture of the Hybrid Vector Retrieval System.**

The React/Vite dashboard communicates with the FastAPI backend through
REST/JSON. FastAPI uses psycopg 3 and `psycopg_pool`, invokes the local
`sentence-transformers/all-MiniLM-L6-v2` encoder for normalized
384-dimensional embeddings, and accesses PostgreSQL 17 with pgvector.
Lexical retrieval uses PostgreSQL full-text search with GIN and
`websearch_to_tsquery`; dense retrieval uses pgvector HNSW with cosine
distance. RRF and normalized score fusion are retained as first-stage
baselines. The strongest relevance pipeline retrieves lexical and dense
Top-50 candidates, forms their union, and reranks the candidate set with
`cross-encoder/ms-marco-MiniLM-L-6-v2` before returning Top-10.

The normalized application schema is kept separate from the large-scale
benchmark/serving corpus. IVFFlat is retained as an experimental ANN
baseline, while the serving path uses HNSW.

## Technology Stack

-   **Database:** PostgreSQL 17, pgvector, PostgreSQL Full-Text Search
-   **Indexes:** B-tree, GIN, HNSW, IVFFlat
-   **Backend / API:** Python 3, FastAPI, psycopg, psycopg_pool, Uvicorn
-   **Retrieval / Evaluation:** NumPy, Sentence Transformers, Hugging
    Face Datasets
-   **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`, 384
    dimensions, normalized embeddings
-   **Frontend:** React 19, Vite 8, plain CSS
-   **Infrastructure:** Docker, Docker Compose

## Database Design

![Database ERD](docs/database-erd.jpg)

**Figure 2. Entity-Relationship Diagram of the normalized application
schema.**

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

The application `document_chunks` table also maintains a generated
PostgreSQL `tsvector`, a GIN full-text index, and a pgvector HNSW index
so newly created or updated application documents are searchable.
Separate benchmark tables isolate performance and scalability
experiments from the primary application schema.

### Relational Integrity

The normalized application schema includes the following integrity
rules:

-   `documents → document_chunks`: one-to-many, `ON DELETE CASCADE`
-   `documents ↔ tags`: many-to-many through `document_tags`
-   `document_tags`: composite primary key `(document_id, tag_id)`
-   `document_chunks`: unique `(document_id, chunk_index)`
-   `users → search_history`: nullable foreign key with
    `ON DELETE SET NULL`
-   `tags.name`, `users.username`, and `users.email`: unique constraints
-   `search_history.search_type`: constrained to `keyword`, `vector`, or
    `hybrid`

Frontend CRUD tests were verified directly against PostgreSQL. Deleting
a document removed its `document_chunks` and `document_tags` rows while
the referenced tag rows remained available for reuse by other documents.

## Retrieval Methods

### Exact Vector Search

Queries and documents are represented as normalized 384-dimensional
embeddings. Exact cosine-distance retrieval uses pgvector:

    SELECT document_id,
           content,
           embedding <=> %s::vector AS distance
    FROM benchmark_documents
    ORDER BY embedding <=> %s::vector
    LIMIT 10;

For normalized vectors:

    cosine_distance = 1 - cosine_similarity

Exact retrieval is also used as ground truth for ANN Recall@10.

### HNSW

HNSW provides low-latency approximate nearest-neighbor retrieval.

    CREATE INDEX idx_benchmark_embedding_hnsw
    ON benchmark_documents
    USING hnsw (embedding vector_cosine_ops)
    WITH (
        m = 16,
        ef_construction = 64
    );

Experiments vary `hnsw.ef_search` to evaluate recall/latency trade-offs.

### IVFFlat

IVFFlat partitions vector space into lists and searches selected
partitions.

    CREATE INDEX idx_benchmark_embedding_ivfflat
    ON benchmark_documents
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

Experiments vary `ivfflat.probes`.

### Full-Text Search

Lexical retrieval uses PostgreSQL `tsvector` and `tsquery`, accelerated
by a GIN index.

    CREATE INDEX idx_benchmark_search_vector
    ON benchmark_documents
    USING GIN (search_vector);

A trigger keeps `search_vector` synchronized with document content.

### Hybrid Retrieval

Reciprocal Rank Fusion (RRF) is implemented as a first-stage hybrid
baseline:

``` text
RRF(d) = Σ 1 / (k + rank(d))
```

The baseline uses `k = 60`. Development experiments also evaluate
dense-dominant weighted RRF and query-local normalized score fusion.
These experiments showed that naïve equal-weight fusion can underperform
dense retrieval, motivating a multi-stage candidate-generation design.

### Cross-Encoder Reranking

The frozen final relevance pipeline is:

``` text
Query
  ├─ PostgreSQL FTS Top-50
  └─ HNSW Dense Top-50
          ↓
   Candidate Union
          ↓
cross-encoder/ms-marco-MiniLM-L-6-v2
          ↓
      Final Top-10
```

The Cross-Encoder scores query-passage pairs after first-stage
retrieval. This preserves lexical recall as a complementary candidate
source while allowing the reranker to determine the final ordering.

## Benchmark Methodology

Database retrieval latency is measured separately from query-embedding
generation. Benchmarks use warm-ups, persistent database connections,
repeated measurements, and report P50/P95/P99 latency and approximate
QPS.

ANN Recall@10 is measured against exact vector Top-K:

    Recall@10 = |Top10_ANN ∩ Top10_Exact| / 10

This measures agreement with exact vector retrieval, **not**
human-judged end-to-end relevance.

## Relevance Evaluation Methodology

ANN Recall@10 and human-relevance evaluation are reported separately.
ANN Recall@10 measures HNSW agreement with exact vector neighbors.
Relevance evaluation uses filtered binary relevance judgments from the
MS MARCO `labeled-list` training split restricted to the local
500K-passage corpus.

The local subset contains 28,329 relevant judgments across 27,329
eligible queries, corresponding to 6.786% of the original relevant
judgments. This subset therefore has incomplete judgments and
corpus-subset bias and is not presented as an official MS MARCO
leaderboard result.

Query partitions were frozen before final evaluation. Development data
was used for weighted-RRF, score-fusion, adaptive-routing, and reranking
experiments. The final 500-query test set was opened only after the
Cross-Encoder pipeline was frozen. Paired 10,000-sample bootstrap
confidence intervals are used for the final Dense-CE vs Union-CE
comparison.

## Final Relevance Results

  --------------------------------------------------------------------------
  Pipeline                      MRR@10            nDCG@10          Recall@10
  ----------------- ------------------ ------------------ ------------------
  Lexical                       0.2239             0.2645             0.3950

  Dense                         0.5207             0.5993             0.8487

  Equal RRF                     0.4754             0.5620             0.8407

  Dense Top-50 →                0.5796             0.6633             0.9257
  Cross-Encoder                                           

  **Dense ∪ Lexical         **0.5931**         **0.6788**         **0.9493**
  → Cross-Encoder**                                       
  --------------------------------------------------------------------------

Candidate union before reranking improved nDCG@10 by **+0.0155**, with
paired-bootstrap 95% CI **\[+0.0070, +0.0253\]**. MRR@10 improved by
**+0.0134** (95% CI **\[+0.0052, +0.0233\]**) and Recall@10 by
**+0.0237** (95% CI **\[+0.0120, +0.0373\]**).

Lexical-rescue analysis found 14 relevant documents absent from Dense
Top-50 but present in the lexical-only candidate set; **13/14** were
promoted into the final Union Top-10. This supports the interpretation
that lexical retrieval is most useful here as complementary candidate
generation rather than naïve equal-weight first-stage fusion.

Compact frozen protocol and result artifacts are versioned under
`artifacts/relevance/`; large per-query outputs remain excluded from
Git.

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

  ------------------------------------------------------------------------
  Method       Recall@10 Mean (ms)  P50 (ms)  P95 (ms)  P99 (ms)       QPS
  ---------- ----------- --------- --------- --------- --------- ---------
  Exact            1.000    66.280    65.117    75.072    77.879     15.09

  HNSW             0.942     0.948     0.890     1.429     1.628   1054.70
  `ef=10`                                                        

  HNSW             0.965     1.090     1.062     1.742     1.898    917.83
  `ef=20`                                                        

  HNSW             0.992     1.387     1.405     1.978     2.449    720.74
  `ef=40`                                                        

  HNSW             0.994     2.039     1.946     3.246     3.489    490.52
  `ef=80`                                                        

  HNSW             0.997     3.047     2.973     4.655     5.337    328.24
  `ef=160`                                                       
  ------------------------------------------------------------------------

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

  ------------------------------------------------------------------------
  Method       Recall@10 Mean (ms)  P50 (ms)  P95 (ms)  P99 (ms)       QPS
  ---------- ----------- --------- --------- --------- --------- ---------
  Exact            1.000   103.348   102.918   108.737   111.612      9.68

  HNSW             0.919     1.544     1.534     2.354     2.893    647.73
  `ef=10`                                                        

  HNSW             0.949     1.639     1.579     2.539     2.902    610.28
  `ef=20`                                                        

  HNSW             0.981     2.157     2.215     3.248     4.339    463.55
  `ef=40`                                                        

  HNSW             0.984     3.427     3.526     5.404     6.310    291.83
  `ef=80`                                                        

  HNSW             0.995     5.853     5.695     9.134    10.057    170.84
  `ef=160`                                                       
  ------------------------------------------------------------------------

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

  ---------------------------------------------------------------------
  Metric                                                         Result
  ----------------------------- ---------------------------------------
  HNSW build time with                                          83.22 s
  `maintenance_work_mem=1GB`    

  HNSW index size                                                976 MB

  Tuples fitting in HNSW build                        482,217 / 500,000
  memory before overflow        

  Fraction fitting before                                        96.44%
  overflow                      

  Current serving table size                                     451 MB

  Current indexes size                                         1,071 MB

  Current total relation                                       2,256 MB
  footprint                     

  GIN full-text index                                             69 MB
  ---------------------------------------------------------------------

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

    maintenance_work_mem = 64MB

PostgreSQL reported that the HNSW graph stopped fitting in
`maintenance_work_mem` after approximately 28K tuples.

  Configuration                     Build Time   HNSW Size
  ------------------------------- ------------ -----------
  `maintenance_work_mem = 64MB`       37.848 s      195 MB
  `maintenance_work_mem = 1GB`        12.038 s      195 MB

The Docker container's shared memory was increased to:

    shm_size: '2gb'

and the tuned build used:

    SET maintenance_work_mem = '1GB';

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

    set -a
    source .env
    set +a

    uvicorn backend.api.main:app --reload --host 127.0.0.1 --port 8000

Interactive OpenAPI documentation is available at
`http://127.0.0.1:8000/docs`.

  --------------------------------------------------------------------------------
  Method             Endpoint                                   Purpose
  ------------------ ------------------------------------------ ------------------
  `GET`              `/`                                        API information

  `GET`              `/health`                                  Health/database
                                                                connectivity

  `GET`              `/stats`                                   Corpus, model,
                                                                database, and
                                                                index statistics

  `POST`             `/search/vector`                           HNSW semantic
                                                                retrieval

  `POST`             `/search/lexical`                          PostgreSQL FTS/GIN
                                                                retrieval

  `POST`             `/search/hybrid`                           Lexical + vector
                                                                retrieval with RRF

  `GET`              `/search/history`                          Persisted recent
                                                                searches

  `GET`              `/documents`                               List application
                                                                documents

  `POST`             `/documents`                               Create a document
                                                                and embedding

  `GET`              `/documents/{document_id}`                 Retrieve one
                                                                document

  `PUT`              `/documents/{document_id}`                 Update a document

  `DELETE`           `/documents/{document_id}`                 Delete a document
                                                                with FK cascades

  `GET`              `/documents/{document_id}/tags`            List document tags

  `POST`             `/documents/{document_id}/tags`            Create/reuse and
                                                                attach a tag

  `DELETE`           `/documents/{document_id}/tags/{tag_id}`   Remove a
                                                                document/tag
                                                                association
  --------------------------------------------------------------------------------

### API Examples

Vector search:

    curl -X POST http://127.0.0.1:8000/search/vector \
      -H "Content-Type: application/json" \
      -d '{"query":"Manhattan Project","top_k":5,"ef_search":40}'

Hybrid search:

    curl -X POST http://127.0.0.1:8000/search/hybrid \
      -H "Content-Type: application/json" \
      -d '{"query":"atomic bomb World War II","top_k":5,"candidate_k":50,"ef_search":40,"rrf_k":60}'

Attach a tag:

    curl -X POST http://127.0.0.1:8000/documents/1/tags \
      -H "Content-Type: application/json" \
      -d '{"name":"database"}'

The tag endpoint reuses unique tags and creates the
`(document_id, tag_id)` association safely. Tests verified that the same
tag can be shared by multiple documents, removing an association does
not delete the tag, and deleting a document cascades to its
`document_chunks` and `document_tags`.

### Backend Architecture

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

The API was refactored from a monolithic implementation into modular
routers, schemas, services, configuration, runtime state, and
database-pool management. Search-specific PostgreSQL settings use
transaction-local state where appropriate so settings do not leak across
pooled connections.

Search-history logging is performed after the measured retrieval timing,
so the history insert itself is not included in reported search latency.

## Web Dashboard

The React/Vite frontend provides a single engineering dashboard for both
retrieval and relational database operations.

### Retrieval UI

-   Switch between **Hybrid**, **Vector**, and **Lexical** search
-   Inspect embedding, retrieval, fusion, and total latency
-   View similarity, FTS rank, RRF score, and component ranks
-   Inspect recent persisted search history
-   View live database, corpus, model, and index statistics

### Database Management UI

-   List and inspect normalized application documents
-   Create, read, update, and delete documents
-   Generate/update document embeddings through the backend
-   Create or reuse tags and attach them to documents
-   Remove only the `document_tags` association without deleting the tag
-   Demonstrate database-managed FK cascades when deleting a document

The UI is intentionally a thin client: relational integrity, vector
operations, and persistence remain enforced by the FastAPI/PostgreSQL
backend.

## Project Structure

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
    │   ├── evaluate_relevance.py
    │   ├── evaluate_cross_encoder_reranking.py
    │   ├── analyze_reranking.py
    │   ├── sweep_weighted_rrf.py
    │   ├── sweep_score_fusion.py
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
    │   ├── 11_large_fulltext.sql
    │   └── 12_application_retrieval.sql
    ├── frontend/
    │   ├── public/
    │   ├── src/
    │   │   ├── components/
    │   │   │   └── DocumentManager.jsx
    │   │   ├── App.jsx
    │   │   ├── api.js
    │   │   ├── index.css
    │   │   └── main.jsx
    │   ├── package.json
    │   └── vite.config.js
    ├── artifacts/
    │   └── relevance/
    │       ├── protocol_frozen.json
    │       ├── final_retrieval_summary.json
    │       ├── final_reranking_analysis.json
    │       └── final_split_manifest.json
    ├── docs/
    │   ├── system-architecture.png
    │   └── database-erd.jpg
    ├── .env.example
    ├── .gitignore
    ├── docker-compose.yml
    ├── requirements.txt
    └── README.md

## Setup

### 1. Clone

    git clone https://github.com/ravan-chuang/hybrid-vector-retrieval.git
    cd hybrid-vector-retrieval

### 2. Configure Environment Variables

    cp .env.example .env

Edit `.env` and replace the example password with a local development
password.

    POSTGRES_DB=retrieval_db
    POSTGRES_USER=retrieval
    POSTGRES_PASSWORD=change_me
    POSTGRES_PORT=5433
    DATABASE_URL=postgresql://retrieval:change_me@127.0.0.1:5433/retrieval_db

Never commit the real `.env`.

### 3. Start PostgreSQL + pgvector

    docker compose up -d

Verify:

    docker exec hybrid-retrieval-db pg_isready -U retrieval -d retrieval_db

### 4. Create Python Environment

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

### 5. Load Environment Variables

    set -a
    source .env
    set +a

### 6. Initialize Schema

    docker exec -i hybrid-retrieval-db psql -U retrieval -d retrieval_db < database/01_schema.sql

### 7. Run the API

    uvicorn backend.api.main:app --reload --host 127.0.0.1 --port 8000

Open `http://127.0.0.1:8000/docs` for the interactive API documentation.

### 8. Run the Web Dashboard

In a second terminal:

``` bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

The development frontend expects the API at `http://127.0.0.1:8000`. The
backend CORS configuration permits the Vite development origin.

### 9. Optional Normalized-Schema Demo Data

For the small normalized-schema demo, `backend/seed_documents.py` can be
used to seed application documents. The 500K corpus itself is
intentionally not committed to Git.

## Main Scripts

  -------------------------------------------------------------------------
  Script                                   Purpose
  ---------------------------------------- --------------------------------
  `benchmark_exact.py`                     10K exact vector-search baseline

  `benchmark_hnsw.py`                      10K HNSW recall/latency
                                           evaluation

  `benchmark_ivfflat.py`                   10K IVFFlat recall/latency
                                           evaluation

  `benchmark_scalability_exact.py`         100K exact baseline

  `benchmark_scalability_hnsw.py`          100K HNSW evaluation

  `benchmark_scalability_large_exact.py`   500K validated exact baseline

  `benchmark_scalability_large_hnsw.py`    500K validated HNSW evaluation

  `hybrid_search.py`                       Full-text + dense retrieval +
                                           RRF

  `ingest_benchmark.py`                    Small benchmark ingestion

  `ingest_scalability.py`                  100K scalability ingestion

  `ingest_scalability_large.py`            Streaming 500K MS MARCO
                                           ingestion

  `seed_documents.py`                      Seed normalized application
                                           documents
  -------------------------------------------------------------------------

## SQL Files

  File                               Purpose
  ---------------------------------- --------------------------------------------
  `01_schema.sql`                    Primary normalized relational schema
  `02_benchmark.sql`                 Benchmark table
  `03_hnsw.sql`                      HNSW index
  `04_ivfflat.sql`                   IVFFlat index
  `05_fulltext.sql`                  Full-text search and GIN index
  `06_scalability.sql`               100K scalability schema
  `07_scalability_hnsw.sql`          100K HNSW index
  `08_scalability_hnsw_tuned.sql`    Tuned HNSW build
  `09_scalability_large.sql`         500K scalability schema
  `10_scalability_large_hnsw.sql`    500K HNSW index
  `11_large_fulltext.sql`            500K stored full-text vector and GIN index
  2_application_retrieval.sql\` Ap   plication FTS/GIN + HNSW retrieval indexes

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
-   [x] Application document lifecycle integrated with retrieval
-   [x] Qrels-based relevance evaluation
-   [x] Weighted RRF and normalized score-fusion ablations
-   [x] Adaptive-routing negative ablation
-   [x] Cross-Encoder reranking
-   [x] Frozen 500-query final relevance evaluation
-   [x] Paired bootstrap and lexical-rescue analysis
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
-   [x] Web interface/dashboard
-   [x] System architecture and database ERD figures
-   [ ] Final presentation plots
-   [x] ER diagram design / schema validation
-   [ ] Final report and live demo

A 1M-document benchmark is **not required for the current project
scope**. The completed 500K experiment is the final planned large-scale
benchmark unless additional scale is needed for a later research
question.

## Key Engineering Findings

1.  **HNSW provides a large speedup at high ANN recall.** At 500K,
    `ef_search=40` achieved 0.981 exact-neighbor Recall@10 with 2.215 ms
    P50, versus 102.918 ms for exact retrieval: approximately **46.47×**
    lower P50.
2.  **Naïve hybrid fusion is not automatically better.** Equal RRF
    reached 0.5620 nDCG@10 versus 0.5993 for dense retrieval on the
    final set.
3.  **Reranking produces the largest relevance gain.** Dense Top-50 →
    Cross-Encoder increased nDCG@10 from 0.5993 to 0.6633.
4.  **Lexical retrieval is valuable as complementary candidate
    generation.** Dense ∪ Lexical → Cross-Encoder reached **0.6788
    nDCG@10** and **0.9493 Recall@10**; ΔnDCG@10 over Dense → CE was
    +0.0155 with 95% CI \[+0.0070, +0.0253\].
5.  **The rescue mechanism is observable.** Lexical retrieval
    contributed 14 relevant documents absent from Dense Top-50, and 13
    were promoted into the final Top-10 by the Cross-Encoder.
6.  **Embedding generation dominates large-scale ingestion.** At 500K,
    embedding generation took 849.79 s while PostgreSQL insertion took
    42.36 s.
7.  **Index-build memory materially affects build time.** At 100K,
    increasing `maintenance_work_mem` from 64 MB to 1 GB reduced HNSW
    build time from 37.848 s to 12.038 s.
8.  **Database correctness remains part of the retrieval system.**
    Application create → search → update → search → delete → search
    behavior is covered by integration/regression tests alongside PK/FK,
    M:N tags, and cascades.

## Limitations

-   Relevance judgments are filtered from the MS MARCO `labeled-list`
    training split to the local 500K passage subset; they are incomplete
    and are not official MS MARCO leaderboard results.
-   The local corpus retains 6.786% of the original relevant judgments,
    introducing subset and judgment-coverage bias.
-   ANN Recall@10 measures exact-neighbor agreement; relevance Recall@10
    uses qrels. They answer different questions.
-   10K, 100K, and 500K experiments are not a perfectly controlled
    scaling study.
-   Latency depends on hardware, cache state, PostgreSQL configuration,
    query distribution, and workload.
-   Formal database retrieval latency excludes query-embedding
    inference.
-   Cross-Encoder latency is an engineering measurement rather than a
    controlled concurrent-serving benchmark; Dense-CE and Union-CE were
    not randomized in execution order.
-   Cross-Encoder reranking and hybrid candidate generation are
    established techniques; this project evaluates and integrates them
    rather than claiming a new retrieval algorithm.
-   The system is an engineering/research prototype, not a production
    service with authentication, observability, replication/failover,
    online index maintenance, SLOs, backup/recovery, and sustained load
    testing.

## Suggested Demo Flow

1.  Show the normalized schema: PK/FK, M:N tags, constraints, generated
    FTS vectors, GIN/HNSW indexes, and cascade behavior.
2.  Open the dashboard and verify PostgreSQL/pgvector statistics.
3.  Demonstrate the application corpus lifecycle: create → search →
    update → search → tag → delete → search.
4.  Run a benchmark query with Lexical, Vector, and Hybrid retrieval and
    explain GIN, HNSW, RRF, candidate depth, and `ef_search`.
5.  Present the 500K ANN result: HNSW `ef_search=40` achieved **0.981
    Recall@10**, **2.215 ms P50**, and approximately **46.47×** lower
    P50 than exact retrieval.
6.  Close with relevance evaluation: Dense → CE reached 0.6633 nDCG@10;
    Dense ∪ Lexical → CE reached **0.6788**, with 95% CI **\[+0.0070,
    +0.0253\]** for the +0.0155 improvement.

## Roadmap

``` text
Core relational database                         DONE
        ↓
Exact / HNSW / IVFFlat retrieval                 DONE
        ↓
PostgreSQL FTS + GIN                             DONE
        ↓
500K scalability evaluation                      DONE
        ↓
Application CRUD ↔ retrieval lifecycle           DONE
        ↓
Qrels-based relevance evaluation                 DONE
        ↓
Fusion / adaptive ablations                      DONE
        ↓
Cross-Encoder multi-stage reranking               DONE
        ↓
Frozen final evaluation + paired analysis         DONE
        ↓
React/Vite dashboard                              DONE
        ↓
Final report + presentation + live demo           IN PROGRESS
```

A 1M-document benchmark is intentionally outside the current project
scope. The validated 500K experiment is the final planned large-scale
benchmark unless a later research question requires additional scale.

## Author

**Ravan Chuang**

Computer Science · Information Retrieval · Backend & Systems Engineering

## License

This project is licensed under the MIT License. See `LICENSE` for
details.

Third-party datasets, models, libraries, and other dependencies remain
subject to their respective licenses and terms.
