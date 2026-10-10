# Hybrid Vector Retrieval System

A PostgreSQL-based multi-stage information retrieval system combining
relational database design, full-text search, dense vector retrieval,
approximate nearest-neighbor (ANN) indexing, hybrid candidate
generation, and Cross-Encoder reranking.

> **Status:** Core database, application retrieval lifecycle, full-corpus
> MS MARCO ingestion and indexing, ANN evaluation, 10K-query relevance
> evaluation, multi-stage reranking, REST API, relational CRUD, search
> history, many-to-many tag management, regression tests, and the
> React/Vite dashboard are complete. The remaining work is final
> presentation/report polish and the live demo.


## Project Overview

![Hybrid Vector Retrieval System Overview](docs/system-overview.jpeg)

**Figure 1. Project status and key results on the full 8.84M-passage MS MARCO corpus.**

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
-   10K, 100K, 500K, and full 8.84M-passage scalability experiments
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

![System Architecture](docs/system-architecture.jpeg)

**Figure 2. System Architecture of the Hybrid Vector Retrieval System.**

The React/Vite dashboard communicates with the FastAPI backend through
REST/JSON. FastAPI uses psycopg 3 and `psycopg_pool`, invokes the local
`sentence-transformers/all-MiniLM-L6-v2` encoder for normalized
384-dimensional embeddings, and accesses PostgreSQL 17 with pgvector.
Lexical retrieval uses PostgreSQL full-text search with GIN and
`websearch_to_tsquery`; dense retrieval uses pgvector HNSW with cosine
distance. RRF and normalized score fusion are retained as first-stage
baselines. The strongest relevance pipeline retrieves lexical and dense
Top-50 candidates, forms their union, and reranks the candidate set with
`cross-encoder/ms-marco-MiniLM-L-6-v2` before returning Top-10. The
primary large-scale relevance result is evaluated on an independent
10,000-query sample over the full 8,841,823-passage MS MARCO corpus.

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

**Figure 3. Entity-Relationship Diagram of the normalized application
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
ANN Recall@10 measures HNSW agreement with exact dense Top-10 neighbors.
Relevance evaluation uses binary positive relevance judgments from the
MS MARCO `labeled-list` training split. These are **not** official MS
MARCO dev or leaderboard qrels.

The full 8,841,823-passage corpus contains all 408,158 unique positive
passages represented by this judgment source, covering 417,462 unique
positive query-passage judgments across 400,282 eligible queries. This
100% corpus coverage means that all positives from this source are
present locally; it does not imply complete human relevance judgments.

Method choices were frozen before the independent 10,000-query
full-corpus validation sample was evaluated. The sample uses seed 43 and
is disjoint from the earlier frozen 500-query full-corpus sample.
Weighted RRF uses `k=60`, lexical weight `0.1`, and dense weight `0.9`.
The large-scale relevance pipeline uses HNSW `ef_search=160` to reduce
ANN approximation effects during evaluation. Paired bootstrap confidence
intervals use 10,000 resamples.

## Final Relevance Results

### Full-Corpus 10K Evaluation (Primary)

| Pipeline | MRR@10 | nDCG@10 | Recall@10 |
|---|---:|---:|---:|
| Lexical | 0.1633 | 0.1940 | 0.2982 |
| Dense HNSW (`ef=160`) | 0.3101 | 0.3741 | 0.5879 |
| Weighted RRF (`0.1 lexical / 0.9 dense`) | 0.3219 | 0.3863 | 0.6010 |
| Dense Top-50 → Cross-Encoder | 0.4022 | 0.4703 | 0.6950 |
| **Dense ∪ Lexical Top-50 → Cross-Encoder** | **0.4166** | **0.4876** | **0.7211** |

On this independent 10,000-query validation sample, weighted RRF
outperformed dense retrieval by **+0.0122 nDCG@10** and **+0.0131
Recall@10**. The largest gain came from Cross-Encoder reranking: Dense
Top-50 → CE reached 0.4703 nDCG@10, while lexical+dense candidate union
reached **0.4876**.

Union CE versus Dense CE improved MRR@10 by **+0.0145** (95% CI
**[+0.0121, +0.0170]**), nDCG@10 by **+0.0173** (95% CI
**[+0.0147, +0.0199]**), and Recall@10 by **+0.0261** (95% CI
**[+0.0223, +0.0299]**). Lexical retrieval supplied 485 relevant
passages absent from Dense Top-50; **339/485 (69.9%)** were promoted into
the final Union Top-10 by the Cross-Encoder. At the query level, 336 of
480 affected queries (70.0%) received at least one promoted lexical-only
relevant passage. This supports lexical retrieval as complementary
candidate generation rather than establishing a new retrieval algorithm.

Repeated HNSW executions under the same `ef_search=160` protocol showed
98.04% exact ordered Top-10 agreement. Another 1.79% of queries had the
same Top-10 membership with ordering differences, only 0.17% had a
membership difference, and mean Top-10 overlap was 9.9983/10.

### Earlier 500-Query Evaluation (Supporting)

The earlier held-out evaluation remains as supporting evidence. On that
protocol, Dense → CE reached 0.6633 nDCG@10 and Dense ∪ Lexical → CE
reached 0.6788, with a +0.0155 paired improvement. The primary result
reported above is the larger independent 10K full-corpus evaluation.

Compact full-corpus protocol and result artifacts are versioned under
`artifacts/msmarco_full/`; large per-query outputs remain excluded from
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

## Full 8.84M MS MARCO Corpus

The final scale extension ingests and embeds the complete local MS MARCO
passage corpus used by this project: **8,841,823 passages** with
384-dimensional normalized MiniLM embeddings. PostgreSQL stores the
corpus and pgvector HNSW index; PostgreSQL FTS uses a GIN expression
index. The full database reached approximately 39 GB, with the HNSW index
approximately 17 GB.

### Full-Corpus Exact vs HNSW

The ANN benchmark uses 100 real MS MARCO training queries. Query vectors
are generated client-side so the exact baseline does not accidentally
include a corpus scan to obtain the query embedding. QPS is reported as
`1000 / mean latency` for this sequential benchmark and is not concurrent
throughput.

| Method | ANN Recall@10 | Mean (ms) | P50 (ms) | P95 (ms) | Approx. QPS |
|---|---:|---:|---:|---:|---:|
| Exact | 1.000 | 5781.9 | 5751.5 | 6949.7 | 0.17 |
| HNSW `ef=10` | 0.692 | 49.5 | 45.7 | 79.8 | 20.19 |
| HNSW `ef=20` | 0.780 | 22.0 | 17.5 | 52.9 | 45.52 |
| HNSW `ef=40` | 0.858 | 38.4 | 31.9 | 92.5 | 26.04 |
| HNSW `ef=80` | 0.905 | 61.6 | 57.8 | 118.3 | 16.23 |
| HNSW `ef=160` | 0.939 | 102.6 | 95.6 | 169.6 | 9.74 |

`ef_search=40` is retained as an efficiency-oriented operating point,
while `ef_search=160` is used for the 10K relevance evaluation to reduce
ANN approximation effects. The anomalous `ef=10` versus `ef=20` latency
ordering is treated as an empirical benchmark artifact rather than a
theoretical property of HNSW.

The full-corpus GIN index is approximately 904 MB. The HNSW graph was
built with `m=16` and `ef_construction=64`; the container shared-memory
allocation was increased to 8 GB to support the large index build. The
exact HNSW build duration was not recorded and is therefore not reported.

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

    shm_size: '8gb'

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

```text
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
│   ├── scripts/
│   │   ├── ingest_msmarco_full.py
│   │   ├── embed_msmarco_full.py
│   │   ├── benchmark_msmarco_full_ann.py
│   │   ├── build_msmarco_full_eval.py
│   │   ├── build_msmarco_full_eval_10k.py
│   │   ├── evaluate_msmarco_full_retrieval.py
│   │   ├── evaluate_msmarco_full_retrieval_10k.py
│   │   ├── extract_msmarco_full_ce_candidates.py
│   │   ├── extract_msmarco_full_ce_candidates_10k.py
│   │   ├── rerank_msmarco_full_ce.py
│   │   └── rerank_msmarco_full_ce_10k.py
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
│   ├── 12_application_retrieval.sql
│   ├── 13_msmarco_full.sql
│   └── 14_msmarco_full_indexes.sql
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
│   ├── relevance/
│   │   ├── protocol_frozen.json
│   │   ├── final_retrieval_summary.json
│   │   ├── final_reranking_analysis.json
│   │   └── final_split_manifest.json
│   └── msmarco_full/
│       ├── ann_benchmark.json
│       ├── eval_manifest_10k.json
│       ├── first_stage_10k_paired.json
│       ├── ce_rerank_10k_summary.json
│       ├── dense_integrity_10k.json
│       └── multistage_10k_summary.json
├── docs/
│   ├── system-overview.jpeg
│   ├── system-architecture.jpeg
│   └── database-erd.jpg
├── .env.example
├── .gitignore
├── docker-compose.yml
├── pytest.ini
├── requirements.txt
└── README.md
```
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
  `12_application_retrieval.sql`       Application FTS/GIN + HNSW retrieval indexes
`13_msmarco_full.sql`              Full 8.84M MS MARCO corpus table
`14_msmarco_full_indexes.sql`      Full-corpus GIN + HNSW indexes
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
-   [x] Full 8.84M MS MARCO corpus ingestion and embeddings
-   [x] Full-corpus GIN + HNSW indexes
-   [x] Full-corpus ANN benchmark
-   [x] Independent 10K-query full-corpus relevance evaluation
-   [x] 10K Cross-Encoder reranking and lexical-rescue analysis
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

## Key Engineering Findings

1. **HNSW remains practical at full-corpus scale.** On 8.84M passages,
   `ef_search=40` achieved 0.858 ANN Recall@10 with 31.9 ms P50 versus
   5751.5 ms for exact dense retrieval.
2. **Dense-dominant hybrid fusion is reproducibly useful.** On the
   independent 10K evaluation, weighted RRF improved nDCG@10 from 0.3741
   to 0.3863 and Recall@10 from 0.5879 to 0.6010.
3. **Reranking produces the largest relevance gain.** Dense Top-50 → CE
   increased nDCG@10 from 0.3741 to 0.4703.
4. **Lexical candidates add complementary relevance.** Union CE reached
   **0.4876 nDCG@10** and **0.7211 Recall@10**; ΔnDCG@10 over Dense CE
   was +0.0173 with 95% CI [+0.0147, +0.0199].
5. **The rescue mechanism is observable at 10K scale.** 339 of 485
   lexical-only relevant passages (69.9%) were promoted into final Top-10.
6. **The full-corpus relevance protocol is stable across repeated ANN
   executions.** Mean Top-10 overlap was 9.9983/10 under `ef_search=160`.
7. **Database resource configuration matters at index-build scale.** The
   full-corpus HNSW build required increasing Docker shared memory to 8 GB
   while using 4 GB `maintenance_work_mem`.
8. **Database correctness remains part of the retrieval system.**
   Application create → search → update → search → delete → search behavior
   is covered by integration/regression tests alongside PK/FK, M:N tags,
   and cascades.

## Limitations

- Relevance judgments are binary positives from the MS MARCO
  `labeled-list` training split, not official dev/leaderboard qrels.
- Full-corpus coverage is 100% for positives represented by this source,
  but the judgments are still incomplete as human relevance labels.
- ANN Recall@10 measures exact-neighbor agreement; relevance Recall@10
  uses qrels. They answer different questions.
- Latency depends on hardware, cache state, PostgreSQL configuration,
  query distribution, and workload.
- Formal database retrieval latency excludes query-embedding inference.
- Cross-Encoder latency reported by the relevance pipeline is reranker
  inference latency, not end-to-end or concurrent-serving latency.
- Cross-Encoder reranking, RRF, and hybrid candidate generation are
  established techniques; this project evaluates and integrates them
  rather than claiming a new retrieval algorithm.
- The system is an engineering/research prototype, not a production
  service with authentication, observability, replication/failover,
  online index maintenance, SLOs, backup/recovery, and sustained load
  testing.

## Suggested Demo Flow

1. Show the normalized schema: PK/FK, M:N tags, constraints, generated FTS
   vectors, GIN/HNSW indexes, and cascade behavior.
2. Open the dashboard and verify PostgreSQL/pgvector statistics.
3. Demonstrate the application corpus lifecycle: create → search → update
   → search → tag → delete → search.
4. Run a benchmark query with Lexical, Vector, and Hybrid retrieval and
   explain GIN, HNSW, RRF, candidate depth, and `ef_search`.
5. Present the full 8.84M ANN result: exact P50 5751.5 ms versus HNSW
   `ef_search=40` P50 31.9 ms at 0.858 ANN Recall@10.
6. Close with the independent 10K relevance evaluation: Dense 0.3741 →
   Weighted RRF 0.3863 → Dense CE 0.4703 → **Union CE 0.4876 nDCG@10**,
   with Union-vs-Dense-CE 95% CI **[+0.0147, +0.0199]**.

## Roadmap

```text
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
Full 8.84M MS MARCO ingestion + indexing         DONE
        ↓
Full-corpus ANN benchmark                        DONE
        ↓
Independent 10K relevance evaluation             DONE
        ↓
Cross-Encoder + lexical-rescue analysis           DONE
        ↓
React/Vite dashboard                              DONE
        ↓
Final report + presentation + live demo           IN PROGRESS
```

## Author

**Ravan Chuang**

Computer Science · Information Retrieval · Backend & Systems Engineering

## License

This project is licensed under the MIT License. See `LICENSE` for
details.

Third-party datasets, models, libraries, and other dependencies remain
subject to their respective licenses and terms.
