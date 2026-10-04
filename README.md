# Hybrid Vector Retrieval System

A PostgreSQL-based hybrid information retrieval system combining relational database design, full-text search, dense vector retrieval, approximate nearest-neighbor (ANN) indexing, and Reciprocal Rank Fusion (RRF).

> **Status:** Active development — scalability evaluation and system integration are in progress.

## Overview

This project integrates traditional relational database functionality with modern semantic retrieval using PostgreSQL and pgvector.

Current capabilities include:

- Relational document and metadata storage
- 384-dimensional dense embeddings
- Exact cosine vector search
- PostgreSQL full-text search with GIN
- HNSW approximate nearest-neighbor retrieval
- IVFFlat approximate nearest-neighbor retrieval
- Hybrid lexical + semantic retrieval using Reciprocal Rank Fusion
- ANN recall, latency, throughput, index-size, and build-time evaluation
- PostgreSQL/Docker resource tuning
- 10K and 100K scalability experiments

The project is designed as both a **database systems project** and a **retrieval engineering project**, with emphasis on schema design, indexing, query execution, performance measurement, and scalability.

## Architecture

```text
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

**Database**
- PostgreSQL 17
- pgvector
- PostgreSQL Full-Text Search
- GIN, HNSW, and IVFFlat indexes

**Backend / Evaluation**
- Python 3
- psycopg
- NumPy
- Sentence Transformers
- Hugging Face Datasets

**Embedding Model**
- `sentence-transformers/all-MiniLM-L6-v2`
- 384 dimensions
- normalized embeddings

**Infrastructure**
- Docker
- Docker Compose

## Database Design

The primary relational schema contains:

| Table | Purpose |
|---|---|
| `users` | User information |
| `documents` | Document-level metadata |
| `document_chunks` | Text chunks and vector embeddings |
| `tags` | Unique tags |
| `document_tags` | Many-to-many document/tag relationship |
| `search_history` | Query type, result count, latency, and history |

The schema demonstrates primary/foreign keys, referential integrity, unique and check constraints, many-to-many relationships, JOIN-based retrieval, and relational metadata + vector data integration.

Separate benchmark tables isolate performance and scalability experiments from the primary application schema.

## Retrieval Methods

### Exact Vector Search

Queries and documents are represented as normalized 384-dimensional embeddings. Exact cosine-distance retrieval uses pgvector:

```sql
SELECT document_id,
       content,
       embedding <=> %s::vector AS distance
FROM benchmark_documents
ORDER BY embedding <=> %s::vector
LIMIT 10;
```

For normalized vectors:

```text
cosine_distance = 1 - cosine_similarity
```

Exact retrieval is also used as ground truth for ANN Recall@10.

### HNSW

HNSW provides low-latency approximate nearest-neighbor retrieval.

```sql
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

IVFFlat partitions vector space into lists and searches selected partitions.

```sql
CREATE INDEX idx_benchmark_embedding_ivfflat
ON benchmark_documents
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

Experiments vary `ivfflat.probes`.

### Full-Text Search

Lexical retrieval uses PostgreSQL `tsvector` and `tsquery`, accelerated by a GIN index.

```sql
CREATE INDEX idx_benchmark_search_vector
ON benchmark_documents
USING GIN (search_vector);
```

A trigger keeps `search_vector` synchronized with document content.

### Hybrid Retrieval

Hybrid retrieval combines lexical and semantic candidates using Reciprocal Rank Fusion:

```text
RRF(d) = Σ 1 / (k + rank(d))
```

The current implementation uses `k = 60`. RRF avoids requiring lexical and vector scores to share the same numerical scale.

## Benchmark Methodology

Database retrieval latency is measured separately from query-embedding generation. Benchmarks use warm-ups, persistent database connections, repeated measurements, and report P50/P95/P99 latency and approximate QPS.

ANN Recall@10 is measured against exact vector Top-K:

```text
Recall@10 = |Top10_ANN ∩ Top10_Exact| / 10
```

This measures agreement with exact vector retrieval, **not** human-judged end-to-end relevance.

## Experimental Results

### 10K Documents — Exact vs HNSW

| Method | Recall@10 | Mean (ms) | P50 (ms) | P95 (ms) | QPS |
|---|---:|---:|---:|---:|---:|
| Exact | 1.000 | 3.335 | 3.219 | 4.192 | 299.86 |
| HNSW `ef=10` | 0.820 | 0.565 | 0.545 | 0.756 | 1770.54 |
| HNSW `ef=20` | 0.910 | 0.656 | 0.621 | 0.822 | 1524.45 |
| HNSW `ef=40` | 0.980 | 0.808 | 0.785 | 1.049 | 1237.50 |
| HNSW `ef=80` | 1.000 | 1.073 | 1.078 | 1.366 | 932.07 |
| HNSW `ef=160` | 1.000 | 1.516 | 1.416 | 2.048 | 659.81 |

### 10K Documents — Exact vs IVFFlat

| Method | Recall@10 | Mean (ms) | P50 (ms) | P95 (ms) | QPS |
|---|---:|---:|---:|---:|---:|
| Exact | 1.000 | 3.233 | 3.106 | 4.154 | 309.33 |
| IVFFlat `probes=1` | 0.410 | 0.419 | 0.406 | 0.529 | 2383.99 |
| IVFFlat `probes=2` | 0.560 | 0.443 | 0.416 | 0.547 | 2259.06 |
| IVFFlat `probes=5` | 0.740 | 0.452 | 0.445 | 0.581 | 2214.38 |
| IVFFlat `probes=10` | 0.850 | 0.633 | 0.627 | 0.758 | 1580.53 |
| IVFFlat `probes=20` | 0.950 | 0.937 | 0.903 | 1.313 | 1067.26 |
| IVFFlat `probes=50` | 1.000 | 1.834 | 1.841 | 2.218 | 545.31 |

On this workload, HNSW provided the stronger high-recall latency trade-off, while IVFFlat offered different build/storage characteristics.

## 100K Scalability Experiment

### Ingestion

| Metric | Result |
|---|---:|
| Documents | 100,000 |
| Embedding dimension | 384 |
| Embedding time | 47.22 s |
| Embedding throughput | 2,117.62 docs/s |
| Insert time | 32.04 s |
| Insert throughput | 3,121.44 rows/s |
| Heap size | 87 MB |

### Exact Baseline

An independent exact-search run produced:

| Metric | Result |
|---|---:|
| Mean | 77.657 ms |
| P50 | 77.357 ms |
| P95 | 83.813 ms |
| P99 | 85.045 ms |
| Approx. QPS | 12.88 |

`EXPLAIN ANALYZE` verified sequential scanning for the exact baseline.

For direct Exact-vs-HNSW speed comparisons below, the exact result from the **same benchmark run** is used to reduce differences caused by query sets and system state.

### Exact vs HNSW at 100K

| Method | Recall@10 | Mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) | QPS |
|---|---:|---:|---:|---:|---:|---:|
| Exact | 1.000 | 66.280 | 65.117 | 75.072 | 77.879 | 15.09 |
| HNSW `ef=10` | 0.942 | 0.948 | 0.890 | 1.429 | 1.628 | 1054.70 |
| HNSW `ef=20` | 0.965 | 1.090 | 1.062 | 1.742 | 1.898 | 917.83 |
| HNSW `ef=40` | 0.992 | 1.387 | 1.405 | 1.978 | 2.449 | 720.74 |
| HNSW `ef=80` | 0.994 | 2.039 | 1.946 | 3.246 | 3.489 | 490.52 |
| HNSW `ef=160` | 0.997 | 3.047 | 2.973 | 4.655 | 5.337 | 328.24 |

At `ef_search=40`:

- Recall@10: **0.992**
- P50 latency: **1.405 ms**
- Same-run exact P50: **65.117 ms**
- P50 speedup: approximately **46.3×**

Increasing `ef_search` from 80 to 160 improved Recall@10 from 0.994 to 0.997 while P50 latency increased from 1.946 ms to 2.973 ms, showing diminishing returns at higher search effort.

## Scalability Observation

Corpus size increased 10× from 10K to 100K.

Using comparable benchmark results:

- Exact P50: approximately 3.219 ms → 65.117 ms
- HNSW (`ef=40`) P50: approximately 0.785 ms → 1.405 ms

This illustrates the different scaling behavior of brute-force exact search and graph-based ANN retrieval on the tested workload.

## HNSW Index Build Tuning

The initial 100K HNSW build used PostgreSQL's default:

```text
maintenance_work_mem = 64MB
```

PostgreSQL reported that the HNSW graph stopped fitting in `maintenance_work_mem` after approximately 28K tuples.

| Configuration | Build Time | HNSW Size |
|---|---:|---:|
| `maintenance_work_mem = 64MB` | 37.848 s | 195 MB |
| `maintenance_work_mem = 1GB` | 12.038 s | 195 MB |

The Docker container's shared memory was increased to:

```yaml
shm_size: '2gb'
```

and the tuned build used:

```sql
SET maintenance_work_mem = '1GB';
```

This reduced HNSW build time by approximately **68.2%**, corresponding to a **3.14× speedup**, while final index size remained unchanged.

The experiment highlights the distinction between Docker memory, POSIX shared memory (`/dev/shm`), PostgreSQL `maintenance_work_mem`, `shared_buffers`, and `work_mem`.

## Project Structure

```text
.
├── backend/
│   ├── benchmark_exact.py
│   ├── benchmark_hnsw.py
│   ├── benchmark_ivfflat.py
│   ├── benchmark_scalability_exact.py
│   ├── benchmark_scalability_hnsw.py
│   ├── hybrid_search.py
│   ├── ingest_benchmark.py
│   ├── ingest_scalability.py
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
│   └── 08_scalability_hnsw_tuned.sql
├── .env.example
├── .gitignore
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Setup

### 1. Clone

```bash
git clone <repository-url>
cd hybrid-vector-retrieval
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and replace the example password with a local development password.

```dotenv
POSTGRES_DB=retrieval_db
POSTGRES_USER=retrieval
POSTGRES_PASSWORD=change_me
POSTGRES_PORT=5433
DATABASE_URL=postgresql://retrieval:change_me@127.0.0.1:5433/retrieval_db
```

Never commit the real `.env`.

### 3. Start PostgreSQL + pgvector

```bash
docker compose up -d
```

Verify:

```bash
docker exec hybrid-retrieval-db pg_isready -U retrieval -d retrieval_db
```

### 4. Create Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 5. Load Environment Variables

```bash
set -a
source .env
set +a
```

### 6. Initialize Schema

```bash
docker exec -i hybrid-retrieval-db psql -U retrieval -d retrieval_db < database/01_schema.sql
```

### 7. Run Semantic Search Demo

```bash
python backend/seed_documents.py
python backend/search.py
```

## Scripts

| Script | Purpose |
|---|---|
| `benchmark_exact.py` | Exact vector-search baseline |
| `benchmark_hnsw.py` | HNSW recall/latency evaluation |
| `benchmark_ivfflat.py` | IVFFlat recall/latency evaluation |
| `benchmark_scalability_exact.py` | Exact scalability baseline |
| `benchmark_scalability_hnsw.py` | HNSW scalability evaluation |
| `hybrid_search.py` | Full-text + dense retrieval + RRF |
| `ingest_benchmark.py` | Benchmark corpus ingestion |
| `ingest_scalability.py` | Larger-scale corpus ingestion |

## SQL Files

| File | Purpose |
|---|---|
| `01_schema.sql` | Primary relational schema |
| `02_benchmark.sql` | Benchmark table |
| `03_hnsw.sql` | HNSW index |
| `04_ivfflat.sql` | IVFFlat index |
| `05_fulltext.sql` | Full-text search and GIN index |
| `06_scalability.sql` | Scalability schema |
| `07_scalability_hnsw.sql` | Scalability HNSW index |
| `08_scalability_hnsw_tuned.sql` | Tuned HNSW build |

## Current Progress

- [x] PostgreSQL 17 + pgvector
- [x] Docker Compose environment
- [x] Relational schema
- [x] Dense embedding pipeline
- [x] Exact vector retrieval
- [x] HNSW ANN retrieval
- [x] IVFFlat ANN retrieval
- [x] PostgreSQL full-text search
- [x] GIN indexing
- [x] Hybrid retrieval with RRF
- [x] 10K benchmark
- [x] 100K scalability benchmark
- [x] ANN recall/latency evaluation
- [x] HNSW build-memory tuning
- [ ] 500K scalability benchmark
- [ ] 1M scalability benchmark
- [ ] Scalability plots
- [ ] REST API
- [ ] Web interface
- [ ] Evaluation dashboard
- [ ] ER diagram
- [ ] Final report and demo

## Planned Evaluation

Future scalability stages will evaluate larger corpora using bounded-memory ingestion.

Planned measurements:

- Exact-search latency
- HNSW Recall@10
- P50/P95/P99 latency
- QPS
- Index build time
- Index size
- Database storage growth
- Recall/latency trade-offs
- Corpus-size scaling curves

Future relevance evaluation may additionally use real queries and relevance judgments to report MRR@10, nDCG@10, and Recall@10. These metrics will be kept conceptually separate from ANN recall against exact vector search.

## Limitations

- ANN Recall@10 measures agreement with exact vector retrieval, not human relevance.
- Small query sets can overestimate generalization of an ANN configuration.
- Latency depends on hardware, cache state, PostgreSQL configuration, and workload.
- Embedding-model inference is intentionally excluded from database retrieval latency.
- Scalability experiments currently reach 100K documents; larger-scale experiments are in progress.

## Roadmap

```text
500K corpus
    ↓
1M corpus
    ↓
scalability analysis
    ↓
REST API
    ↓
web interface / dashboard
    ↓
ER diagram
    ↓
final report and live demo
```

## Author

**Ravan Chuang**

Computer Science · Information Retrieval · Backend & Systems Engineering
