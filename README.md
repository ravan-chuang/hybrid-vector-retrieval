# Hybrid Vector Retrieval System

**Full-corpus information retrieval · PostgreSQL / pgvector · C++ BM25 · Cross-Encoder serving**

A multi-stage search system combining relational data management, lexical and dense retrieval, approximate nearest-neighbor (ANN) indexing, hybrid candidate generation, Cross-Encoder reranking, and bounded **cross-request dynamic batching**. The project emphasizes measurable retrieval quality, database systems engineering, and reproducible performance analysis.

> **Project status — v0.7.7 (October 2026):** The database, full MS MARCO corpus, retrieval pipelines, relevance evaluation, FastAPI backend, relational CRUD, and React/Vite dashboard are implemented. C++ BM25 serving, batched Cross-Encoder inference, concurrency/parity tests, and real-document profiling have also been evaluated. This is an **engineering/research prototype**, not a production-ready hosted search service.

![Project overview](docs/system-overview.jpeg)

## At a glance

| Dimension | Measured result / implementation |
| --- | --- |
| Indexed corpus | **8,841,823 MS MARCO passages** |
| Dense representation | **384-dimensional**, normalized `all-MiniLM-L6-v2` embeddings |
| Vector index | PostgreSQL 17 + pgvector **HNSW**; IVFFlat evaluated separately |
| Lexical retrieval | PostgreSQL FTS/GIN; **C++ BM25** available as a serving backend |
| Best evaluated relevance pipeline | Lexical Top-50 ∪ Dense Top-50 → Cross-Encoder → Top-10 |
| Full-corpus relevance, 10K queries | **MRR@10 0.4166 · nDCG@10 0.4876 · Recall@10 0.7211** |
| Full-corpus ANN, `ef_search=40` | **0.858 ANN Recall@10 · 31.9 ms P50** (exact-search P50: 5751.5 ms) |
| Dynamic batching, C16 | **10.895 vs 8.360 QPS** (batched vs baseline; one formal 1000-request run, configuration B) |
| Batched ranking parity | **10/10** queries: identical Top-1, Top-10 membership, and Top-10 order |
| Real reranker input profile | **914 query–passage pairs**; mean 78.60 tokens; P95 138; 0% truncated |

**Important:** The relevance numbers above use an independent **10,000-query sample from the MS MARCO `labeled-list` training split**, *not* the official MS MARCO dev leaderboard. ANN Recall measures agreement with exact nearest neighbors, not human relevance. Concurrent HTTP QPS and database-only sequential reciprocal-latency estimates are different metrics and must not be compared directly.

## Contents

- [Architecture](#architecture)
- [Features and technology](#features-and-technology)
- [Database design](#database-design)
- [Retrieval pipelines](#retrieval-pipelines)
- [Evaluation protocol](#evaluation-protocol)
- [Retrieval quality](#retrieval-quality)
- [ANN and scalability](#ann-and-scalability)
- [Dynamic batching and serving](#dynamic-batching-and-serving)
- [Cross-Encoder profiling](#cross-encoder-profiling)
- [Getting started](#getting-started)
- [API examples](#api-examples)
- [Repository guide](#repository-guide)
- [Testing and reproducibility](#testing-and-reproducibility)
- [Limitations and roadmap](#limitations-and-roadmap)

## Architecture

![System architecture](docs/system-architecture.jpeg)

```text
                         React / Vite dashboard
                                  │ REST / JSON
                                  ▼
                              FastAPI
                                  │
                  ┌───────────────┴────────────────┐
                  │                                │
          Relational application             Search pipeline
          documents / tags / history               │
                  │                         Query embedding
                  │                                │
                  │              ┌─────────────────┴──────────────────┐
                  │              ▼                                    ▼
                  │       Lexical retrieval                      Dense retrieval
                  │       PostgreSQL FTS / GIN                  pgvector HNSW
                  │       or C++ BM25                           cosine distance
                  │              │                                    │
                  │              └─────────────────┬──────────────────┘
                  │                                ▼
                  │                       Candidate union / fusion
                  │                                │
                  │                   ┌────────────┴─────────────┐
                  │                   ▼                          ▼
                  │              Standard CE              Batched CE worker
                  │                   │                    bounded queue +
                  │                   │                    cross-request batching
                  │                   └────────────┬─────────────┘
                  │                                ▼
                  │                          Ranked Top-K
                  ▼
          PostgreSQL 17 + pgvector
```

The normalized application schema and the large benchmark/serving corpus are logically separated. Application CRUD, referential integrity, and search history remain database-backed. The full MS MARCO corpus and its HNSW/GIN indexes support large-scale retrieval experiments. The Cross-Encoder is `cross-encoder/ms-marco-MiniLM-L-6-v2`.

## Features and technology

| Layer | Components |
| --- | --- |
| Database | PostgreSQL 17, pgvector, B-tree, GIN, HNSW, experimental IVFFlat |
| Backend | Python, FastAPI, Uvicorn, psycopg 3, `psycopg_pool` |
| Dense retrieval | `sentence-transformers/all-MiniLM-L6-v2` (384D, normalized) |
| Lexical retrieval | PostgreSQL `tsvector` / `websearch_to_tsquery`; C++ BM25 |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2`, PyTorch CPU/MPS |
| Serving | Per-process bounded dynamic batch worker, queue/timeout controls, single-request baseline |
| Frontend | React 19, Vite 8, CSS |
| Infrastructure | Docker Compose, local Python virtual environment |
| Evaluation | MS MARCO relevance judgments, ANN exact-neighbor comparisons, HTTP concurrency tests, paired bootstrap |

### Supported capabilities

- Exact cosine vector retrieval and approximate HNSW retrieval.
- PostgreSQL full-text search and C++ BM25 lexical retrieval.
- Hybrid RRF and normalized-score fusion baselines; lexical/dense candidate union.
- Cross-Encoder reranking with both standard and batched API paths.
- Search-history persistence and document/tag CRUD with foreign-key constraints.
- Dashboard for retrieval, statistics, and relational database operations.
- Index-build, ingestion, relevance, latency, throughput, and ranking-parity experiments.

## Database design

![Entity-relationship diagram](docs/database-erd.jpg)

| Table | Role |
| --- | --- |
| `users` | Application users |
| `documents` | Document metadata |
| `document_chunks` | Searchable chunks, embeddings, generated full-text representation |
| `tags` | Unique tag definitions |
| `document_tags` | Document–tag many-to-many association |
| `search_history` | Query type, result count, timing, and history |

The normalized application schema enforces primary/foreign keys, uniqueness, checks, and cascade rules. Deleting a document cascades to its chunks and tag associations without deleting reusable tag records. Application chunks have full-text and vector indexes. Large-scale benchmark tables are maintained separately from the application workload.

**Representative vector query** (illustrative schema; adapt to the actual table and connection parameters):

```sql
SELECT document_id, content, embedding <=> %s::vector AS distance
FROM benchmark_documents
ORDER BY embedding <=> %s::vector
LIMIT 10;
```

**Representative HNSW index:**

```sql
CREATE INDEX idx_benchmark_embedding_hnsw
ON benchmark_documents
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

`ef_search` is varied at query time to study ANN recall–latency trade-offs. A GIN index accelerates PostgreSQL full-text queries. Full-corpus HNSW and GIN indexes are substantial local assets; they are **not included in this Git repository**.

## Retrieval pipelines

### First-stage retrieval

1. **Lexical:** PostgreSQL FTS/GIN or the local C++ BM25 backend.
2. **Dense:** Encode the query with normalized MiniLM embeddings; retrieve cosine nearest neighbors through pgvector HNSW.
3. **Fusion:** Compare reciprocal rank fusion (RRF), weighted RRF, and normalized-score fusion; alternatively take the lexical/dense union for reranking.

For two ranked lists, RRF is:

\[
\mathrm{RRF}(d)=\sum_{r\in\mathcal R}\frac{1}{k+\operatorname{rank}_r(d)}
\]

The evaluated weighted-RRF baseline uses `k=60`, lexical weight `0.1`, and dense weight `0.9`.

### Multi-stage reranking

```text
Query → lexical Top-50 + dense Top-50 → deduplicated union
      → Cross-Encoder(query, passage) → final Top-10
```

The **primary 10K-query relevance study** used **PostgreSQL FTS** as its lexical source. More recent live serving and concurrency tests use **C++ BM25**. These are different retrieval configurations; do not attribute the 10K relevance results to the C++ BM25 serving backend without a corresponding evaluation.

## Evaluation protocol

- **ANN Recall@K:** overlap between approximate and exact vector nearest neighbors; not a relevance metric.
- **MRR@10, nDCG@10, relevance Recall@10:** computed using binary positive judgments from the MS MARCO `labeled-list` training split.
- **Corpus:** 8,841,823 passages; all 408,158 unique positively judged passages in this judgment source are represented locally. This is corpus coverage, **not exhaustive human judgment coverage**.
- **Primary relevance sample:** independent 10,000-query full-corpus sample, seed `43`, disjoint from an earlier frozen 500-query sample.
- **ANN setting for relevance study:** HNSW `ef_search=160` to reduce approximation effects.
- **Uncertainty:** paired bootstrap, 10,000 resamples for reported pipeline deltas.
- **Serving:** HTTP closed-loop concurrency tests; client latency, server latency, queue wait, batch execution, and errors recorded separately.

## Retrieval quality

### Primary full-corpus evaluation — 10,000 queries

| Pipeline | MRR@10 | nDCG@10 | Relevance Recall@10 |
| --- | ---: | ---: | ---: |
| Lexical (PostgreSQL FTS) | 0.1633 | 0.1940 | 0.2982 |
| Dense HNSW (`ef=160`) | 0.3101 | 0.3741 | 0.5879 |
| Weighted RRF (0.1 lexical / 0.9 dense) | 0.3219 | 0.3863 | 0.6010 |
| Dense Top-50 → Cross-Encoder | 0.4022 | 0.4703 | 0.6950 |
| **Lexical ∪ Dense Top-50 → Cross-Encoder** | **0.4166** | **0.4876** | **0.7211** |

**Union CE versus Dense CE:** ΔMRR@10 **+0.0145** (95% CI `[+0.0121, +0.0170]`); ΔnDCG@10 **+0.0173** (95% CI `[+0.0147, +0.0199]`); ΔRecall@10 **+0.0261** (95% CI `[+0.0223, +0.0299]`).

Lexical retrieval contributed **485 relevant passages absent from Dense Top-50**; the Cross-Encoder promoted **339/485 (69.9%)** into the final union Top-10. This is evidence for complementary candidate generation, **not a claim of a novel retrieval algorithm**.

Earlier 500-query results are supporting evidence and should not be mixed with the independent 10K protocol.

## ANN and scalability

### Full 8.84M-passage corpus

| Method | ANN Recall@10 | Mean (ms) | P50 (ms) | P95 (ms) | Approx. sequential QPS* |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exact | 1.000 | 5781.9 | 5751.5 | 6949.7 | 0.17 |
| HNSW `ef=10` | 0.692 | 49.5 | 45.7 | 79.8 | 20.19 |
| HNSW `ef=20` | 0.780 | 22.0 | 17.5 | 52.9 | 45.52 |
| HNSW `ef=40` | 0.858 | 38.4 | 31.9 | 92.5 | 26.04 |
| HNSW `ef=80` | 0.905 | 61.6 | 57.8 | 118.3 | 16.23 |
| HNSW `ef=160` | 0.939 | 102.6 | 95.6 | 169.6 | 9.74 |

\* `1000 / mean_latency_ms` for sequential database-only measurements; **not** concurrent API throughput. The non-monotonic `ef=10`/`ef=20` latency is a measured anomaly, not an expected property of HNSW.

**Storage observations:** full database approximately **39 GB**; HNSW index approximately **17 GB**; full-corpus GIN index approximately **904 MB**. HNSW build used `m=16`, `ef_construction=64`; Docker shared memory was increased to 8 GB. The exact full-corpus HNSW build duration was not recorded.

### Smaller-scale experiments

| Corpus | Exact P50 | HNSW `ef=40` P50 | HNSW ANN Recall@10 |
| --- | ---: | ---: | ---: |
| 10K | 3.219 ms | 0.785 ms | 0.980 |
| 100K | 65.117 ms | 1.405 ms | 0.992 |
| 500K | 102.918 ms | 2.215 ms | 0.981 |

The corpus/query conditions differ across scale experiments; these are engineering observations, **not** a strictly controlled scaling-law experiment. At 100K, increasing `maintenance_work_mem` from 64 MB to 1 GB reduced HNSW build time from **37.848 s** to **12.038 s** (approximately **3.14×** faster), with unchanged index size.

## Dynamic batching and serving

The standard endpoint reranks each request independently. The batched endpoint submits requests to a **process-local bounded queue**, where a worker aggregates query–candidate pairs and calls the same Cross-Encoder. Results are split back into the originating requests. Controls include maximum requests, maximum pairs, queue wait window, queue capacity, and request timeout.

**Current tested configuration B** (local Apple Silicon MPS):

```bash
CROSS_ENCODER_DEVICE=mps
CROSS_ENCODER_BATCH_SIZE=128
RERANK_MAX_BATCH_REQUESTS=8
RERANK_MAX_BATCH_PAIRS=1024
RERANK_MAX_BATCH_WAIT_MS=5
```

### Formal online concurrency test — configuration B

100 requests per concurrency level **per pipeline** (1,000 total measured requests); both endpoints returned successfully in this run.

| Concurrent clients | Baseline QPS | Batched QPS | Baseline mean client (ms) | Batched mean client (ms) | Baseline P95 (ms) | Batched P95 (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8.228 | 8.213 | 121.49 | 121.71 | 152.72 | 147.43 |
| 2 | 9.930 | 9.314 | 200.23 | 214.68 | 239.68 | 263.01 |
| 4 | 9.381 | 10.177 | 418.96 | 392.94 | 485.79 | 450.33 |
| 8 | 9.296 | 10.238 | 829.16 | 758.91 | 898.39 | 865.39 |
| 16 | 8.360 | **10.895** | 1770.23 | **1401.22** | 2078.05 | **1836.62** |

At C16, batched throughput was **30.3% higher than baseline within this run**. At C1–C2 batching provided no throughput benefit. The benchmark uses a closed-loop workload and a fixed baseline-before-batched execution order; this result **does not establish a general 30.3% improvement**. Randomized order, repeated runs, and open-loop load tests remain future work.

### Ranking parity

A separate 10-query HTTP comparison observed **10/10 Top-1, 10/10 Top-10 set, and 10/10 exact Top-10 order agreement**, with no request errors; eight of ten batched requests participated in cross-request batches. This is a useful regression check, not a large-scale parity guarantee. Retrieval was independently executed for the two endpoints rather than using frozen identical candidate sets.

### Metric interpretation

`batch_queue_wait_ms` includes waiting behind earlier batches. `batch_inference_ms` is the **entire worker batch-processing interval** (including pair construction, model access, inference-lock wait, prediction, and MPS synchronization), not GPU-only forward time. Batch-level timing is copied into each member response; summing those values over requests double-counts batch execution. The batched endpoint's `rerank_lock_wait_ms=0.0` is a placeholder and **must not be interpreted as measured zero lock contention**.

## Cross-Encoder profiling

The v0.7.7 studies used the same Cross-Encoder on Apple Silicon MPS with internal `batch_size=128`.

### Synthetic versus real input

| Pairs | Synthetic predict (ms) | Real-document predict (ms) | Real / synthetic |
| ---: | ---: | ---: | ---: |
| 64 | 25.72 | 61.10 | 2.38× |
| 128 | 45.00 | 130.30 | 2.90× |
| 256 | 78.02 | 203.94 | 2.61× |
| 384 | 119.97 | 274.26 | 2.29× |
| 512 | 151.66 | 369.69 | 2.44× |
| 768 | 230.04 | 557.80 | 2.42× |

The synthetic pairs were short (approximately **20 tokens**). The real workload contained **914 pairs from 10 queries**, with mean **78.60**, P50 **71**, P95 **138**, maximum **254**, and **0%** exceeding the model's 512-token limit. These experiments were run separately and are **not** a fully controlled causal comparison.

### Real-document stage breakdown

| Pairs | Total (ms) | Tokenization (ms) | MPS forward (ms) | Padding waste |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 111.92 | 15.70 | 93.64 | 58.2% |
| 256 | 216.30 | 27.33 | 184.40 | 55.9% |
| 384 | 317.11 | 39.50 | 270.13 | 55.3% |
| 512 | 402.44 | 50.39 | 341.49 | 54.0% |
| 768 | 629.03 | 78.36 | 533.53 | 51.8% |

This manual breakdown measures a **different execution path** from `CrossEncoder.predict()` and should not be subtracted directly from its total. At 768 pairs, model forward accounted for approximately **84.8% of this manual profile's total**.

### Exploratory length-aware input ordering

| Pairs | Original predict (ms) | Externally length-sorted (ms) | Estimated padding waste (original → sorted) |
| ---: | ---: | ---: | ---: |
| 128 | 130.84 | 130.91 | 58.2% → 58.2% |
| 256 | 204.91 | 205.27 | 55.9% → 40.2% |
| 384 | 273.72 | 274.33 | 55.3% → 29.3% |
| 512 | 382.86 | 369.99 | 54.0% → 24.1% |
| 768 | 558.01 | 559.49 | 51.8% → 15.1% |

The reported score differences were **0.000000 at six-decimal display precision**. External sorting did **not** produce a consistent end-to-end speedup. **Interpretation caveat:** the padding estimate models input ordering in 128-item chunks; `CrossEncoder.predict()` may already perform internal length sorting, depending on the installed version. Verify its implementation before attributing the measured timing to actual padding reduction. Length-aware scheduling has **not** been deployed online.

## Getting started

> **Resource warning:** The full 8.84M corpus, embeddings, HNSW graph, and C++ BM25 index require substantial disk/RAM and lengthy preparation. The commands below initialize the **application environment**; they do not automatically download or build the full benchmark corpus. Avoid rebuilding indexes or starting multiple API workers when using a prebuilt local index.

### 1. Clone and configure

```bash
git clone https://github.com/ravan-chuang/hybrid-vector-retrieval.git
cd hybrid-vector-retrieval
cp .env.example .env
```

Edit `.env` to match your Docker port, credentials, and local device. Example values from the original local setup:

```dotenv
POSTGRES_DB=retrieval_db
POSTGRES_USER=retrieval
POSTGRES_PASSWORD=change_me
POSTGRES_PORT=5433
DATABASE_URL=postgresql://retrieval:change_me@127.0.0.1:5433/retrieval_db
CROSS_ENCODER_DEVICE=mps
CROSS_ENCODER_BATCH_SIZE=128
```

**These are examples, not guaranteed current defaults.** Match `DATABASE_URL` to the active Compose configuration. Use `CROSS_ENCODER_DEVICE=cpu` on machines without compatible MPS support. Never commit `.env` or credentials.

### 2. Start PostgreSQL

```bash
docker compose up -d
docker compose ps
```

The original Compose service used a container named `hybrid-retrieval-db`; verify the name with `docker compose ps` before using `docker exec`.

### 3. Install backend dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
set -a
source .env
set +a
```

### 4. Initialize the application schema (new databases only)

```bash
docker exec -i hybrid-retrieval-db \
  psql -U retrieval -d retrieval_db < database/01_schema.sql
```

**Do not rerun schema/index scripts blindly against an existing populated database.** Review migration and index creation statements first. The full-corpus ingestion and indexing pipeline is separate; see the scripts in `backend/scripts/` and SQL in `database/`.

### 5. Start the API

```bash
python -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Open `http://127.0.0.1:8000/docs` for OpenAPI. On a machine with the full C++ BM25 index, startup may take significant time; keep the server running while executing benchmarks from another terminal. The batch worker is **process-local**; adding Uvicorn workers changes batching behavior and is not equivalent to the measured one-worker configuration.

For the **measured configuration B**, set the following environment variables before launching Uvicorn:

```bash
RERANK_MAX_BATCH_REQUESTS=8 \
RERANK_MAX_BATCH_PAIRS=1024 \
RERANK_MAX_BATCH_WAIT_MS=5 \
python -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

### 6. Start the dashboard

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The dashboard covers retrieval and relational operations; newer C++ BM25/batched endpoints are API capabilities and are **not asserted to have dedicated dashboard controls**.

## API examples

The main retrieval endpoints include:

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Health and database connectivity |
| `GET` | `/stats` | Database, corpus, model, and index statistics |
| `POST` | `/search/vector` | Dense HNSW retrieval |
| `POST` | `/search/lexical` | Lexical retrieval |
| `POST` | `/search/hybrid` | Hybrid fusion retrieval |
| `POST` | `/search/rerank` | Standard Cross-Encoder reranking |
| `POST` | `/search/rerank/batched` | Cross-request dynamic batching |
| `GET` | `/search/history` | Persisted search history |
| `GET/POST` | `/documents` | List/create application documents |
| `GET/PUT/DELETE` | `/documents/{document_id}` | Read/update/delete a document |
| `GET/POST` | `/documents/{document_id}/tags` | List/attach tags |
| `DELETE` | `/documents/{document_id}/tags/{tag_id}` | Detach a tag |

**Live benchmark rerank example** (requires the prebuilt benchmark corpus/index):

```bash
curl -sS -X POST http://127.0.0.1:8000/search/rerank/batched \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "what is information retrieval",
    "corpus": "benchmark",
    "lexical_backend": "cpp_bm25",
    "stopword_filter": true,
    "top_k": 10,
    "candidate_k": 50,
    "ef_search": 160
  }'
```

To compare the standard path, change the URL to `/search/rerank` and keep the request body identical. The repository also contains a **localhost-restricted, undocumented** `/internal/benchmark/candidates` endpoint for exporting the full candidate union in local tests; it is not intended as a public interface.

## Repository guide

```text
backend/
  api/
    routers/          # search, rerank, documents, tags, system
    schemas/          # request and response validation
    services/         # embedding, Cross-Encoder, dynamic batching
    main.py           # FastAPI application and lifespan
  scripts/            # ingestion, evaluation, profiling, benchmarks
  benchmark_*.py      # smaller-scale database benchmarks
database/             # relational schema, corpus, FTS and ANN SQL
frontend/             # React / Vite dashboard
artifacts/            # compact versioned experiment summaries
results/              # larger local benchmark outputs (may be Git-ignored)
docs/                 # architecture and ERD figures
requirements.txt
.env.example
docker-compose.yml
pytest.ini
```

### Selected experiment scripts

| Script | Purpose |
| --- | --- |
| `backend/scripts/benchmark_dynamic_batch_v075.py` | Closed-loop HTTP concurrency benchmark |
| `backend/scripts/benchmark_ranking_parity_v075.py` | Standard vs batched Top-K agreement |
| `backend/scripts/profile_cross_encoder_v077.py` | Synthetic Cross-Encoder profile |
| `backend/scripts/analyze_real_tokens_v077.py` | Real candidate token-length distribution |
| `backend/scripts/profile_real_cross_encoder_v077.py` | Real-pair `predict()` throughput |
| `backend/scripts/profile_real_breakdown_v077.py` | Manual tokenization/transfer/forward breakdown |
| `backend/scripts/benchmark_length_aware_v077.py` | Exploratory external length-ordering comparison |
| `backend/scripts/benchmark_msmarco_full_ann.py` | Full-corpus ANN benchmark |
| `backend/scripts/evaluate_msmarco_full_retrieval_10k.py` | Primary full-corpus retrieval evaluation |
| `backend/scripts/rerank_msmarco_full_ce_10k.py` | Primary Cross-Encoder relevance evaluation |

Some experimental scripts and `results/` files may exist only in the local development checkout. Check `git status` and the tracked repository before assuming every listed artifact is available after cloning.

## Testing and reproducibility

```bash
# From repository root, with the virtual environment active
python -m pytest -q
```

A development run of the v0.7.5 suite reported **52 passing tests** (with one warning). This is a historical result, **not a claim that the latest checkout has been retested**. Integration tests may require a configured PostgreSQL instance and local corpus assets.

For the formal dynamic-batching experiment, the development command was:

```bash
python -m backend.scripts.benchmark_dynamic_batch_v075 \
  --requests 100 \
  --concurrency 1 2 4 8 16 \
  --timeout 30 \
  --output results/msmarco_full/latency/api_v076/config_B_formal_1000
```

Selected local result paths:

```text
results/msmarco_full/latency/api_v076/config_B_formal_1000/
  summary.json
  per_request.json
results/msmarco_full/parity/v075/
  summary.json
  comparisons.json
results/msmarco_full/latency/api_v077/
  real_token_summary.json
  real_cross_encoder_profile.json
  real_breakdown_profile.json
  length_aware_profile.json
```

For meaningful performance comparisons, record the exact commit, hardware, Python/PyTorch/`sentence-transformers` versions, PostgreSQL settings, index configuration, warm-up policy, request distribution, and raw per-run results. Prefer randomized or alternating baseline/treatment order with multiple runs; do not mix **database-only latency**, **model-only latency**, and **end-to-end API latency**.

## Key engineering findings

1. **Candidate complementarity matters.** Lexical-only relevant passages recovered by candidate union improved the primary 10K-query Cross-Encoder result over dense-only reranking.
2. **Index search effort has a measurable cost.** Full-corpus HNSW `ef_search` trades ANN recall against latency; exact retrieval is much more expensive at this scale.
3. **Serving optimizations depend on concurrency.** Dynamic batching helped under higher tested concurrency but was neutral or worse at low concurrency.
4. **Synthetic inference inputs can mislead.** Short synthetic pairs substantially overstated throughput relative to sampled real MS MARCO candidates.
5. **Padding statistics are not a speedup guarantee.** External length sorting reduced *estimated* padding but did not consistently reduce `CrossEncoder.predict()` time; internal sorting must be checked before drawing causal conclusions.
6. **Database engineering is part of retrieval engineering.** Schema integrity, HNSW build memory, index footprint, connection management, and physical query plans affect operational behavior.

## Limitations and roadmap

### Current limitations

- The primary relevance labels come from MS MARCO **training** judgments and are incomplete; they are not official dev/leaderboard results.
- The primary 10K relevance evaluation used PostgreSQL lexical retrieval, whereas newer live serving tests use C++ BM25; quality transfer between those configurations has not been established.
- The online concurrency results are a single fixed-order closed-loop benchmark; statistical repeatability and open-loop load behavior remain to be established.
- Batched parity has been tested on 10 queries, not the full relevance set; request timeout/cancellation, queue saturation, and shutdown paths require deeper stress testing.
- The length-aware comparison may be confounded by internal `CrossEncoder.predict()` sorting.
- Full-corpus assets are large and not distributed through Git; a fresh clone does not reproduce the full-scale deployment automatically.
- Authentication, SLOs, replication/failover, backups, comprehensive observability, and sustained production load testing are outside the current prototype scope.

### Planned work (not implemented)

| Priority | Next step | Success criterion |
| --- | --- | --- |
| P0 | Inspect `CrossEncoder.predict()` sorting and sweep internal batch size | Reproducible throughput/latency comparison on identical real pairs |
| P0 | Repeat and randomize HTTP A/B benchmarks | Stable QPS/P95/error estimates across multiple runs |
| P0 | Evaluate **C++ BM25** retrieval quality | Matched-query comparison with PostgreSQL FTS and dense retrieval |
| P1 | Profile SQL/pgvector and C++ BM25 separately | Per-stage latency, physical plans, and resource utilization |
| P1 | Candidate-depth and fusion ablations | nDCG/MRR/Recall versus inference cost |
| P1 | Queue saturation, cancellation, and lifecycle tests | Correctness under failure and overload |
| P2 | Adaptive candidate selection / selective reranking | Better quality–cost frontier than fixed policies |
| P2 | Polish reproducibility, dashboard/demo, and technical report | Documented v1.0 release |

## Demo outline

1. Show the ERD, PK/FK constraints, many-to-many tags, and CRUD cascade behavior.
2. Compare PostgreSQL FTS, C++ BM25, HNSW dense retrieval, and hybrid results.
3. Explain the full-corpus exact-vs-HNSW recall–latency trade-off.
4. Present the independent 10K-query relevance table and lexical-rescue analysis.
5. Compare standard and batched reranking under controlled concurrency, including low-concurrency regressions.
6. Explain the real-document profiling findings and what they do **not** prove.

## Author

**Ravan Chuang**
Computer Science · Information Retrieval · Backend & Systems Engineering

## License

The original project README identifies the project as MIT-licensed; confirm the `LICENSE` file is present in the distributed repository. Third-party datasets, model weights, and dependencies retain their respective terms and licenses.
