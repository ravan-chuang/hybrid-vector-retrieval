# Cross-Encoder Candidate Budgets

On 120 fixed dev-small queries over 8,841,823 passages, filtering the two-way candidate union reduced token count and request latency. All three capped strategies had paired 95% interval lower bounds below the provisional −0.005 nDCG@10 margin relative to the union. The standard API continues to use the full union.

| Strategy | Mean pairs | Mean tokens | HTTP P95 | nDCG@10 | Paired difference vs. union (95% CI) |
| --- | ---: | ---: | ---: | ---: | --- |
| RRF40 | 40.0 | 3,329 | 560 ms | 0.419643 | −0.021865 [−0.049930, +0.002697] |
| Dense30 + BM25-only 10 | 40.0 | 3,323 | 600 ms | 0.428140 | −0.013369 [−0.041028, +0.011369] |
| RRF50 | 50.0 | 4,168 | 611 ms | 0.439258 | −0.002251 [−0.017766, +0.010990] |
| Union | 89.8 | 7,542 | 749 ms | 0.441508 | Reference |

RRF40 and 30+10 each used about 56% fewer tokens than the union. RRF50 used about 45% fewer tokens. Measured nDCG@10 and token counts for the three shared strategies matched the original three-way pilot on all 360 query–strategy pairs. See the [480 numerical observations](../../artifacts/relevance/candidate_budgets/observations.csv). Timings apply to this replay and its conditions.

## Protocol

The 120 measured queries match the original candidate-budget pilot. Twelve additional queries warm up the service. [Sample manifest](../../artifacts/relevance/candidate_budgets/sample.json) records their order as zero-based positions in the [dev-small manifest](../../artifacts/msmarco_full/dev_small_manifest.json), with a SHA-256 check. The pilot sampled unique judged query texts with seed 5013 from the hash-based test half, excluding texts used in the HNSW sweep.

Each measured query uses all four strategies, with the 24 possible execution orders repeated five times. Requests run sequentially on one local API process. Each calls the existing live retrieval code with C++ BM25 and stopword filtering, HNSW `ef_search=160`, and up to 50 candidates per retriever. The harness checks that the lexical and dense rankings agree across strategies and uses the existing Cross-Encoder inference path. Stopword filtering is enabled for this experiment, while the standard endpoint defaults to disabled.

- **Union** uses the current service's candidate selection and reranks all unique lexical and dense candidates.
- **RRF40 and RRF50** rank the union with lexical weight 0.4, dense weight 0.6, and RRF constant 60, then cap the pool at 40 or 50.
- **Dense30 + BM25-only 10** retains the first 30 dense candidates and up to 10 lexical candidates outside the dense Top-50. Dense candidates fill the remaining slots.

HTTP timing covers retrieving, selecting, and reranking candidates over loopback, plus response serialization. The experiment routes omit history writes and return diagnostic IDs. A separate pass counts tokenizer output after truncation and before padding, including special tokens. It also checks that the IDs used for token counting match the timed requests. This benchmark measures the cost of each strategy with one request at a time.

Quality uses MS MARCO relevance judgments. Paired 95% intervals use 10,000 query-level bootstrap draws at seed 5013. The provisional acceptable nDCG@10 loss is 0.005 against the Union baseline. RRF50 is retained as the prior pilot reference.

## Re-run

To recompute the reported statistics from the committed numerical observations, use Python 3.13 with the project's requirements installed.

```bash
python -m backend.scripts.benchmark_candidate_budgets \
  --summarize artifacts/relevance/candidate_budgets/observations.csv
```

For a live replay, prepare the PostgreSQL corpus, HNSW index, and BM25 binary using the [full-corpus setup](../../README.md#reproduction). Set `DATABASE_URL`, `CPP_CORPUS_PATH`, and, for a specific compiled binary, `RETRIEVAL_CPP_EXTENSION`. Use a database connection with read-only access. Start the separate loopback server from the repository root.

```bash
PGOPTIONS="-c default_transaction_read_only=on" \
  CROSS_ENCODER_DEVICE=cpu CROSS_ENCODER_BATCH_SIZE=32 OMP_NUM_THREADS=2 \
  python -m backend.scripts.candidate_budget_server --port 8011
```

In a second terminal, activate the same Python environment and install `httpx==0.28.1` for the benchmark client.

```bash
python -m pip install 'httpx==0.28.1'
python -m backend.scripts.benchmark_candidate_budgets \
  --base-url http://127.0.0.1:8011 \
  --output results/reproduce-candidate-budgets/observations.csv
```

The published replay used macOS arm64, Python 3.13.13, PostgreSQL 17.11 with pgvector 0.8.6, Torch 2.15.0, sentence-transformers 6.3.0, transformers 5.23.0, and NumPy 2.5.4. It used API base commit `e780a3e` and the v0.7.9 C++ binary with SHA-256 `03acbc8cfadf17cbcf016bae0f6370d5989aa0a8069e942d898302c575f5e02d`. The cached model revisions were `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` for all-MiniLM-L6-v2 and `233902d25c440f23af6f7d6e94d2946bac0bee0a` for ms-marco-MiniLM-L-6-v2. Record binary and model versions when comparing replayed values. [MS MARCO data and terms](https://microsoft.github.io/msmarco/)
