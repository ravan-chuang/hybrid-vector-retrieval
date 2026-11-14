# Recent Full-Corpus Evaluations

The November 13–14, 2026 runs used API source at [`0d1f1e2`](https://github.com/ravan-chuang/hybrid-vector-retrieval/commit/0d1f1e25bd8db530c39683e4588d9f020461482f) and the local v0.7.9 C++ BM25 binary. The benchmark corpus held 8,841,823 MS MARCO passages in PostgreSQL 17.11 with pgvector 0.8.6. Test APIs read the same corpus and disabled search-history writes. The tables link to compact summaries. The request-level files and experiment harnesses are archived locally.

| Input | SHA-256 |
| --- | --- |
| [MS MARCO dev-small query manifest](../../artifacts/msmarco_full/dev_small_manifest.json) | `e4809fe62414882bbba2a435d1adcff7ab8227507bb7c2e56aa417084a1c7e0d` |
| macOS arm64 v0.7.9 BM25 extension | `03acbc8cfadf17cbcf016bae0f6370d5989aa0a8069e942d898302c575f5e02d` |

## HNSW search breadth

The test sent 6,480 measured requests to `/search/hybrid` using 240 distinct judged queries (seed 1337). It crossed `ef_search=50, 80, 160` with client concurrency 1, 8, and 16 over three rounds. It retrieved 50 candidates per method and returned 10 results using RRF with lexical and dense weights of 0.1 and 0.9. Each round and configuration used 32 warmup requests.

| ef_search | Concurrent clients | SQL P95 | HTTP P95 | Dense qrel Recall@50 | Hybrid nDCG@10 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 16 | 293 ms | 476 ms | 0.6604 | 0.3386 |
| 80 | 16 | 314 ms | 479 ms | 0.6854 | 0.3510 |
| 160 | 16 | 600 ms | 901 ms | 0.7313 | 0.3694 |

Each latency is the median of three round P95 values. SQL time spans the database cursor call and result fetch, excluding pool wait. Quality uses MS MARCO relevance judgments and evaluates hybrid ranking before reranking. The paired nDCG@10 differences versus 160 were −0.0308 (95% CI [−0.0511, −0.0140]) at 50 and −0.0184 (95% CI [−0.0331, −0.0064]) at 80. The evaluation baseline stays at 160. [All nine configurations](hnsw-search-breadth.csv)

## Cross-Encoder candidate budgets

The pilot used 120 judged queries from the dev-small test split, excluding query texts from the HNSW study (seed 5013). It fixed `ef_search=160`, retrieved 50 candidates per method, and used the same MiniLM Cross-Encoder on CPU with an inference batch size of 32. RRF used lexical and dense weights of 0.4 and 0.6. The 30+10 quota takes 30 dense candidates and up to 10 BM25 candidates outside the dense Top-50, filling any remaining slots from the dense ranking.

| Strategy | nDCG@10 | Mean tokens per query | End-to-end HTTP P95 |
| --- | ---: | ---: | ---: |
| RRF40 | 0.4196 | 3,329 | 582 ms |
| Dense30 + BM25-only 10 | 0.4281 | 3,323 | 568 ms |
| RRF50 | 0.4393 | 4,168 | 701 ms |

The token count uses the model tokenizer after truncation and before padding, including special tokens. The two 40-candidate strategies saved about 20% of tokens. Both paired 95% confidence intervals extended below the provisional −0.005 nDCG@10 margin relative to RRF50. The API continues to use the union of the two candidate lists. [Strategy summaries and intervals](candidate-budget.csv)

## Open-loop rerank capacity

The benchmark replayed the same Poisson arrivals and query sequence for `/search/rerank` and `/search/rerank/batched` at each rate. Each of the 18 sessions had 30 seconds of warmup and 120 seconds of measurement. The 5,460 measured requests used live retrieval with `ef_search=160`, 50 candidates per method, and the candidate union. Tests ran on one local CPU API process. The batch scheduler used the pair policy, with a limit of four requests or 512 pairs per batch and a 5 ms formation window.

| Target | Standard P95 | Batched P95 | Standard errors | Batched errors |
| ---: | ---: | ---: | ---: | ---: |
| 2 RPS | 2.88 s | 2.66 s | 0 / 716 | 0 / 716 |
| 2.5 RPS | 8.79 s | 4.45 s | 0 / 918 | 0 / 918 |
| 3 RPS | 30.01 s | 10.13 s | 345 / 1,096 | 336 / 1,096 |

P95 is the median of three round values, measured from scheduled arrival through response or error. At 2 RPS, both routes delivered a median of 1.93 successful requests per second within the measurement window. The batch route passed all three rounds and the standard route passed two under the provisional targets of P95 ≤ 3 seconds, errors ≤ 1%, and at least 95% of arrivals completed successfully in the window. At 3 RPS, the standard client timed out at 30 seconds and the batch scheduler returned HTTP 504 after its separate 10-second timeout. Among 2,157 matched successful responses, the Top-10 document sets agreed and 12 had order changes. The demo uses the standard route. [Per-round results](rerank-capacity-rounds.csv)

For the offline ANN study, the full-corpus quality comparisons, and the frozen-candidate MPS experiments, see the [main evaluation section](../../README.md#evaluation).
