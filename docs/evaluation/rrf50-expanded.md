# RRF50 vs. the Full Candidate Union

On 600 paired dev-small queries, RRF50 reduced mean input tokens by **44.3%** and HTTP P95 by **22.8%**. The nDCG@10 difference (RRF50 minus union) was **−0.012176**, with a paired 95% confidence interval of **[−0.022078, −0.003215]**. Its lower bound fell below the provisional **−0.005** threshold. The demo uses the full candidate union.

| Strategy | Mean candidates | Mean tokens | HTTP P95 | Inference P95 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full union | 88.7 | 7,410 | 913 ms | 575 ms | 0.494443 |
| RRF50 | 50.0 | 4,129 | 705 ms | 352 ms | 0.482267 |

RRF50 scored higher on 21 queries, tied on 560, and scored lower on 19. The confidence interval uses 10,000 paired percentile bootstrap draws at query level with seed 141701.

## Recompute the archived results

From the repository root, run the command below with NumPy 2.x installed. The archived run and local check used Python 3.13 with NumPy 2.5.4.

```bash
python -m backend.scripts.reproduce_expanded_rrf50 --check
```

The command checks the sample manifest, both artifact hashes, the strategy metrics, the interval, and the quality decision against the archived summary. CI runs the same check. The observation CSV contains sequential row numbers and measured values. [Fixed protocol](../../artifacts/relevance/rrf50_expanded/protocol.json) · [1,200 strategy observations](../../artifacts/relevance/rrf50_expanded/observations.csv) · [Reference summary](../../artifacts/relevance/rrf50_expanded/reference.json)

## Protocol and live replay

The experiment used the 8,841,823-passage MS MARCO corpus with PostgreSQL 17.11 and pgvector 0.8.6. It fixed `ef_search=160`, used C++ BM25 v0.7.9 with stopword filtering, and retrieved up to 50 candidates per method. The RRF weights were 0.4 lexical and 0.6 dense with rank constant 60. Both strategies share the same Cross-Encoder and use one local CPU request at a time. The artifact records the API commit, native extension hash, cached model revisions, and sample order.

The sample consists of 600 distinct, judged query texts from the original dev-small hash test half, with 24 additional warmup queries. It excludes texts from the candidate-budget pilot and the HNSW sweep. The measured queries were previously in the test split of the earlier offline quota study. This follow-up tests a separate sample from the recent pilot within the same dataset. See the [pilot protocol](candidate-budgets.md#protocol).

Alternate which strategy runs first, and check that the lexical and vector lists agree for each pair. A separate pass counts input tokens after truncation and before padding, including special tokens. The experiment used read-only database access and omitted history writes. HTTP P95 includes live retrieval, candidate selection, inference, and the loopback response. It describes sequential request latency under this setup.

Prepare the full corpus and start the loopback experiment server as documented in the [candidate-budget benchmark setup](candidate-budgets.md#re-run). With that service running and the client dependencies installed, use the fixed sample:

```bash
python -m backend.scripts.reproduce_expanded_rrf50 \
  --base-url http://127.0.0.1:8011 \
  --output results/reproduce-expanded-rrf50/observations.csv
```

To summarize a replayed CSV, pass its path with `--observations`. Latencies depend on the replay environment. The archive check applies to the recorded observations.
