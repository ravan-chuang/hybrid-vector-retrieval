# Four-pipeline MS MARCO dev-small comparison

The 6,980 queries produced 18 pairwise comparisons across BM25, Dense HNSW (`ef_search=160`), Weighted RRF, and Hybrid Union + Cross-Encoder. The metrics are MRR@10, binary nDCG@10, and relevance Recall@10. All runs searched the 8,841,823-passage corpus. Candidate pools and budgets varied across pipelines, so the differences describe the complete configurations.

| Comparison | Metric | Mean difference | Paired 95% CI | Holm-adjusted permutation p |
| --- | --- | ---: | ---: | ---: |
| RRF minus Dense | MRR@10 | +0.001865 | [−0.001903, +0.005633] | 0.328967 |
| RRF minus Dense | nDCG@10 | +0.003779 | [+0.000642, +0.006888] | 0.037996 |
| RRF minus Dense | Recall@10 | +0.009312 | [+0.005062, +0.013658] | 0.001800 |
| Hybrid Union + CE minus RRF | nDCG@10 | +0.114349 | [+0.107234, +0.121522] | 0.001800 |

The listed directions put the later pipeline first. The archived comparisons use `method_a - method_b` in the fixed method order above. The intervals are pointwise percentile intervals from 10,000 paired query resamples. The separate two-sided sign-flip test uses 10,000 Monte Carlo permutations and plus-one correction. Holm adjustment covers all 18 tests as one family. Both procedures use seed 42 and retain the query ordering of their original runs. The analysis describes the public dev-small set and historical rankings.

## Recompute from a clean checkout

Run from the repository root with Python 3.13 and NumPy 2.x. The original results and replay were verified with NumPy 2.5.3.

```bash
python -m pip install 'numpy>=2,<3'
python -m backend.scripts.reproduce_dev_small_bootstrap --check
```

Omit `--check` to print the recalculated means, intervals, and p-values as JSON. CI compares all rows with the [historical reference](expected.json).

## Archived observations

Each row in [the compressed JSON Lines archive](hit_ranks.jsonl.gz) stores a query ID followed by four arrays of distinct, one-based ranks of judged relevant hits in the Top-10. The [dev-small manifest](../../msmarco_full/dev_small_manifest.json) supplies the denominator for recall and the ideal ranking for nDCG. It also provides the query ID set checked by the replay. The export matched the official query IDs and qrels, applied the project's `pid + 1` document mapping, and checked v0.8.0 against v0.8.1 BM25 Top-100 IDs and scores. [Source paths, SHA-256 digests and protocol](source.json)

The replay recomputes the statistics from these observations. Generating new rankings requires the original corpus, indexes and models recorded for the runs. HTTP latency and load tests use separate workloads.
