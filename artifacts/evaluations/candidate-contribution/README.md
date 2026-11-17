# Historical candidate contribution

This archive reproduces an offline analysis of 6,980 MS MARCO dev-small queries. The source run compared RRF and dense candidate lists of up to 50 documents, each scored by a Cross-Encoder. A query's group depends on the presence of judged relevant documents in the candidate lists.

| Split | RRF-only hits | Dense-only hits | Hits in both lists | Hits in neither list |
| --- | ---: | ---: | ---: | ---: |
| Validation | 184 | 39 | 2,588 | 756 |
| Test | 165 | 31 | 2,567 | 650 |

The reduced archive contains a query ID, split, recorded group, and per-query metric deltas. It reproduces the original group counts, means, contributions and win/tie/loss counts. The original inputs with document IDs are identified by digest in [source.json](source.json).

From the repository root, run

```bash
python -m backend.scripts.reproduce_candidate_contribution --check
```
