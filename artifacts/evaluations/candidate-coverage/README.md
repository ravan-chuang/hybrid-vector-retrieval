# Historical candidate coverage and Cross-Encoder budgets

This archive reproduces the candidate coverage, fixed-depth budget sweep and Top-50 Cross-Encoder summaries for 6,980 MS MARCO dev-small queries. It uses the repository's dev-small manifest for query IDs and positive relevance judgments. SHA-256 of the query ID assigns 3,567 queries to validation and 3,413 to test.

| Analysis | First-stage inputs | RRF weights (BM25 / dense) | Reproduced output |
| --- | --- | --- | --- |
| Candidate coverage | C++ BM25 Top-50 plus cached dense Top-50 | 0.1 / 0.9 | Per-query recall averaged across all queries, candidate overlap and retained relevant hits |
| CE budget sweep and Top-50 ablation | The same frozen lists and the original MiniLM Cross-Encoder scores | 0.4 / 0.6 | MRR@10, nDCG@10 and Recall@10 by split, for dense and RRF at budgets 10, 20, 30, 40 and 50 |

Both analyses use RRF `k=60`. The frozen dense cache used HNSW `ef_search=160`. For coverage, the full candidate union had mean recall of 0.8286, and the RRF Top-50 had mean recall of 0.7389. The retention of 0.8903 divides total judged-relevant hits in RRF Top-50 by total hits in the union. For each budget, the sweep takes the first K dense or fused candidates and reranks them by cached `cross-encoder/ms-marco-MiniLM-L-6-v2` scores. At budget 50, the test split returned nDCG@10 of 0.4387 for Dense + CE and 0.4652 for RRF + CE.

The compressed observations retain each query ID, relevant-hit positions by ranking and budget, the candidate union and overlap sizes, and the number of candidates scored in the original ablation. The replay uses these observations to recompute the summaries. Repeating retrieval or scoring requires the original corpus, indexes, candidate cache and model. [Source paths, digests and protocols](source.json) · [Historical reference summaries](expected.json)

From the repository root, run the check with Python's standard library.

```bash
python -m backend.scripts.reproduce_candidate_coverage --check
```

Omit `--check` to print the recalculated summary as JSON.
