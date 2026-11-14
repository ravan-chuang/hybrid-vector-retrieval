# Held-out Cross-Encoder Reranking

On 500 held-out queries over a 500,000-passage subset, reranking the union of dense and lexical candidates increased nDCG@10 by **0.015535** over dense-only candidates. The paired 95% confidence interval was **[+0.006970, +0.025332]**.

| Metric | Dense + CE | Union + CE | Delta | Paired 95% CI |
| --- | ---: | ---: | ---: | --- |
| MRR@10 | 0.579639 | 0.593063 | +0.013425 | [+0.005236, +0.023278] |
| nDCG@10 | 0.663292 | 0.678827 | +0.015535 | [+0.006970, +0.025332] |
| Recall@10 | 0.925667 | 0.949333 | +0.023667 | [+0.012000, +0.037333] |

The [frozen protocol](../../artifacts/relevance/protocol_frozen.json) uses HNSW dense Top-50 at `ef_search=50`, PostgreSQL full-text Top-50, and the same MiniLM Cross-Encoder for both pools. The union arm can contain up to 100 candidates. Relevance judgments cover only passages present in the subset. This compares complete candidate pools at their respective sizes.

## Recompute the published table

From the repository root, use Python 3.13. The following commands were verified with NumPy 2.5.4 and pandas 3.1.1.

```bash
python -m pip install 'numpy==2.5.4' 'pandas==3.1.1'
python -m backend.scripts.reproduce_heldout_reranking --check
```

The [CSV](../../artifacts/relevance/heldout_reranking/hit_ranks.csv) contains a sequential row number, the number of judged relevant passages in the subset, and the relevant ranks for both Top-10 lists. Ranks are one-based and pipe-separated. An empty field means the ranking contained no judged hit. These fields reproduce binary nDCG, MRR, and Recall directly. The original row order and 10,000 paired bootstrap resamples at seed 42 reproduce the archived intervals. CI runs the same check against the [archived analysis](../../artifacts/relevance/final_reranking_analysis.json). The CSV contains only these numerical observations.

## Generate rankings from a prepared corpus

Use a PostgreSQL database containing the original 500,000 passages and their MiniLM embeddings in `scalability_large_documents`. The [setup scripts](../../database) use the vector extension, `09_scalability_large.sql`, the [500K ingestion script](../../backend/ingest_scalability_large.py), and `10_scalability_large_hnsw.sql` and `11_large_fulltext.sql` for the indexes. Run ingestion only against a disposable database because it truncates the table. The ingestion script takes the first 500,000 nonempty passages from the streamed dataset. The historical corpus and model revisions were not recorded in the frozen protocol. Verify the available inputs before comparing results. [MS MARCO data and terms](https://microsoft.github.io/msmarco/)

In the project's Python environment, install the [runtime dependencies](../../requirements.txt) and pandas. Set `DATABASE_URL` for the prepared database. Supply the original frozen query JSONL and, if available, the matching `results/relevance/eligible_qrels.jsonl` and `results/relevance/coverage.json` cache. Without that cache the retrieval script rebuilds it from the dataset and the passages available in the database.

To rebuild the split from the same ordered eligible query cache, regenerate the baseline sample with seed 42 and run the two split scripts. Baseline evaluation can also create the eligible query cache. Compare the final JSONL checksum before running retrieval.

```bash
python backend/evaluate_relevance.py --queries 500 --seed 42 \
  --output-dir results/relevance/baseline_equal_rrf
python backend/create_relevance_splits.py
python backend/create_adaptive_final_split.py
```

```bash
python backend/evaluate_relevance.py \
  --query-file results/relevance/splits/adaptive_final_500.jsonl \
  --output-dir results/reproduce-heldout/retrieval

python backend/evaluate_cross_encoder_reranking.py \
  --input results/reproduce-heldout/retrieval/per_query_results.csv \
  --output-dir results/reproduce-heldout/reranking

python -m backend.scripts.reproduce_heldout_reranking \
  --rankings-csv results/reproduce-heldout/reranking/per_query_results.csv
```

Use `--check` with `--rankings-csv` when checking rerun statistics against the archived values. A rerun uses the local corpus, index, model weights and inference device. Compare the rankings when scores differ. The one-command table check uses the committed hit ranks.

On November 14, 2026 the full ranking replay used the original 500K table with read-only access, PostgreSQL 17.11, pgvector 0.8.6, sentence-transformers 6.3.0, and Torch 2.15.0 on CPU. Both first-stage candidate orders and both reranked Top-10 orders matched the archived run for all 500 queries. The archived reranking used Apple MPS.

| Archived input | SHA-256 |
| --- | --- |
| Frozen 500-query JSONL | `c5d4fe5163dcdafec5619e38d0ed4d146a63efd0e4a9f82d81a826a0d4fbabd9` |
| First-stage per-query CSV | `4bb7de510a82998625cb8f5117bc03c224e6c536f70d806ec589d3a5db4ffe0b` |
| Cross-Encoder per-query CSV | `24efe1b97a009d354dc562faac0c688f324852b13123edcb1a25094e7b7051f8` |
| Exported hit-rank CSV | `29ceb09fcaaaecab89bb66ddd549620e7881147b7ec0d58ab70649670dd0e3e3` |

CSV hashes include the original timing columns and identify the archived files. For the replayed 500K database, the ordered external-ID hash is `425eb53d5b2396bd3b1c9dda832492cce15dbbeac340507acc1020bc544d1500`, computed over UTF-8 external IDs in `document_id` order, one ID and LF per record. This identifies passage membership and insertion order.
