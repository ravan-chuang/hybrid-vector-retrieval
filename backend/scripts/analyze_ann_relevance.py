import json
from pathlib import Path
from statistics import mean

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "results/msmarco_full/ann_dev_6980"
MANIFEST = BASE / "artifacts/msmarco_full/dev_small_manifest.json"
OUTPUT = DATA / "ann_relevance_analysis_6980q.json"

EF_VALUES = ["10", "20", "40", "80", "160"]
EXPECTED_QUERIES = 6980
TOP_K = 10


def reciprocal_rank(ids, relevant):
    for rank, doc_id in enumerate(ids[:TOP_K], 1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def hit_at_10(ids, relevant):
    return int(any(doc_id in relevant for doc_id in ids[:TOP_K]))


manifest = json.loads(MANIFEST.read_text())
exact = json.loads((DATA / "exact_ground_truth.json").read_text())
hnsw = json.loads((DATA / "hnsw_checkpoint.json").read_text())

queries = manifest["queries"]

if len(queries) != EXPECTED_QUERIES:
    raise RuntimeError("Manifest query count mismatch")

qrels = {
    str(q["query_id"]): set(q["relevant_document_ids"])
    for q in queries
}

query_ids = list(qrels)

if len(query_ids) != EXPECTED_QUERIES:
    raise RuntimeError("Duplicate query IDs")

if set(exact) != set(query_ids):
    raise RuntimeError("Exact checkpoint query IDs mismatch")

for ef in EF_VALUES:
    if ef not in hnsw:
        raise RuntimeError(f"Missing ef={ef}")

    if set(hnsw[ef]) != set(query_ids):
        raise RuntimeError(f"HNSW checkpoint mismatch: ef={ef}")

results = {
    "dataset": "msmarco_passage_dev_small",
    "query_count": EXPECTED_QUERIES,
    "top_k": TOP_K,
    "methods": {},
    "per_query": {},
}

for qid in query_ids:
    relevant = qrels[qid]
    exact_ids = exact[qid]["document_ids"]

    if len(exact_ids) != TOP_K:
        raise RuntimeError(f"Invalid Exact Top-10: {qid}")

    exact_rr = reciprocal_rank(exact_ids, relevant)

    results["per_query"][qid] = {
        "query": exact[qid]["query"],
        "exact_rr": exact_rr,
        "exact_hit": hit_at_10(exact_ids, relevant),
        "hnsw": {},
    }

    for ef in EF_VALUES:
        ids = hnsw[ef][qid]["document_ids"]

        if len(ids) != TOP_K:
            raise RuntimeError(
                f"Invalid HNSW Top-10: {qid}, ef={ef}"
            )

        ann_recall = len(set(ids) & set(exact_ids)) / TOP_K
        rr = reciprocal_rank(ids, relevant)

        results["per_query"][qid]["hnsw"][ef] = {
            "ann_recall_at_10": ann_recall,
            "rr_at_10": rr,
            "hit_at_10": hit_at_10(ids, relevant),
            "rr_delta_vs_exact": rr - exact_rr,
        }

exact_rows = [
    results["per_query"][qid]
    for qid in query_ids
]

results["methods"]["exact"] = {
    "ann_recall_at_10": 1.0,
    "mrr_at_10": mean(
        row["exact_rr"] for row in exact_rows
    ),
    "hit_at_10": mean(
        row["exact_hit"] for row in exact_rows
    ),
}

for ef in EF_VALUES:
    rows = [
        results["per_query"][qid]["hnsw"][ef]
        for qid in query_ids
    ]

    results["methods"][f"hnsw_ef{ef}"] = {
        "ann_recall_at_10": mean(
            row["ann_recall_at_10"] for row in rows
        ),
        "mrr_at_10": mean(
            row["rr_at_10"] for row in rows
        ),
        "hit_at_10": mean(
            row["hit_at_10"] for row in rows
        ),
        "rr_improved_queries": sum(
            row["rr_delta_vs_exact"] > 0 for row in rows
        ),
        "rr_degraded_queries": sum(
            row["rr_delta_vs_exact"] < 0 for row in rows
        ),
        "rr_unchanged_queries": sum(
            row["rr_delta_vs_exact"] == 0 for row in rows
        ),
        "low_ann_recall_no_rr_loss": sum(
            row["ann_recall_at_10"] < 0.8
            and row["rr_delta_vs_exact"] >= 0
            for row in rows
        ),
    }

OUTPUT.write_text(
    json.dumps(results, indent=2, ensure_ascii=False)
)

print(f"Queries evaluated: {EXPECTED_QUERIES}")

print(
    f"\n{'Method':<18}"
    f"{'ANN Recall':>13}"
    f"{'MRR@10':>12}"
    f"{'Hit@10':>12}"
)

print("-" * 55)

for method, row in results["methods"].items():
    label = (
        "Exact" if method == "exact"
        else method.replace("hnsw_ef", "HNSW ef=")
    )

    print(
        f"{label:<18}"
        f"{row['ann_recall_at_10']:>13.4f}"
        f"{row['mrr_at_10']:>12.4f}"
        f"{row['hit_at_10']:>12.4f}"
    )

print("\nPer-query relevance changes:")

for ef in EF_VALUES:
    row = results["methods"][f"hnsw_ef{ef}"]

    print(
        f"ef={ef:<3} "
        f"Improved={row['rr_improved_queries']:<5} "
        f"Degraded={row['rr_degraded_queries']:<5} "
        f"Unchanged={row['rr_unchanged_queries']:<5} "
        f"LowRecallNoLoss={row['low_ann_recall_no_rr_loss']}"
    )

print(f"\nSaved: {OUTPUT}")
