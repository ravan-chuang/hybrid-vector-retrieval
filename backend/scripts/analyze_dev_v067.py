import json
from pathlib import Path

import numpy as np


BASE = Path("results/msmarco_full/relevance/dev_v067")

CPP_PATH = BASE / "cpp_ce/results_dev_6980q.json"
PG_PATH = BASE / "pg_ce/results_dev_6980q.json"
CACHE_PATH = BASE / "ce_candidates_dev.json"

OUT_PATH = BASE / "dev_v067_comparison.json"

METRICS = (
    "mrr_at_10",
    "ndcg_at_10",
    "recall_at_10",
)

BOOTSTRAP_SAMPLES = 10_000
SEED = 42


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def extract_metrics(row):
    return row["union_ce"]["metrics"]


def main():
    cpp_data = load(CPP_PATH)
    pg_data = load(PG_PATH)
    cache_data = load(CACHE_PATH)

    cpp = {
        str(row["query_id"]): row
        for row in cpp_data["per_query"]
    }

    pg = {
        str(row["query_id"]): row
        for row in pg_data["per_query"]
    }

    cache = {
        str(row["query_id"]): row
        for row in cache_data["queries"]
    }

    assert len(cpp) == 6980
    assert len(pg) == 6980
    assert len(cache) == 6980
    assert set(cpp) == set(pg) == set(cache)

    qids = list(cpp)

    print("=" * 72)
    print("v0.6.7e — OFFICIAL DEV PAIRED EVALUATION")
    print("=" * 72)
    print(f"Matched Queries: {len(qids)}")

    results = {}

    for metric in METRICS:
        cpp_values = np.array(
            [extract_metrics(cpp[qid])[metric] for qid in qids],
            dtype=np.float64,
        )

        pg_values = np.array(
            [extract_metrics(pg[qid])[metric] for qid in qids],
            dtype=np.float64,
        )

        delta = cpp_values - pg_values

        rng = np.random.default_rng(SEED)

        samples = np.empty(
            BOOTSTRAP_SAMPLES,
            dtype=np.float64,
        )

        for i in range(BOOTSTRAP_SAMPLES):
            indices = rng.integers(
                0,
                len(delta),
                size=len(delta),
            )

            samples[i] = delta[indices].mean()

        ci = np.quantile(samples, [0.025, 0.975])

        result = {
            "cpp_mean": float(cpp_values.mean()),
            "pg_mean": float(pg_values.mean()),
            "delta": float(delta.mean()),
            "ci95": [float(ci[0]), float(ci[1])],
            "wins": int(np.sum(delta > 1e-12)),
            "ties": int(np.sum(np.abs(delta) <= 1e-12)),
            "losses": int(np.sum(delta < -1e-12)),
        }

        results[metric] = result

        print(f"\n{metric}")
        print(f"  C++      : {result['cpp_mean']:.6f}")
        print(f"  PG       : {result['pg_mean']:.6f}")
        print(f"  Delta    : {result['delta']:+.6f}")
        print(
            "  95% CI   : "
            f"[{ci[0]:+.6f}, {ci[1]:+.6f}]"
        )
        print(
            "  W/T/L    : "
            f"{result['wins']}/"
            f"{result['ties']}/"
            f"{result['losses']}"
        )

    candidate_rows = []

    for qid in qids:
        relevant = set(
            map(int, cache[qid]["relevant_document_ids"])
        )

        dense = {
            int(x["document_id"])
            for x in cache[qid]["dense"]["candidates"]
        }

        lexical = {
            int(x["document_id"])
            for x in cache[qid]["lexical"]["candidates"]
        }

        pg_union = dense | lexical

        cpp_union_count = cpp[qid]["candidate_counts"]["union"]

        pg_union_count = len(pg_union)

        pg_relevant = relevant & pg_union

        cpp_top10 = set(
            map(int, cpp[qid]["union_ce"]["top10"])
        )

        pg_top10 = set(
            map(int, pg[qid]["union_ce"]["top10"])
        )

        candidate_rows.append({
            "query_id": qid,
            "pg_candidate_recall": len(pg_relevant) / len(relevant),
            "pg_union_count": pg_union_count,
            "cpp_union_count": cpp_union_count,
            "cpp_top10_relevant": len(cpp_top10 & relevant),
            "pg_top10_relevant": len(pg_top10 & relevant),
        })

    summary = {
        "dataset": "msmarco_passage_dev_small",
        "queries": len(qids),
        "paired_bootstrap": results,
        "candidate_budget": {
            "pg_mean": float(np.mean([
                x["pg_union_count"] for x in candidate_rows
            ])),
            "cpp_mean": float(np.mean([
                x["cpp_union_count"] for x in candidate_rows
            ])),
        },
        "pg_candidate_recall": float(np.mean([
            x["pg_candidate_recall"] for x in candidate_rows
        ])),
    }

    OUT_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== Candidate Budget ===")
    print(summary["candidate_budget"])

    print("\n=== PostgreSQL Candidate Recall ===")
    print(f"{summary['pg_candidate_recall']:.6f}")

    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
