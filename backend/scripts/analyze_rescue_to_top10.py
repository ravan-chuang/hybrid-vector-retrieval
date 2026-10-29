import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


BASE = Path("results/msmarco_full/relevance")

PATHS = {
    "candidate": BASE / "cpp_v066/candidate_complementarity.json",
    "cpp": BASE / "cpp_v065/results_10k_10000q.json",
    "pg": BASE / "ce_rerank_results_10k.json",
}

OUTPUT_DIR = BASE / "cpp_v066"

METRICS = (
    "mrr_at_10",
    "ndcg_at_10",
    "recall_at_10",
)


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def index_by_query(rows, label):
    result = {}

    for row in rows:
        query_id = str(row["query_id"])

        if query_id in result:
            raise ValueError(
                f"Duplicate query_id in {label}: {query_id}"
            )

        result[query_id] = row

    return result


def mean(values):
    if not values:
        return None

    return float(np.mean(values))


def main():
    print("Loading evaluation results...")

    candidate = load(PATHS["candidate"])
    cpp = load(PATHS["cpp"])
    pg = load(PATHS["pg"])

    candidate_rows = index_by_query(
        candidate["per_query"], "candidate"
    )

    cpp_rows = index_by_query(
        cpp["per_query"], "cpp"
    )

    pg_rows = index_by_query(
        pg["per_query"], "pg"
    )

    ids = set(candidate_rows)

    if ids != set(cpp_rows) or ids != set(pg_rows):
        raise ValueError(
            "Query IDs do not match across evaluation files."
        )

    records = []
    stats = Counter()

    total_rescued_docs = 0
    total_promoted_docs = 0

    for query_id in sorted(ids):

        c = candidate_rows[query_id]
        cpp_row = cpp_rows[query_id]
        pg_row = pg_rows[query_id]

        if not (
            c["query"]
            == cpp_row["query"]
            == pg_row["query"]
        ):
            raise ValueError(
                f"Query text mismatch: {query_id}"
            )

        rescued = set(
            c["cpp_rescued_document_ids"]
        )

        pg_only = set(
            c["pg_only_document_ids"]
        )

        cpp_top10 = set(
            cpp_row["union_ce"]["top10"]
        )

        pg_top10 = set(
            pg_row["union_ce"]["top10"]
        )

        promoted = rescued & cpp_top10

        cpp_metrics = cpp_row["union_ce"]["metrics"]
        pg_metrics = pg_row["union_ce"]["metrics"]

        delta = {
            metric: (
                float(cpp_metrics[metric])
                - float(pg_metrics[metric])
            )
            for metric in METRICS
        }

        candidate_delta = (
            c["cpp_candidate_recall"]
            - c["pg_candidate_recall"]
        )

        has_rescue = bool(rescued)

        if has_rescue:
            stats["rescue_queries"] += 1

        if promoted:
            stats["successful_rescue_queries"] += 1

        if has_rescue and not promoted:
            stats["unpromoted_rescue_queries"] += 1

        if candidate_delta > 1e-12:
            stats["candidate_recall_improved"] += 1

        if (
            candidate_delta > 1e-12
            and delta["mrr_at_10"] < -1e-12
        ):
            stats["ranking_regression_queries"] += 1

        total_rescued_docs += len(rescued)
        total_promoted_docs += len(promoted)

        record = {
            "query_id": query_id,
            "query": c["query"],

            "candidate_recall_delta": candidate_delta,

            "cpp_rescued_count": len(rescued),
            "pg_only_count": len(pg_only),

            "promoted_count": len(promoted),

            "cpp_rescued_document_ids": sorted(rescued),
            "promoted_document_ids": sorted(promoted),

            "mrr_delta": delta["mrr_at_10"],
            "ndcg_delta": delta["ndcg_at_10"],
            "recall_delta": delta["recall_at_10"],
        }

        records.append(record)

    rescue_records = [
        r for r in records
        if r["cpp_rescued_count"] > 0
    ]

    non_rescue_records = [
        r for r in records
        if r["cpp_rescued_count"] == 0
    ]

    summary = {
        "queries": len(records),

        "rescue_queries": stats["rescue_queries"],

        "successful_rescue_queries":
            stats["successful_rescue_queries"],

        "unpromoted_rescue_queries":
            stats["unpromoted_rescue_queries"],

        "candidate_recall_improved_queries":
            stats["candidate_recall_improved"],

        "ranking_regression_queries":
            stats["ranking_regression_queries"],

        "total_rescued_relevant_documents":
            total_rescued_docs,

        "total_promoted_relevant_documents":
            total_promoted_docs,

        "rescue_promotion_rate": (
            total_promoted_docs / total_rescued_docs
            if total_rescued_docs else 0.0
        ),

        "rescue_query_deltas": {
            metric: mean([
                r[f"{metric.split('_at_')[0]}_delta"]
                if metric != "recall_at_10"
                else r["recall_delta"]
                for r in rescue_records
            ])
            for metric in METRICS
        },

        "non_rescue_query_deltas": {
            metric: mean([
                r[f"{metric.split('_at_')[0]}_delta"]
                if metric != "recall_at_10"
                else r["recall_delta"]
                for r in non_rescue_records
            ])
            for metric in METRICS
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = (
        OUTPUT_DIR / "rescue_to_top10_10k.json"
    )

    csv_path = (
        OUTPUT_DIR / "rescue_to_top10_10k.csv"
    )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "per_query": records,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        fields = [
            "query_id",
            "query",
            "candidate_recall_delta",
            "cpp_rescued_count",
            "pg_only_count",
            "promoted_count",
            "mrr_delta",
            "ndcg_delta",
            "recall_delta",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for record in records:
            writer.writerow({
                key: record[key]
                for key in fields
            })

    print("\n=== Rescue-to-Top10 Analysis ===")

    print("Queries:", summary["queries"])

    print(
        "C++ Rescue Queries:",
        summary["rescue_queries"]
    )

    print(
        "Successful Rescue Queries:",
        summary["successful_rescue_queries"]
    )

    print(
        "Unpromoted Rescue Queries:",
        summary["unpromoted_rescue_queries"]
    )

    print(
        "Total Rescued Relevant Documents:",
        total_rescued_docs
    )

    print(
        "Promoted Relevant Documents:",
        total_promoted_docs
    )

    print(
        "Rescue Promotion Rate:",
        f"{summary['rescue_promotion_rate']:.4%}"
    )

    print(
        "Candidate Recall Improved Queries:",
        summary["candidate_recall_improved_queries"]
    )

    print(
        "Ranking Regression Queries:",
        summary["ranking_regression_queries"]
    )

    print("\n=== Rescue Query Metric Deltas ===")

    for metric, value in summary[
        "rescue_query_deltas"
    ].items():
        print(f"{metric}: {value:+.6f}")

    print("\n=== Non-Rescue Query Metric Deltas ===")

    for metric, value in summary[
        "non_rescue_query_deltas"
    ].items():
        print(f"{metric}: {value:+.6f}")

    print("\nSaved:", json_path)
    print("Saved:", csv_path)


if __name__ == "__main__":
    main()
