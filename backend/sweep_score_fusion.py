import argparse
import csv
import json
import math
from pathlib import Path

import pandas as pd


TOP_K = 10
WEIGHTS = [i / 10 for i in range(11)]


def split_ids(value):
    if pd.isna(value) or str(value).strip() == "":
        return []
    return [
        x for x in str(value).split("|")
        if x
    ]


def split_scores(value):
    if pd.isna(value) or str(value).strip() == "":
        return []
    return [
        float(x)
        for x in str(value).split("|")
        if x
    ]


def minmax_scores(ids, scores, single_value=0.0):
    if len(ids) != len(scores):
        raise ValueError(
            "Candidate/score length mismatch"
        )

    if not ids:
        return {}

    if len(ids) == 1:
        return {
            ids[0]: float(single_value)
        }

    lo = min(scores)
    hi = max(scores)

    if math.isclose(hi, lo):
        return {
            pid: 0.0
            for pid in ids
        }

    return {
        pid: (score - lo) / (hi - lo)
        for pid, score in zip(ids, scores)
    }


def fuse(
    lexical_ids,
    lexical_scores,
    vector_ids,
    vector_scores,
    lexical_weight,
):
    lexical = minmax_scores(
        lexical_ids,
        lexical_scores,
        single_value=0.0,
    )

    vector = minmax_scores(
        vector_ids,
        vector_scores,
        single_value=0.0,
    )

    vector_weight = 1.0 - lexical_weight

    all_ids = set(lexical) | set(vector)

    scores = {}

    for pid in all_ids:
        scores[pid] = (
            lexical_weight
            * lexical.get(pid, 0.0)
            + vector_weight
            * vector.get(pid, 0.0)
        )

    ranked = sorted(
        scores.items(),
        key=lambda x: (
            -x[1],
            int(x[0]),
        ),
    )

    return [
        pid
        for pid, _ in ranked[:TOP_K]
    ]


def mrr_at_k(ranked, relevant):
    relevant = set(relevant)

    for rank, pid in enumerate(
        ranked[:TOP_K],
        start=1,
    ):
        if pid in relevant:
            return 1.0 / rank

    return 0.0


def recall_at_k(ranked, relevant):
    relevant = set(relevant)

    if not relevant:
        return 0.0

    hits = len(
        set(ranked[:TOP_K]) & relevant
    )

    return hits / len(relevant)


def ndcg_at_k(ranked, relevant):
    relevant = set(relevant)

    if not relevant:
        return 0.0

    dcg = 0.0

    for i, pid in enumerate(
        ranked[:TOP_K],
        start=1,
    ):
        if pid in relevant:
            dcg += 1.0 / math.log2(i + 1)

    ideal_hits = min(
        len(relevant),
        TOP_K,
    )

    idcg = sum(
        1.0 / math.log2(i + 1)
        for i in range(1, ideal_hits + 1)
    )

    return dcg / idcg


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scores",
        required=True,
    )

    parser.add_argument(
        "--qrels-results",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    args = parser.parse_args()

    scores = pd.read_csv(args.scores)
    qrels = pd.read_csv(args.qrels_results)

    scores["query_id"] = (
        scores["query_id"].astype(str)
    )

    qrels["query_id"] = (
        qrels["query_id"].astype(str)
    )

    df = qrels[
        [
            "query_id",
            "relevant_doc_ids",
            "vector_ndcg@10",
            "vector_mrr@10",
            "vector_recall@10",
        ]
    ].merge(
        scores,
        on="query_id",
        how="inner",
        validate="one_to_one",
    )

    if len(df) != len(qrels):
        raise RuntimeError(
            f"Expected {len(qrels)} rows, "
            f"matched {len(df)}"
        )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = []
    per_query_all = []

    for lexical_weight in WEIGHTS:
        rows = []

        for _, row in df.iterrows():
            relevant = split_ids(
                row["relevant_doc_ids"]
            )

            lexical_ids = split_ids(
                row["lexical_candidates"]
            )

            lexical_scores = split_scores(
                row["lexical_scores"]
            )

            vector_ids = split_ids(
                row["vector_candidates"]
            )

            vector_scores = split_scores(
                row["vector_scores"]
            )

            ranked = fuse(
                lexical_ids,
                lexical_scores,
                vector_ids,
                vector_scores,
                lexical_weight,
            )

            mrr = mrr_at_k(
                ranked,
                relevant,
            )

            ndcg = ndcg_at_k(
                ranked,
                relevant,
            )

            recall = recall_at_k(
                ranked,
                relevant,
            )

            rows.append(
                (mrr, ndcg, recall)
            )

            per_query_all.append({
                "query_id":
                    row["query_id"],

                "lexical_weight":
                    lexical_weight,

                "vector_weight":
                    1.0
                    - lexical_weight,

                "mrr@10": mrr,
                "ndcg@10": ndcg,
                "recall@10": recall,

                "top10":
                    "|".join(ranked),
            })

        mean_mrr = sum(
            x[0] for x in rows
        ) / len(rows)

        mean_ndcg = sum(
            x[1] for x in rows
        ) / len(rows)

        mean_recall = sum(
            x[2] for x in rows
        ) / len(rows)

        summary.append({
            "lexical_weight":
                lexical_weight,

            "vector_weight":
                1.0 - lexical_weight,

            "mrr@10":
                mean_mrr,

            "ndcg@10":
                mean_ndcg,

            "recall@10":
                mean_recall,
        })

    summary_df = pd.DataFrame(
        summary
    )

    summary_df.to_csv(
        output_dir
        / "sweep_summary.csv",
        index=False,
    )

    pd.DataFrame(
        per_query_all
    ).to_csv(
        output_dir
        / "per_query_all_weights.csv",
        index=False,
    )

    best = max(
        summary,
        key=lambda x: (
            x["ndcg@10"],
            -x["lexical_weight"],
        ),
    )

    with (
        output_dir
        / "selected_weight.json"
    ).open("w") as f:
        json.dump(
            {
                "selection_metric":
                    "mean nDCG@10",

                "normalization":
                    "query-local min-max",

                "lexical_single_candidate":
                    0.0,

                **best,
            },
            f,
            indent=2,
        )

    print("=" * 76)
    print(
        "DEVELOPMENT NORMALIZED "
        "SCORE-FUSION SWEEP"
    )
    print("=" * 76)

    print()
    print(
        f"{'wL':>5} "
        f"{'wD':>5} "
        f"{'MRR@10':>10} "
        f"{'nDCG@10':>10} "
        f"{'Recall@10':>11}"
    )

    for row in summary:
        marker = (
            "  <-- SELECTED"
            if row["lexical_weight"]
            == best["lexical_weight"]
            else ""
        )

        print(
            f"{row['lexical_weight']:5.1f} "
            f"{row['vector_weight']:5.1f} "
            f"{row['mrr@10']:10.4f} "
            f"{row['ndcg@10']:10.4f} "
            f"{row['recall@10']:11.4f}"
            f"{marker}"
        )

    dense_ndcg = float(
        df["vector_ndcg@10"].mean()
    )

    print()
    print(
        "Original Dense nDCG@10: "
        f"{dense_ndcg:.4f}"
    )

    print(
        "Selected Fusion nDCG@10: "
        f"{best['ndcg@10']:.4f}"
    )

    print(
        "Gain vs Dense: "
        f"{best['ndcg@10'] - dense_ndcg:+.4f}"
    )

    print()
    print(
        "Saved:",
        output_dir,
    )


if __name__ == "__main__":
    main()
