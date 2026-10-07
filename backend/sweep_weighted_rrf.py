import argparse
import json
from pathlib import Path

import pandas as pd


TOP_K = 10
RRF_K = 60

# lexical weight: 0.0 -> 1.0
WEIGHTS = [
    round(i / 10, 1)
    for i in range(11)
]


def split_ids(value):
    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    return value.split("|")


def weighted_rrf(
    lexical_results,
    vector_results,
    lexical_weight,
    top_k=TOP_K,
    rrf_k=RRF_K,
):
    vector_weight = 1.0 - lexical_weight
    scores = {}

    # Important:
    # Do not insert zero-weight candidates into the union.
    if lexical_weight > 0.0:
        for rank, pid in enumerate(
            lexical_results,
            start=1,
        ):
            scores[pid] = (
                scores.get(pid, 0.0)
                + lexical_weight
                / (rrf_k + rank)
            )

    if vector_weight > 0.0:
        for rank, pid in enumerate(
            vector_results,
            start=1,
        ):
            scores[pid] = (
                scores.get(pid, 0.0)
                + vector_weight
                / (rrf_k + rank)
            )

    ranked = sorted(
        scores.items(),
        key=lambda item: (
            -item[1],
            int(item[0]),
        ),
    )

    return [
        pid
        for pid, _ in ranked[:top_k]
    ]


def reciprocal_rank_at_k(
    ranked,
    relevant,
    k=TOP_K,
):
    for rank, pid in enumerate(
        ranked[:k],
        start=1,
    ):
        if pid in relevant:
            return 1.0 / rank

    return 0.0


def recall_at_k(
    ranked,
    relevant,
    k=TOP_K,
):
    if not relevant:
        return 0.0

    retrieved = set(ranked[:k])

    return (
        len(retrieved & relevant)
        / len(relevant)
    )


def ndcg_at_k(
    ranked,
    relevant,
    k=TOP_K,
):
    import math

    dcg = 0.0

    for rank, pid in enumerate(
        ranked[:k],
        start=1,
    ):
        if pid in relevant:
            dcg += 1.0 / math.log2(
                rank + 1
            )

    ideal_hits = min(
        len(relevant),
        k,
    )

    if ideal_hits == 0:
        return 0.0

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(
            1,
            ideal_hits + 1,
        )
    )

    return dcg / idcg


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "per_query_results.csv "
            "containing complete candidate rankings"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input)

    required = {
        "query_id",
        "relevant_doc_ids",
        "lexical_candidates",
        "vector_candidates",
        "lexical_top10",
        "vector_top10",
        "hybrid_top10",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    aggregate_rows = []
    per_query_rows = []

    for lexical_weight in WEIGHTS:
        vector_weight = (
            1.0 - lexical_weight
        )

        mrr_values = []
        ndcg_values = []
        recall_values = []

        for _, row in df.iterrows():
            relevant = set(
                split_ids(
                    row["relevant_doc_ids"]
                )
            )

            lexical = split_ids(
                row["lexical_candidates"]
            )

            vector = split_ids(
                row["vector_candidates"]
            )

            ranked = weighted_rrf(
                lexical,
                vector,
                lexical_weight,
            )

            mrr = reciprocal_rank_at_k(
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

            mrr_values.append(mrr)
            ndcg_values.append(ndcg)
            recall_values.append(recall)

            per_query_rows.append(
                {
                    "query_id":
                        row["query_id"],
                    "lexical_weight":
                        lexical_weight,
                    "vector_weight":
                        vector_weight,
                    "mrr@10":
                        mrr,
                    "ndcg@10":
                        ndcg,
                    "recall@10":
                        recall,
                    "top10":
                        "|".join(ranked),
                }
            )

        aggregate_rows.append(
            {
                "lexical_weight":
                    lexical_weight,
                "vector_weight":
                    vector_weight,
                "mrr@10":
                    sum(mrr_values)
                    / len(mrr_values),
                "ndcg@10":
                    sum(ndcg_values)
                    / len(ndcg_values),
                "recall@10":
                    sum(recall_values)
                    / len(recall_values),
            }
        )

    aggregate_df = pd.DataFrame(
        aggregate_rows
    )

    per_query_df = pd.DataFrame(
        per_query_rows
    )

    # Primary model-selection metric:
    # development mean nDCG@10.
    best = aggregate_df.sort_values(
        by=[
            "ndcg@10",
            "mrr@10",
            "recall@10",
        ],
        ascending=False,
    ).iloc[0]

    aggregate_path = (
        args.output_dir
        / "weight_sweep.csv"
    )

    per_query_path = (
        args.output_dir
        / "per_query_weighted.csv"
    )

    selection_path = (
        args.output_dir
        / "selected_weight.json"
    )

    aggregate_df.to_csv(
        aggregate_path,
        index=False,
    )

    per_query_df.to_csv(
        per_query_path,
        index=False,
    )

    selection = {
        "selection_metric":
            "mean nDCG@10",
        "lexical_weight":
            float(
                best["lexical_weight"]
            ),
        "vector_weight":
            float(
                best["vector_weight"]
            ),
        "mrr@10":
            float(best["mrr@10"]),
        "ndcg@10":
            float(best["ndcg@10"]),
        "recall@10":
            float(best["recall@10"]),
        "top_k":
            TOP_K,
        "rrf_k":
            RRF_K,
        "grid":
            WEIGHTS,
        "input":
            str(args.input),
    }

    selection_path.write_text(
        json.dumps(
            selection,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 72)
    print("WEIGHTED RRF SWEEP")
    print("=" * 72)
    print(
        aggregate_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    print()
    print("Selected on mean nDCG@10:")
    print(
        f"Lexical weight: "
        f"{selection['lexical_weight']:.1f}"
    )
    print(
        f"Vector weight:  "
        f"{selection['vector_weight']:.1f}"
    )
    print(
        f"MRR@10:         "
        f"{selection['mrr@10']:.4f}"
    )
    print(
        f"nDCG@10:        "
        f"{selection['ndcg@10']:.4f}"
    )
    print(
        f"Recall@10:      "
        f"{selection['recall@10']:.4f}"
    )

    print()
    print(f"Saved: {aggregate_path}")
    print(f"Saved: {per_query_path}")
    print(f"Saved: {selection_path}")


if __name__ == "__main__":
    main()
