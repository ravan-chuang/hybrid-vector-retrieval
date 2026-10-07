import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
BOOTSTRAP_SAMPLES = 10000

METRICS = [
    ("mrr@10", "dense_rerank_mrr@10", "union_rerank_mrr@10"),
    ("ndcg@10", "dense_rerank_ndcg@10", "union_rerank_ndcg@10"),
    ("recall@10", "dense_rerank_recall@10", "union_rerank_recall@10"),
]


def split_ids(value):
    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value or value == "nan":
        return []

    return [x for x in value.split("|") if x]


def paired_bootstrap(a, b, rng):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if len(a) != len(b):
        raise ValueError("Paired arrays have different lengths")

    n = len(a)
    observed = float(np.mean(b - a))

    deltas = np.empty(
        BOOTSTRAP_SAMPLES,
        dtype=float,
    )

    for i in range(BOOTSTRAP_SAMPLES):
        idx = rng.integers(
            0,
            n,
            size=n,
        )

        deltas[i] = np.mean(
            b[idx] - a[idx]
        )

    lower, upper = np.percentile(
        deltas,
        [2.5, 97.5],
    )

    # Auxiliary two-sided bootstrap sign-style probability.
    p_lower = np.mean(deltas <= 0.0)
    p_upper = np.mean(deltas >= 0.0)

    p_approx = min(
        1.0,
        2.0 * min(p_lower, p_upper),
    )

    diff = b - a

    wins = int(np.sum(diff > 1e-12))
    ties = int(np.sum(np.abs(diff) <= 1e-12))
    losses = int(np.sum(diff < -1e-12))

    return {
        "dense_mean": float(np.mean(a)),
        "union_mean": float(np.mean(b)),
        "delta_union_minus_dense": observed,
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "bootstrap_p_approx": float(p_approx),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rerank-results",
        required=True,
    )

    parser.add_argument(
        "--retrieval-results",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    args = parser.parse_args()

    rerank = pd.read_csv(
        args.rerank_results
    )

    retrieval = pd.read_csv(
        args.retrieval_results
    )

    rerank["query_id"] = (
        rerank["query_id"].astype(str)
    )

    retrieval["query_id"] = (
        retrieval["query_id"].astype(str)
    )

    needed = retrieval[
        [
            "query_id",
            "lexical_candidates",
            "vector_candidates",
        ]
    ]

    df = rerank.merge(
        needed,
        on="query_id",
        how="inner",
        validate="one_to_one",
    )

    if len(df) != len(rerank):
        raise RuntimeError(
            f"Expected {len(rerank)} matched rows, "
            f"got {len(df)}"
        )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = np.random.default_rng(SEED)

    paired = {}

    for (
        metric_name,
        dense_col,
        union_col,
    ) in METRICS:
        paired[metric_name] = (
            paired_bootstrap(
                df[dense_col].to_numpy(),
                df[union_col].to_numpy(),
                rng,
            )
        )

    rescue_rows = []

    rescued_queries = 0
    promoted_queries = 0
    rescued_relevant_docs = 0
    promoted_relevant_docs = 0

    lexical_adds_any = 0
    lexical_adds_relevant = 0

    for _, row in df.iterrows():
        relevant = set(
            split_ids(
                row["relevant_doc_ids"]
            )
        )

        lexical = set(
            split_ids(
                row["lexical_candidates"]
            )
        )

        vector = set(
            split_ids(
                row["vector_candidates"]
            )
        )

        dense_top10 = set(
            split_ids(
                row["dense_reranked_top10"]
            )
        )

        union_top10 = set(
            split_ids(
                row["union_reranked_top10"]
            )
        )

        lexical_only = lexical - vector

        if lexical_only:
            lexical_adds_any += 1

        rescued = (
            relevant
            & lexical_only
        )

        if rescued:
            lexical_adds_relevant += 1
            rescued_queries += 1
            rescued_relevant_docs += len(
                rescued
            )

        promoted = (
            rescued
            & union_top10
        )

        if promoted:
            promoted_queries += 1
            promoted_relevant_docs += len(
                promoted
            )

        # Relevant docs already available to Dense Top-50
        dense_available_relevant = (
            relevant & vector
        )

        union_available_relevant = (
            relevant
            & (vector | lexical)
        )

        rescue_rows.append({
            "query_id":
                row["query_id"],

            "query":
                row["query"],

            "relevant_count":
                len(relevant),

            "lexical_only_count":
                len(lexical_only),

            "rescued_relevant_count":
                len(rescued),

            "promoted_rescued_count":
                len(promoted),

            "dense_available_relevant_count":
                len(dense_available_relevant),

            "union_available_relevant_count":
                len(union_available_relevant),

            "rescued_relevant_ids":
                "|".join(
                    sorted(
                        rescued,
                        key=int,
                    )
                ),

            "promoted_rescued_ids":
                "|".join(
                    sorted(
                        promoted,
                        key=int,
                    )
                ),

            "dense_rerank_ndcg@10":
                row[
                    "dense_rerank_ndcg@10"
                ],

            "union_rerank_ndcg@10":
                row[
                    "union_rerank_ndcg@10"
                ],

            "ndcg_delta":
                row[
                    "union_rerank_ndcg@10"
                ]
                - row[
                    "dense_rerank_ndcg@10"
                ],

            "dense_rerank_recall@10":
                row[
                    "dense_rerank_recall@10"
                ],

            "union_rerank_recall@10":
                row[
                    "union_rerank_recall@10"
                ],

            "recall_delta":
                row[
                    "union_rerank_recall@10"
                ]
                - row[
                    "dense_rerank_recall@10"
                ],
        })

    rescue_df = pd.DataFrame(
        rescue_rows
    )

    rescue_summary = {
        "queries": len(df),

        "queries_with_lexical_only_candidates":
            lexical_adds_any,

        "queries_with_lexical_only_relevant":
            lexical_adds_relevant,

        "rescued_queries":
            rescued_queries,

        "rescued_relevant_documents":
            rescued_relevant_docs,

        "queries_where_rescued_relevant_reaches_union_top10":
            promoted_queries,

        "rescued_relevant_documents_promoted_to_union_top10":
            promoted_relevant_docs,

        "promotion_rate_per_rescued_query":
            (
                promoted_queries
                / rescued_queries
                if rescued_queries
                else 0.0
            ),

        "promotion_rate_per_rescued_document":
            (
                promoted_relevant_docs
                / rescued_relevant_docs
                if rescued_relevant_docs
                else 0.0
            ),
    }

    # Candidate-availability ceiling at candidate stage.
    dense_available = (
        rescue_df[
            "dense_available_relevant_count"
        ].sum()
    )

    union_available = (
        rescue_df[
            "union_available_relevant_count"
        ].sum()
    )

    rescue_summary[
        "dense_available_relevant_documents"
    ] = int(dense_available)

    rescue_summary[
        "union_available_relevant_documents"
    ] = int(union_available)

    rescue_summary[
        "additional_relevant_documents_in_union"
    ] = int(
        union_available - dense_available
    )

    # Representative improvements / regressions.
    best = rescue_df.sort_values(
        [
            "ndcg_delta",
            "rescued_relevant_count",
        ],
        ascending=[False, False],
    ).head(10)

    worst = rescue_df.sort_values(
        "ndcg_delta",
        ascending=True,
    ).head(10)

    best.to_csv(
        output_dir
        / "largest_improvements.csv",
        index=False,
    )

    worst.to_csv(
        output_dir
        / "largest_regressions.csv",
        index=False,
    )

    rescue_df.to_csv(
        output_dir
        / "rescue_per_query.csv",
        index=False,
    )

    summary = {
        "protocol": {
            "bootstrap_samples":
                BOOTSTRAP_SAMPLES,
            "seed":
                SEED,
            "comparison":
                "union rerank minus dense rerank",
        },
        "paired_analysis":
            paired,
        "lexical_rescue_analysis":
            rescue_summary,
    }

    with (
        output_dir
        / "analysis_summary.json"
    ).open("w") as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print("=" * 76)
    print(
        "CROSS-ENCODER PAIRED + "
        "LEXICAL RESCUE ANALYSIS"
    )
    print("=" * 76)

    print()
    print(
        f"Queries: {len(df)}"
    )

    print()
    print("PAIRED BOOTSTRAP")

    for metric_name in [
        "mrr@10",
        "ndcg@10",
        "recall@10",
    ]:
        r = paired[metric_name]

        print()
        print(metric_name.upper())

        print(
            f"  Dense: "
            f"{r['dense_mean']:.4f}"
        )

        print(
            f"  Union: "
            f"{r['union_mean']:.4f}"
        )

        print(
            f"  Delta: "
            f"{r['delta_union_minus_dense']:+.4f}"
        )

        print(
            "  95% CI: "
            f"[{r['ci95_lower']:+.4f}, "
            f"{r['ci95_upper']:+.4f}]"
        )

        print(
            "  W/T/L: "
            f"{r['wins']}/"
            f"{r['ties']}/"
            f"{r['losses']}"
        )

        print(
            "  bootstrap p≈"
            f"{r['bootstrap_p_approx']:.4f}"
        )

    print()
    print("LEXICAL RESCUE")

    print(
        "  Queries with lexical-only candidates: "
        f"{lexical_adds_any}/{len(df)}"
    )

    print(
        "  Queries with lexical-only relevant docs: "
        f"{rescued_queries}/{len(df)}"
    )

    print(
        "  Rescued relevant documents: "
        f"{rescued_relevant_docs}"
    )

    print(
        "  Rescued queries promoted into Union Top-10: "
        f"{promoted_queries}/{rescued_queries}"
    )

    print(
        "  Rescued relevant docs promoted into Top-10: "
        f"{promoted_relevant_docs}/"
        f"{rescued_relevant_docs}"
    )

    print(
        "  Dense available relevant docs: "
        f"{dense_available}"
    )

    print(
        "  Union available relevant docs: "
        f"{union_available}"
    )

    print(
        "  Additional relevant docs from lexical: "
        f"{union_available - dense_available}"
    )

    print()
    print("Saved:", output_dir)


if __name__ == "__main__":
    main()
