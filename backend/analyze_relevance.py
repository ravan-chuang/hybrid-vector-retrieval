import json
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path("results/relevance/per_query_results.csv")
OUTPUT_DIR = Path("results/relevance/analysis")

SEED = 42
BOOTSTRAP_SAMPLES = 10_000

METHODS = {
    "Lexical": "lexical",
    "Dense": "vector",
    "Hybrid": "hybrid",
}

METRICS = {
    "MRR@10": "mrr@10",
    "nDCG@10": "ndcg@10",
    "Recall@10": "recall@10",
}

COMPARISONS = [
    ("Dense", "Hybrid"),
    ("Dense", "Lexical"),
    ("Hybrid", "Lexical"),
]


def bootstrap_mean_difference(a, b, n_bootstrap, seed):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if len(a) != len(b):
        raise ValueError("Paired samples must have equal length")

    differences = a - b
    observed = float(differences.mean())

    rng = np.random.default_rng(seed)
    n = len(differences)

    bootstrap_means = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        bootstrap_means[i] = differences[indices].mean()

    lower, upper = np.percentile(
        bootstrap_means,
        [2.5, 97.5],
    )

    # Two-sided bootstrap sign probability.
    p_lower = np.mean(bootstrap_means <= 0)
    p_upper = np.mean(bootstrap_means >= 0)
    p_value = min(
        1.0,
        2.0 * min(p_lower, p_upper),
    )

    return {
        "mean_difference": observed,
        "ci_95_lower": float(lower),
        "ci_95_upper": float(upper),
        "bootstrap_p_value": float(p_value),
    }


def win_tie_loss(a, b, tolerance=1e-12):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    difference = a - b

    wins = int(np.sum(difference > tolerance))
    losses = int(np.sum(difference < -tolerance))
    ties = int(len(difference) - wins - losses)

    return {
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "win_rate": wins / len(difference),
        "tie_rate": ties / len(difference),
        "loss_rate": losses / len(difference),
    }


def metric_column(method, metric):
    return f"{METHODS[method]}_{METRICS[metric]}"


def build_representative_cases(df):
    result = df[
        [
            "query_id",
            "query",
            "relevant_doc_ids",
            "lexical_top10",
            "vector_top10",
            "hybrid_top10",
            "lexical_ndcg@10",
            "vector_ndcg@10",
            "hybrid_ndcg@10",
        ]
    ].copy()

    result["dense_minus_hybrid"] = (
        result["vector_ndcg@10"]
        - result["hybrid_ndcg@10"]
    )

    result["hybrid_minus_dense"] = (
        result["hybrid_ndcg@10"]
        - result["vector_ndcg@10"]
    )

    result["lexical_minus_dense"] = (
        result["lexical_ndcg@10"]
        - result["vector_ndcg@10"]
    )

    result["hybrid_synergy"] = (
        result["hybrid_ndcg@10"]
        - result[
            [
                "lexical_ndcg@10",
                "vector_ndcg@10",
            ]
        ].max(axis=1)
    )

    cases = []

    categories = [
        (
            "dense_dominance",
            "dense_minus_hybrid",
            True,
        ),
        (
            "hybrid_over_dense",
            "hybrid_minus_dense",
            True,
        ),
        (
            "lexical_rescue",
            "lexical_minus_dense",
            True,
        ),
        (
            "hybrid_synergy",
            "hybrid_synergy",
            True,
        ),
    ]

    for category, column, descending in categories:
        subset = result.sort_values(
            column,
            ascending=not descending,
        ).head(10)

        for _, row in subset.iterrows():
            record = row.to_dict()
            record["category"] = category
            cases.append(record)

    # Queries where all methods completely fail.
    all_fail = result[
        (result["lexical_ndcg@10"] == 0)
        & (result["vector_ndcg@10"] == 0)
        & (result["hybrid_ndcg@10"] == 0)
    ].head(10)

    for _, row in all_fail.iterrows():
        record = row.to_dict()
        record["category"] = "all_fail"
        cases.append(record)

    return pd.DataFrame(cases)


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(INPUT_PATH)

    print("=" * 70)
    print("PAIRED RELEVANCE ANALYSIS")
    print("=" * 70)
    print(f"Queries:           {len(df):,}")
    print(f"Bootstrap samples: {BOOTSTRAP_SAMPLES:,}")
    print(f"Seed:              {SEED}")
    print()

    paired_rows = []
    wtl_rows = []
    bootstrap_rows = []

    for metric in METRICS:
        print(metric)
        print("-" * 70)

        for method_a, method_b in COMPARISONS:
            column_a = metric_column(
                method_a,
                metric,
            )
            column_b = metric_column(
                method_b,
                metric,
            )

            a = df[column_a].to_numpy()
            b = df[column_b].to_numpy()

            mean_a = float(a.mean())
            mean_b = float(b.mean())

            wtl = win_tie_loss(a, b)

            bootstrap = bootstrap_mean_difference(
                a,
                b,
                BOOTSTRAP_SAMPLES,
                SEED,
            )

            row = {
                "metric": metric,
                "method_a": method_a,
                "method_b": method_b,
                "mean_a": mean_a,
                "mean_b": mean_b,
                **wtl,
                **bootstrap,
            }

            paired_rows.append(row)

            wtl_rows.append(
                {
                    "metric": metric,
                    "method_a": method_a,
                    "method_b": method_b,
                    **wtl,
                }
            )

            bootstrap_rows.append(
                {
                    "metric": metric,
                    "method_a": method_a,
                    "method_b": method_b,
                    "mean_a": mean_a,
                    "mean_b": mean_b,
                    **bootstrap,
                }
            )

            print(
                f"{method_a:7s} vs {method_b:7s} | "
                f"Δ={bootstrap['mean_difference']:+.4f} | "
                f"95% CI "
                f"[{bootstrap['ci_95_lower']:+.4f}, "
                f"{bootstrap['ci_95_upper']:+.4f}] | "
                f"W/T/L="
                f"{wtl['wins']}/"
                f"{wtl['ties']}/"
                f"{wtl['losses']} | "
                f"p≈{bootstrap['bootstrap_p_value']:.4f}"
            )

        print()

    paired_df = pd.DataFrame(paired_rows)
    wtl_df = pd.DataFrame(wtl_rows)
    bootstrap_df = pd.DataFrame(bootstrap_rows)

    representative_df = build_representative_cases(df)

    paired_df.to_csv(
        OUTPUT_DIR / "paired_comparison.csv",
        index=False,
    )

    wtl_df.to_csv(
        OUTPUT_DIR / "win_tie_loss.csv",
        index=False,
    )

    bootstrap_df.to_csv(
        OUTPUT_DIR / "bootstrap_ci.csv",
        index=False,
    )

    representative_df.to_csv(
        OUTPUT_DIR / "representative_cases.csv",
        index=False,
    )

    summary = {
        "query_count": len(df),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "seed": SEED,
        "comparisons": paired_rows,
    }

    (
        OUTPUT_DIR / "statistical_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 70)
    print("Saved analysis artifacts:")
    for filename in [
        "paired_comparison.csv",
        "win_tie_loss.csv",
        "bootstrap_ci.csv",
        "representative_cases.csv",
        "statistical_summary.json",
    ]:
        print(f"  {OUTPUT_DIR / filename}")


if __name__ == "__main__":
    main()
