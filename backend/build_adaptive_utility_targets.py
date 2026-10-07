import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


EPS = 1e-12

DENSE_COLUMN = "dense_ndcg@10"
FIXED_COLUMN = "wL_0.1_ndcg@10"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    features_path = Path(args.features)
    oracle_path = Path(args.oracle)
    output_path = Path(args.output)
    summary_path = Path(args.summary)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with features_path.open(newline="") as f:
        feature_rows = {
            row["query_id"]: row
            for row in csv.DictReader(f)
        }

    with oracle_path.open(newline="") as f:
        oracle_rows = {
            row["query_id"]: row
            for row in csv.DictReader(f)
        }

    if set(feature_rows) != set(oracle_rows):
        raise ValueError(
            "Feature and oracle query IDs do not match"
        )

    rows_out = []
    status_counts = Counter()

    total_dense = 0.0
    total_fixed = 0.0
    total_positive_utility = 0.0
    total_negative_utility = 0.0

    for qid, features in feature_rows.items():
        oracle = oracle_rows[qid]

        dense = float(oracle[DENSE_COLUMN])
        fixed = float(oracle[FIXED_COLUMN])
        delta = fixed - dense

        if delta > EPS:
            status = "beneficial"
            beneficial = 1
            total_positive_utility += delta
        elif delta < -EPS:
            status = "harmful"
            beneficial = 0
            total_negative_utility += delta
        else:
            status = "neutral"
            beneficial = 0

        status_counts[status] += 1
        total_dense += dense
        total_fixed += fixed

        # Remove old oracle-derived label.
        clean_features = {
            key: value
            for key, value in features.items()
            if key != "lexical_required"
        }

        rows_out.append({
            **clean_features,

            # Training target.
            "beneficial_0.1": beneficial,

            # These are evaluation metadata only.
            # NEVER include them in X.
            "utility_status": status,
            "dense_ndcg@10": dense,
            "fixed_0.1_ndcg@10": fixed,
            "delta_0.1_ndcg@10": delta,
        })

    n = len(rows_out)

    summary = {
        "query_count": n,
        "action": {
            "dense_lexical_weight": 0.0,
            "fusion_lexical_weight": 0.1,
            "fusion_vector_weight": 0.9,
        },
        "counts": {
            "beneficial": status_counts["beneficial"],
            "neutral": status_counts["neutral"],
            "harmful": status_counts["harmful"],
        },
        "fractions": {
            "beneficial": status_counts["beneficial"] / n,
            "neutral": status_counts["neutral"] / n,
            "harmful": status_counts["harmful"] / n,
        },
        "mean_dense_ndcg@10": total_dense / n,
        "mean_fixed_0.1_ndcg@10": total_fixed / n,
        "fixed_gain_over_dense":
            (total_fixed - total_dense) / n,
        "total_positive_utility":
            total_positive_utility,
        "total_negative_utility":
            total_negative_utility,
    }

    fieldnames = list(rows_out[0].keys())

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows_out)

    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 72)
    print("UTILITY-ALIGNED ADAPTIVE TARGETS")
    print("=" * 72)
    print(f"Queries:     {n}")
    print()
    print(
        f"Beneficial:  "
        f"{status_counts['beneficial']:4d} "
        f"({100 * status_counts['beneficial'] / n:.1f}%)"
    )
    print(
        f"Neutral:     "
        f"{status_counts['neutral']:4d} "
        f"({100 * status_counts['neutral'] / n:.1f}%)"
    )
    print(
        f"Harmful:     "
        f"{status_counts['harmful']:4d} "
        f"({100 * status_counts['harmful'] / n:.1f}%)"
    )
    print()
    print(
        f"Dense mean nDCG@10: "
        f"{total_dense / n:.4f}"
    )
    print(
        f"Fixed mean nDCG@10: "
        f"{total_fixed / n:.4f}"
    )
    print(
        f"Fixed gain:          "
        f"{(total_fixed - total_dense) / n:+.4f}"
    )
    print()
    print(
        f"Positive utility sum: "
        f"{total_positive_utility:.4f}"
    )
    print(
        f"Negative utility sum: "
        f"{total_negative_utility:.4f}"
    )
    print()
    print("IMPORTANT:")
    print(
        "  dense_ndcg@10, fixed_0.1_ndcg@10 and "
        "delta_0.1_ndcg@10 are evaluation-only metadata."
    )
    print(
        "  They must NEVER be used as model input features."
    )
    print()
    print(f"Saved: {output_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
