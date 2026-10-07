import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


TOP_K = 10
RRF_K = 60
WEIGHTS = [i / 10 for i in range(11)]
FIXED_LEXICAL_WEIGHT = 0.1


def parse_ranking(value):
    if not value:
        return []
    return [x for x in value.split("|") if x]


def parse_relevant(value):
    if not value:
        return set()
    return {x for x in value.split("|") if x}


def ndcg_at_k(ranking, relevant, k=TOP_K):
    if not relevant:
        return 0.0

    dcg = 0.0
    for rank, pid in enumerate(ranking[:k], start=1):
        if pid in relevant:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(relevant), k)
    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )
    return dcg / idcg if idcg else 0.0


def weighted_rrf(lexical, vector, lexical_weight):
    vector_weight = 1.0 - lexical_weight
    scores = {}

    # Important: zero-weight rankings must not inject zero-score docs.
    if lexical_weight > 0:
        for rank, pid in enumerate(lexical, start=1):
            scores[pid] = scores.get(pid, 0.0) + (
                lexical_weight / (RRF_K + rank)
            )

    if vector_weight > 0:
        for rank, pid in enumerate(vector, start=1):
            scores[pid] = scores.get(pid, 0.0) + (
                vector_weight / (RRF_K + rank)
            )

    ranked = sorted(
        scores.items(),
        key=lambda item: (-item[1], item[0]),
    )
    return [pid for pid, _ in ranked[:TOP_K]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_out = []
    weight_best_counts = Counter()
    weight_tie_counts = Counter()

    dense_values = []
    fixed_values = []
    oracle_values = []
    oracle_gains = []

    with input_path.open(newline="") as f:
        reader = csv.DictReader(f)

        required = {
            "query_id",
            "query",
            "relevant_doc_ids",
            "lexical_candidates",
            "vector_candidates",
            "vector_ndcg@10",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}"
            )

        for row in reader:
            relevant = parse_relevant(row["relevant_doc_ids"])
            lexical = parse_ranking(row["lexical_candidates"])
            vector = parse_ranking(row["vector_candidates"])

            scores = {}
            for weight in WEIGHTS:
                ranking = weighted_rrf(
                    lexical,
                    vector,
                    weight,
                )
                scores[weight] = ndcg_at_k(
                    ranking,
                    relevant,
                )

            dense_ndcg = float(row["vector_ndcg@10"])
            fixed_ndcg = scores[FIXED_LEXICAL_WEIGHT]
            oracle_ndcg = max(scores.values())

            # All weights achieving the oracle score.
            best_weights = [
                w for w, score in scores.items()
                if math.isclose(
                    score,
                    oracle_ndcg,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ]

            # Deterministic representative weight:
            # prefer the smallest lexical contribution among tied optima.
            representative_weight = min(best_weights)

            weight_best_counts[representative_weight] += 1
            for w in best_weights:
                weight_tie_counts[w] += 1

            gain = oracle_ndcg - fixed_ndcg

            dense_values.append(dense_ndcg)
            fixed_values.append(fixed_ndcg)
            oracle_values.append(oracle_ndcg)
            oracle_gains.append(gain)

            rows_out.append({
                "query_id": row["query_id"],
                "query": row["query"],
                "dense_ndcg@10": dense_ndcg,
                "fixed_0.1_ndcg@10": fixed_ndcg,
                "oracle_ndcg@10": oracle_ndcg,
                "oracle_gain_over_fixed": gain,
                "oracle_gain_over_dense": oracle_ndcg - dense_ndcg,
                "representative_best_lexical_weight":
                    representative_weight,
                "all_best_lexical_weights":
                    "|".join(f"{w:.1f}" for w in best_weights),
                **{
                    f"wL_{w:.1f}_ndcg@10": scores[w]
                    for w in WEIGHTS
                },
            })

    if not rows_out:
        raise ValueError("No queries found")

    dense = np.asarray(dense_values)
    fixed = np.asarray(fixed_values)
    oracle = np.asarray(oracle_values)
    gains = np.asarray(oracle_gains)

    n = len(rows_out)

    fixed_better_dense = int(np.sum(fixed > dense + 1e-12))
    fixed_equal_dense = int(
        np.sum(np.isclose(fixed, dense, atol=1e-12, rtol=1e-12))
    )
    fixed_worse_dense = n - fixed_better_dense - fixed_equal_dense

    oracle_better_fixed = int(np.sum(oracle > fixed + 1e-12))
    oracle_equal_fixed = n - oracle_better_fixed

    oracle_better_dense = int(np.sum(oracle > dense + 1e-12))
    oracle_equal_dense = int(
        np.sum(np.isclose(oracle, dense, atol=1e-12, rtol=1e-12))
    )
    oracle_worse_dense = n - oracle_better_dense - oracle_equal_dense

    # "Lexical required" means no oracle-optimal solution has wL=0.
    lexical_required = sum(
        1
        for row in rows_out
        if "0.0" not in row["all_best_lexical_weights"].split("|")
    )

    summary = {
        "protocol": "development-only oracle weighted RRF diagnostic",
        "query_count": n,
        "top_k": TOP_K,
        "rrf_k": RRF_K,
        "weights": WEIGHTS,
        "fixed_lexical_weight": FIXED_LEXICAL_WEIGHT,

        "mean_dense_ndcg@10": float(dense.mean()),
        "mean_fixed_ndcg@10": float(fixed.mean()),
        "mean_oracle_ndcg@10": float(oracle.mean()),

        "mean_oracle_gain_over_fixed":
            float((oracle - fixed).mean()),
        "mean_oracle_gain_over_dense":
            float((oracle - dense).mean()),

        "oracle_gain_over_fixed_p50":
            float(np.percentile(gains, 50)),
        "oracle_gain_over_fixed_p95":
            float(np.percentile(gains, 95)),

        "fixed_vs_dense": {
            "wins": fixed_better_dense,
            "ties": fixed_equal_dense,
            "losses": fixed_worse_dense,
        },

        "oracle_vs_fixed": {
            "wins": oracle_better_fixed,
            "ties": oracle_equal_fixed,
        },

        "oracle_vs_dense": {
            "wins": oracle_better_dense,
            "ties": oracle_equal_dense,
            "losses": oracle_worse_dense,
        },

        "queries_where_lexical_is_required_for_oracle":
            lexical_required,
        "lexical_required_fraction":
            lexical_required / n,

        "representative_best_weight_counts": {
            f"{w:.1f}": weight_best_counts[w]
            for w in WEIGHTS
        },

        "oracle_tie_membership_counts": {
            f"{w:.1f}": weight_tie_counts[w]
            for w in WEIGHTS
        },

        "input": str(input_path),
    }

    per_query_path = output_dir / "oracle_per_query.csv"
    summary_path = output_dir / "oracle_summary.json"

    fieldnames = list(rows_out[0].keys())
    with per_query_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 72)
    print("DEVELOPMENT ORACLE WEIGHTED-RRF HEADROOM")
    print("=" * 72)
    print(f"Queries:              {n}")
    print(f"Dense mean nDCG@10:   {dense.mean():.4f}")
    print(f"Fixed 0.1/0.9:        {fixed.mean():.4f}")
    print(f"Oracle mean nDCG@10:  {oracle.mean():.4f}")
    print()
    print(
        "Oracle gain vs fixed: "
        f"{(oracle - fixed).mean():+.4f}"
    )
    print(
        "Oracle gain vs dense: "
        f"{(oracle - dense).mean():+.4f}"
    )
    print(
        "Gain vs fixed P50/P95: "
        f"{np.percentile(gains, 50):.4f} / "
        f"{np.percentile(gains, 95):.4f}"
    )
    print()
    print(
        "Fixed vs Dense W/T/L: "
        f"{fixed_better_dense}/"
        f"{fixed_equal_dense}/"
        f"{fixed_worse_dense}"
    )
    print(
        "Oracle vs Fixed W/T:  "
        f"{oracle_better_fixed}/"
        f"{oracle_equal_fixed}"
    )
    print(
        "Oracle vs Dense W/T/L:"
        f" {oracle_better_dense}/"
        f"{oracle_equal_dense}/"
        f"{oracle_worse_dense}"
    )
    print()
    print(
        "Lexical required for oracle: "
        f"{lexical_required}/{n} "
        f"({100 * lexical_required / n:.1f}%)"
    )
    print()
    print("Representative best lexical-weight distribution:")
    for w in WEIGHTS:
        count = weight_best_counts[w]
        print(
            f"  wL={w:.1f}: "
            f"{count:4d} ({100 * count / n:5.1f}%)"
        )
    print()
    print(f"Saved: {per_query_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
