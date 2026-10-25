#!/usr/bin/env python3
"""Paired bootstrap evaluation for MS MARCO Passage Dev Small."""

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_DENSE = Path(
    "results/msmarco_full/relevance/"
    "dev_dense_ce/results_dev_6980q.json"
)

DEFAULT_HYBRID = Path(
    "results/msmarco_full/relevance/"
    "dev_v067/cpp_ce/results_dev_6980q.json"
)

DEFAULT_OUTPUT = Path(
    "results/msmarco_full/relevance/"
    "dev_dense_ce/paired_bootstrap.json"
)


def load_results(path, method):
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["per_query"]

    results = {}

    for row in rows:
        qid = str(row["query_id"])

        if qid in results:
            raise ValueError(f"Duplicate query ID: {qid}")

        results[qid] = {
            "ndcg": float(
                row[method]["metrics"]["ndcg_at_10"]
            ),
            "qrels": set(
                map(int, row["relevant_document_ids"])
            ),
        }

    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dense",
        type=Path,
        default=DEFAULT_DENSE,
    )

    parser.add_argument(
        "--hybrid",
        type=Path,
        default=DEFAULT_HYBRID,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if args.bootstrap <= 0:
        raise ValueError("--bootstrap must be positive")

    dense = load_results(args.dense, "dense_ce")
    hybrid = load_results(args.hybrid, "union_ce")

    if dense.keys() != hybrid.keys():
        raise ValueError("Query ID sets do not match")

    if len(dense) != 6980:
        raise ValueError(
            f"Expected 6980 queries, got {len(dense)}"
        )

    qids = sorted(dense)

    for qid in qids:
        if dense[qid]["qrels"] != hybrid[qid]["qrels"]:
            raise ValueError(
                f"Qrels mismatch for query {qid}"
            )

    dense_scores = np.array(
        [dense[qid]["ndcg"] for qid in qids],
        dtype=np.float64,
    )

    hybrid_scores = np.array(
        [hybrid[qid]["ndcg"] for qid in qids],
        dtype=np.float64,
    )

    differences = hybrid_scores - dense_scores

    observed = float(differences.mean())

    rng = np.random.default_rng(args.seed)

    bootstrap_means = np.empty(
        args.bootstrap,
        dtype=np.float64,
    )

    for i in range(args.bootstrap):
        indices = rng.integers(
            0,
            len(differences),
            size=len(differences),
        )

        bootstrap_means[i] = differences[indices].mean()

    ci_low, ci_high = np.percentile(
        bootstrap_means,
        [2.5, 97.5],
    )

    dense_mean = float(dense_scores.mean())
    hybrid_mean = float(hybrid_scores.mean())

    relative = (
        observed / dense_mean * 100
        if dense_mean != 0
        else None
    )

    report = {
        "dataset": "MS MARCO Passage Dev Small",
        "queries": len(qids),
        "metric": "ndcg_at_10",
        "comparison": "hybrid_ce_minus_dense_ce",
        "dense_source": str(args.dense),
        "hybrid_source": str(args.hybrid),
        "dense_mean": dense_mean,
        "hybrid_mean": hybrid_mean,
        "absolute_improvement": observed,
        "relative_improvement_percent": relative,
        "bootstrap": {
            "method": "paired_percentile",
            "samples": args.bootstrap,
            "seed": args.seed,
            "confidence_level": 0.95,
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "ci_excludes_zero": bool(
                ci_low > 0 or ci_high < 0
            ),
        },
        "per_query_comparison": {
            "hybrid_better": int(
                np.sum(differences > 0)
            ),
            "dense_better": int(
                np.sum(differences < 0)
            ),
            "ties": int(
                np.sum(differences == 0)
            ),
        },
        "limitations": [
            "Candidate-set sizes differ between configurations.",
            "Historical experiment versions may differ.",
            "The confidence interval does not isolate the causal effect of fusion.",
        ],
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    print("\n===== PAIRED BOOTSTRAP =====")
    print(f"Queries: {len(qids)}")
    print(f"Dense CE nDCG@10: {dense_mean:.6f}")
    print(f"Hybrid CE nDCG@10: {hybrid_mean:.6f}")
    print(f"Absolute improvement: {observed:+.6f}")
    print(f"Relative improvement: {relative:+.2f}%")
    print(
        f"95% CI: [{ci_low:+.6f}, {ci_high:+.6f}]"
    )
    print(
        "Hybrid better:",
        report["per_query_comparison"]["hybrid_better"],
    )
    print(
        "Dense better:",
        report["per_query_comparison"]["dense_better"],
    )
    print(
        "Ties:",
        report["per_query_comparison"]["ties"],
    )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
