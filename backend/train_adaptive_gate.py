import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


SEED = 42
TEST_SIZE = 0.20

TARGET = "lexical_required"
ID_COLUMN = "query_id"

FORBIDDEN_PATTERNS = (
    "ndcg",
    "mrr",
    "recall",
    "relevant",
    "oracle",
    "best_weight",
)

# Retrieval decision for Gate v1:
# 0 -> Dense
# 1 -> development-selected fixed fusion (wL=0.1, wD=0.9)
FIXED_WEIGHT_COLUMN = "wL_0.1_ndcg@10"
DENSE_COLUMN = "dense_ndcg@10"
ORACLE_COLUMN = "oracle_ndcg@10"


def load_oracle(path):
    with open(path, newline="") as f:
        return {
            row["query_id"]: row
            for row in csv.DictReader(f)
        }


def evaluate_gate(name, y_true, y_prob, y_pred, qids, oracle):
    dense = np.array([
        float(oracle[qid][DENSE_COLUMN])
        for qid in qids
    ])

    fixed = np.array([
        float(oracle[qid][FIXED_WEIGHT_COLUMN])
        for qid in qids
    ])

    oracle_ndcg = np.array([
        float(oracle[qid][ORACLE_COLUMN])
        for qid in qids
    ])

    # Gate chooses between Dense and fixed 0.1/0.9 fusion.
    adaptive = np.where(
        y_pred == 1,
        fixed,
        dense,
    )

    result = {
        "model": name,
        "queries": len(qids),

        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "precision_positive": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall_positive": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "f1_positive": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                y_prob,
            )
        ),

        "predicted_fusion_queries":
            int(np.sum(y_pred == 1)),

        "dense_ndcg@10":
            float(dense.mean()),

        "fixed_0.1_ndcg@10":
            float(fixed.mean()),

        "adaptive_ndcg@10":
            float(adaptive.mean()),

        "oracle_ndcg@10":
            float(oracle_ndcg.mean()),

        "adaptive_gain_vs_dense":
            float(
                adaptive.mean() - dense.mean()
            ),

        "adaptive_gain_vs_fixed":
            float(
                adaptive.mean() - fixed.mean()
            ),

        "remaining_oracle_headroom":
            float(
                oracle_ndcg.mean()
                - adaptive.mean()
            ),

        "confusion_matrix":
            confusion_matrix(
                y_true,
                y_pred,
                labels=[0, 1],
            ).tolist(),
    }

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        required=True,
    )
    parser.add_argument(
        "--oracle",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(args.features)
    oracle = load_oracle(args.oracle)

    if len(df) != 1000:
        raise ValueError(
            f"Expected 1000 development rows, got {len(df)}"
        )

    if set(df[ID_COLUMN].astype(str)) != set(oracle):
        raise ValueError(
            "Feature/oracle query IDs do not match"
        )

    feature_columns = [
        c for c in df.columns
        if c not in {
            ID_COLUMN,
            TARGET,
        }
    ]

    leaked = [
        c for c in feature_columns
        if any(
            pattern in c.lower()
            for pattern in FORBIDDEN_PATTERNS
        )
    ]

    if leaked:
        raise ValueError(
            f"Potential target leakage: {leaked}"
        )

    X = df[feature_columns].copy()
    y = df[TARGET].astype(int)
    qids = df[ID_COLUMN].astype(str)

    (
        X_train,
        X_val,
        y_train,
        y_val,
        qid_train,
        qid_val,
    ) = train_test_split(
        X,
        y,
        qids,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )

    print("=" * 72)
    print("ADAPTIVE GATE V1 — DEVELOPMENT VALIDATION")
    print("=" * 72)
    print(f"Train queries:      {len(X_train)}")
    print(f"Validation queries: {len(X_val)}")
    print(
        "Train positives:    "
        f"{int(y_train.sum())}"
    )
    print(
        "Validation positives:"
        f" {int(y_val.sum())}"
    )
    print()
    print("Features:")
    for c in feature_columns:
        print(f"  {c}")

    models = {
        "logistic_regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=5000,
                random_state=SEED,
            ),
        ),

        "decision_tree": make_pipeline(
            SimpleImputer(strategy="median"),
            DecisionTreeClassifier(
                max_depth=3,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=SEED,
            ),
        ),
    }

    results = []

    # Always Dense baseline.
    dense_pred = np.zeros(
        len(y_val),
        dtype=int,
    )
    dense_prob = np.zeros(
        len(y_val),
        dtype=float,
    )

    results.append(
        evaluate_gate(
            "always_dense",
            y_val.to_numpy(),
            dense_prob,
            dense_pred,
            qid_val.tolist(),
            oracle,
        )
    )

    for name, model in models.items():
        model.fit(
            X_train,
            y_train,
        )

        prob = model.predict_proba(
            X_val
        )[:, 1]

        # v1 intentionally uses the standard frozen 0.5
        # classification threshold. No threshold tuning yet.
        pred = (
            prob >= 0.5
        ).astype(int)

        result = evaluate_gate(
            name,
            y_val.to_numpy(),
            prob,
            pred,
            qid_val.tolist(),
            oracle,
        )

        results.append(result)

        print()
        print("-" * 72)
        print(name.upper())
        print("-" * 72)
        print(
            classification_report(
                y_val,
                pred,
                digits=4,
                zero_division=0,
            )
        )

    print()
    print("=" * 72)
    print("RETRIEVAL-LEVEL VALIDATION RESULTS")
    print("=" * 72)

    for r in results:
        print()
        print(r["model"])
        print(
            f"  Accuracy:       "
            f"{r['accuracy']:.4f}"
        )
        print(
            f"  Positive P/R/F1:"
            f" {r['precision_positive']:.4f} /"
            f" {r['recall_positive']:.4f} /"
            f" {r['f1_positive']:.4f}"
        )
        print(
            f"  PR-AUC:         "
            f"{r['pr_auc']:.4f}"
        )
        print(
            f"  Fusion queries: "
            f"{r['predicted_fusion_queries']}"
        )
        print(
            f"  Dense nDCG:     "
            f"{r['dense_ndcg@10']:.4f}"
        )
        print(
            f"  Fixed nDCG:     "
            f"{r['fixed_0.1_ndcg@10']:.4f}"
        )
        print(
            f"  Adaptive nDCG:  "
            f"{r['adaptive_ndcg@10']:.4f}"
        )
        print(
            f"  Oracle nDCG:    "
            f"{r['oracle_ndcg@10']:.4f}"
        )
        print(
            f"  Gain vs Dense:  "
            f"{r['adaptive_gain_vs_dense']:+.4f}"
        )
        print(
            f"  Gain vs Fixed:  "
            f"{r['adaptive_gain_vs_fixed']:+.4f}"
        )
        print(
            f"  Headroom left:  "
            f"{r['remaining_oracle_headroom']:.4f}"
        )
        print(
            f"  Confusion:      "
            f"{r['confusion_matrix']}"
        )

    summary = {
        "protocol": {
            "development_queries": len(df),
            "train_queries": len(X_train),
            "validation_queries": len(X_val),
            "split_seed": SEED,
            "stratified": True,
            "threshold": 0.5,
            "adaptive_final_test_used": False,
            "gate_action_0": "dense",
            "gate_action_1":
                "fixed weighted RRF wL=0.1 wD=0.9",
        },
        "feature_columns": feature_columns,
        "results": results,
    }

    summary_path = (
        output_dir / "validation_summary.json"
    )

    with summary_path.open("w") as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
