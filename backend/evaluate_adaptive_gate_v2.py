import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


SEED = 42
N_SPLITS = 5

ID = "query_id"
TARGET = "beneficial_0.1"

EVAL_COLUMNS = {
    "utility_status",
    "dense_ndcg@10",
    "fixed_0.1_ndcg@10",
    "delta_0.1_ndcg@10",
}

THRESHOLDS = np.arange(0.05, 0.951, 0.05)


def adaptive_ndcg(dense, fixed, prediction):
    values = np.where(
        prediction == 1,
        fixed,
        dense,
    )
    return float(values.mean())


def choose_threshold(y_prob, dense, fixed):
    """
    Select threshold using TRAINING DATA ONLY.

    Objective:
        maximize mean adaptive nDCG@10.

    Tie breaking:
        prefer the higher threshold, i.e. the more
        conservative routing policy.
    """
    best_threshold = None
    best_ndcg = -np.inf

    for threshold in THRESHOLDS:
        pred = (y_prob >= threshold).astype(int)
        score = adaptive_ndcg(
            dense,
            fixed,
            pred,
        )

        if (
            score > best_ndcg + 1e-12
            or (
                abs(score - best_ndcg) <= 1e-12
                and (
                    best_threshold is None
                    or threshold > best_threshold
                )
            )
        ):
            best_ndcg = score
            best_threshold = float(threshold)

    return best_threshold, best_ndcg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(args.input)

    feature_columns = [
        c for c in df.columns
        if c not in (
            {ID, TARGET} | EVAL_COLUMNS
        )
    ]

    forbidden = (
        "ndcg",
        "mrr",
        "recall",
        "relevant",
        "oracle",
        "utility",
        "target",
        "beneficial",
        "delta",
    )

    leaked = [
        c for c in feature_columns
        if any(
            word in c.lower()
            for word in forbidden
        )
    ]

    if leaked:
        raise ValueError(
            f"Potential leakage features: {leaked}"
        )

    X = df[feature_columns]
    y = df[TARGET].astype(int).to_numpy()

    dense = df[
        "dense_ndcg@10"
    ].astype(float).to_numpy()

    fixed = df[
        "fixed_0.1_ndcg@10"
    ].astype(float).to_numpy()

    delta = df[
        "delta_0.1_ndcg@10"
    ].astype(float).to_numpy()

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED,
    )

    model_factories = {
        "logistic_regression": lambda: make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=5000,
                random_state=SEED,
            ),
        ),

        "decision_tree": lambda: make_pipeline(
            SimpleImputer(strategy="median"),
            DecisionTreeClassifier(
                max_depth=3,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=SEED,
            ),
        ),
    }

    print("=" * 76)
    print("ADAPTIVE GATE V2 — 5-FOLD OOF UTILITY EVALUATION")
    print("=" * 76)
    print(f"Queries:      {len(df)}")
    print(f"Positive:     {int(y.sum())}")
    print(f"Negative:     {len(y) - int(y.sum())}")
    print(f"Folds:        {N_SPLITS}")
    print(f"Features:     {len(feature_columns)}")
    print()
    print("Binary-action reference:")
    print(f"  Dense:      {dense.mean():.4f}")
    print(f"  Fixed 0.1:  {fixed.mean():.4f}")

    binary_oracle = np.where(
        delta > 0,
        fixed,
        dense,
    )

    print(
        f"  Oracle:     "
        f"{binary_oracle.mean():.4f}"
    )
    print()

    all_results = {}

    for model_name, factory in model_factories.items():
        oof_prob = np.zeros(
            len(df),
            dtype=float,
        )
        oof_pred = np.zeros(
            len(df),
            dtype=int,
        )

        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(X, y),
            start=1,
        ):
            model = factory()

            model.fit(
                X.iloc[train_idx],
                y[train_idx],
            )

            train_prob = model.predict_proba(
                X.iloc[train_idx]
            )[:, 1]

            val_prob = model.predict_proba(
                X.iloc[val_idx]
            )[:, 1]

            # IMPORTANT:
            # Threshold is selected only on training fold.
            threshold, train_ndcg = choose_threshold(
                train_prob,
                dense[train_idx],
                fixed[train_idx],
            )

            val_pred = (
                val_prob >= threshold
            ).astype(int)

            oof_prob[val_idx] = val_prob
            oof_pred[val_idx] = val_pred

            val_ndcg = adaptive_ndcg(
                dense[val_idx],
                fixed[val_idx],
                val_pred,
            )

            fold_results.append({
                "fold": fold,
                "threshold": threshold,
                "train_adaptive_ndcg":
                    train_ndcg,
                "validation_adaptive_ndcg":
                    val_ndcg,
                "validation_dense_ndcg":
                    float(dense[val_idx].mean()),
                "validation_fixed_ndcg":
                    float(fixed[val_idx].mean()),
                "fusion_queries":
                    int(val_pred.sum()),
                "validation_queries":
                    len(val_idx),
            })

        oof_adaptive = np.where(
            oof_pred == 1,
            fixed,
            dense,
        )

        pr_auc = average_precision_score(
            y,
            oof_prob,
        )

        beneficial_routed = int(
            np.sum(
                (oof_pred == 1)
                & (delta > 1e-12)
            )
        )

        harmful_routed = int(
            np.sum(
                (oof_pred == 1)
                & (delta < -1e-12)
            )
        )

        neutral_routed = int(
            np.sum(
                (oof_pred == 1)
                & (np.abs(delta) <= 1e-12)
            )
        )

        result = {
            "model": model_name,
            "oof_pr_auc": float(pr_auc),

            "dense_ndcg@10":
                float(dense.mean()),

            "fixed_ndcg@10":
                float(fixed.mean()),

            "adaptive_oof_ndcg@10":
                float(oof_adaptive.mean()),

            "binary_action_oracle_ndcg@10":
                float(binary_oracle.mean()),

            "gain_vs_dense":
                float(
                    oof_adaptive.mean()
                    - dense.mean()
                ),

            "gain_vs_fixed":
                float(
                    oof_adaptive.mean()
                    - fixed.mean()
                ),

            "remaining_binary_oracle_headroom":
                float(
                    binary_oracle.mean()
                    - oof_adaptive.mean()
                ),

            "fusion_queries":
                int(oof_pred.sum()),

            "beneficial_routed":
                beneficial_routed,

            "harmful_routed":
                harmful_routed,

            "neutral_routed":
                neutral_routed,

            "folds":
                fold_results,
        }

        all_results[model_name] = result

        print("-" * 76)
        print(model_name.upper())
        print("-" * 76)

        for fr in fold_results:
            print(
                f"Fold {fr['fold']}: "
                f"threshold={fr['threshold']:.2f} | "
                f"val nDCG="
                f"{fr['validation_adaptive_ndcg']:.4f} | "
                f"fusion="
                f"{fr['fusion_queries']}/"
                f"{fr['validation_queries']}"
            )

        print()
        print(
            f"OOF PR-AUC:        "
            f"{pr_auc:.4f}"
        )
        print(
            f"OOF Adaptive nDCG: "
            f"{oof_adaptive.mean():.4f}"
        )
        print(
            f"Gain vs Dense:     "
            f"{oof_adaptive.mean() - dense.mean():+.4f}"
        )
        print(
            f"Gain vs Fixed:     "
            f"{oof_adaptive.mean() - fixed.mean():+.4f}"
        )
        print(
            f"Oracle headroom:   "
            f"{binary_oracle.mean() - oof_adaptive.mean():.4f}"
        )
        print(
            f"Fusion queries:    "
            f"{int(oof_pred.sum())}/"
            f"{len(df)}"
        )
        print(
            f"Beneficial routed: "
            f"{beneficial_routed}/128"
        )
        print(
            f"Harmful routed:    "
            f"{harmful_routed}/121"
        )
        print(
            f"Neutral routed:    "
            f"{neutral_routed}/751"
        )
        print()

    summary = {
        "protocol": {
            "type":
                "5-fold stratified out-of-fold",
            "seed": SEED,
            "folds": N_SPLITS,
            "threshold_grid":
                [float(x) for x in THRESHOLDS],
            "threshold_selection":
                "training fold only; maximize adaptive nDCG@10",
            "threshold_tie_break":
                "prefer higher conservative threshold",
            "adaptive_final_test_used":
                False,
            "action_0": "dense",
            "action_1":
                "weighted RRF wL=0.1 wD=0.9",
        },
        "feature_columns": feature_columns,
        "reference": {
            "dense_ndcg@10":
                float(dense.mean()),
            "fixed_ndcg@10":
                float(fixed.mean()),
            "binary_action_oracle_ndcg@10":
                float(binary_oracle.mean()),
        },
        "results": all_results,
    }

    path = output_dir / "oof_summary.json"

    with path.open("w") as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
