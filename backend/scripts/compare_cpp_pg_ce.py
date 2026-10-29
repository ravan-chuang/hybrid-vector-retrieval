import json
from pathlib import Path

import numpy as np


BASE = Path("results/msmarco_full/relevance")

CPP_PATH = (
    BASE / "cpp_v065/results_500_500q.json"
)

PG_PATH = (
    BASE / "ce_rerank_results.json"
)

METRICS = (
    "mrr_at_10",
    "ndcg_at_10",
    "recall_at_10",
)

BOOTSTRAP_SAMPLES = 10_000
SEED = 42


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    cpp = load_json(CPP_PATH)
    pg = load_json(PG_PATH)

    cpp_rows = {
        str(row["query_id"]): row
        for row in cpp["per_query"]
    }

    pg_rows = {
        str(row["query_id"]): row
        for row in pg["per_query"]
    }

    if set(cpp_rows) != set(pg_rows):
        raise RuntimeError("Query sets do not match")

    ids = sorted(cpp_rows)

    print(f"Matched queries: {len(ids)}")
    print()

    rng = np.random.default_rng(SEED)

    for metric in METRICS:
        cpp_values = np.array([
            cpp_rows[qid]["union_ce"]["metrics"][metric]
            for qid in ids
        ])

        pg_values = np.array([
            pg_rows[qid]["union_ce"]["metrics"][metric]
            for qid in ids
        ])

        differences = cpp_values - pg_values

        observed_delta = float(differences.mean())

        bootstrap_deltas = np.empty(
            BOOTSTRAP_SAMPLES,
            dtype=np.float64,
        )

        for i in range(BOOTSTRAP_SAMPLES):
            sample = rng.integers(
                0,
                len(differences),
                size=len(differences),
            )

            bootstrap_deltas[i] = (
                differences[sample].mean()
            )

        ci_low, ci_high = np.percentile(
            bootstrap_deltas,
            [2.5, 97.5],
        )

        wins = int(np.sum(differences > 1e-12))
        ties = int(np.sum(np.abs(differences) <= 1e-12))
        losses = int(np.sum(differences < -1e-12))

        print(metric)
        print(f"  C++ mean : {cpp_values.mean():.6f}")
        print(f"  PG mean  : {pg_values.mean():.6f}")
        print(f"  Delta    : {observed_delta:+.6f}")
        print(
            f"  95% CI   : "
            f"[{ci_low:+.6f}, {ci_high:+.6f}]"
        )
        print(
            f"  W/T/L    : "
            f"{wins}/{ties}/{losses}"
        )
        print()


if __name__ == "__main__":
    main()
