import json
import random
from pathlib import Path


SEED = 20261008
FINAL_SIZE = 500

ELIGIBLE_PATH = Path("results/relevance/eligible_qrels.jsonl")
BASELINE_PATH = Path(
    "results/relevance/baseline_equal_rrf/evaluation_queries.jsonl"
)
DEVELOPMENT_PATH = Path(
    "results/relevance/splits/development_1000.jsonl"
)
FIXED_HOLDOUT_PATH = Path(
    "results/relevance/splits/holdout_500.jsonl"
)

OUTPUT_DIR = Path("results/relevance/splits")
OUTPUT_PATH = OUTPUT_DIR / "adaptive_final_500.jsonl"
MANIFEST_PATH = OUTPUT_DIR / "adaptive_final_manifest.json"


def load_jsonl(path):
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def qid(row):
    return str(row["query_id"])


def main():
    eligible = load_jsonl(ELIGIBLE_PATH)
    baseline = load_jsonl(BASELINE_PATH)
    development = load_jsonl(DEVELOPMENT_PATH)
    fixed_holdout = load_jsonl(FIXED_HOLDOUT_PATH)

    eligible_ids = {qid(row) for row in eligible}
    baseline_ids = {qid(row) for row in baseline}
    development_ids = {qid(row) for row in development}
    fixed_holdout_ids = {qid(row) for row in fixed_holdout}

    assert len(eligible_ids) == len(eligible)
    assert len(baseline_ids) == len(baseline)
    assert len(development_ids) == len(development)
    assert len(fixed_holdout_ids) == len(fixed_holdout)

    # Existing partitions must already be mutually disjoint.
    assert baseline_ids.isdisjoint(development_ids)
    assert baseline_ids.isdisjoint(fixed_holdout_ids)
    assert development_ids.isdisjoint(fixed_holdout_ids)

    used_ids = (
        baseline_ids
        | development_ids
        | fixed_holdout_ids
    )

    remaining = [
        row for row in eligible
        if qid(row) not in used_ids
    ]

    rng = random.Random(SEED)
    rng.shuffle(remaining)

    adaptive_final = remaining[:FINAL_SIZE]
    adaptive_final_ids = {qid(row) for row in adaptive_final}

    assert len(adaptive_final) == FINAL_SIZE
    assert len(adaptive_final_ids) == FINAL_SIZE

    assert adaptive_final_ids.isdisjoint(baseline_ids)
    assert adaptive_final_ids.isdisjoint(development_ids)
    assert adaptive_final_ids.isdisjoint(fixed_holdout_ids)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w") as f:
        for row in adaptive_final:
            f.write(json.dumps(row) + "\n")

    manifest = {
        "purpose": (
            "Untouched one-shot final evaluation split "
            "for adaptive hybrid retrieval"
        ),
        "seed": SEED,
        "eligible_query_count": len(eligible),
        "baseline_query_count": len(baseline),
        "development_query_count": len(development),
        "fixed_weight_holdout_query_count": len(fixed_holdout),
        "used_before_adaptive_final": len(used_ids),
        "remaining_before_sampling": len(remaining),
        "adaptive_final_query_count": len(adaptive_final),
        "adaptive_final_file": str(OUTPUT_PATH),
        "constraints": {
            "disjoint_from_baseline": True,
            "disjoint_from_development": True,
            "disjoint_from_fixed_weight_holdout": True,
            "must_not_be_evaluated_before_model_freeze": True,
        },
    }

    with MANIFEST_PATH.open("w") as f:
        json.dump(manifest, f, indent=2)

    print("=" * 72)
    print("ADAPTIVE FINAL TEST SPLIT FROZEN")
    print("=" * 72)
    print(f"Eligible:             {len(eligible):,}")
    print(f"Baseline:             {len(baseline):,}")
    print(f"Development:          {len(development):,}")
    print(f"Fixed holdout:        {len(fixed_holdout):,}")
    print(f"Previously used:      {len(used_ids):,}")
    print(f"Remaining pool:       {len(remaining):,}")
    print(f"Adaptive final:       {len(adaptive_final):,}")
    print()
    print(
        "Baseline ∩ Adaptive:    ",
        len(baseline_ids & adaptive_final_ids),
    )
    print(
        "Development ∩ Adaptive: ",
        len(development_ids & adaptive_final_ids),
    )
    print(
        "Holdout ∩ Adaptive:     ",
        len(fixed_holdout_ids & adaptive_final_ids),
    )
    print()
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Saved: {MANIFEST_PATH}")
    print()
    print(
        "IMPORTANT: Do not run relevance evaluation on "
        "adaptive_final_500.jsonl until the adaptive model is frozen."
    )


if __name__ == "__main__":
    main()
