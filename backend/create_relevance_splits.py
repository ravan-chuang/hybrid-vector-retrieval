import json
import random
from pathlib import Path


RESULTS_DIR = Path("results/relevance")
QRELS_PATH = RESULTS_DIR / "eligible_qrels.jsonl"
BASELINE_PATH = (
    RESULTS_DIR
    / "baseline_equal_rrf"
    / "evaluation_queries.jsonl"
)
SPLIT_DIR = RESULTS_DIR / "splits"

SEED = 42
DEV_SIZE = 1000
HOLDOUT_SIZE = 500


def read_jsonl(path):
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def write_jsonl(path, rows):
    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def main():
    if not QRELS_PATH.exists():
        raise FileNotFoundError(
            f"Missing qrels cache: {QRELS_PATH}"
        )

    if not BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"Missing frozen baseline: {BASELINE_PATH}"
        )

    eligible = read_jsonl(QRELS_PATH)
    baseline = read_jsonl(BASELINE_PATH)

    eligible_by_id = {
        str(row["query_id"]): row
        for row in eligible
    }

    baseline_ids = {
        str(row["query_id"])
        for row in baseline
    }

    if len(baseline_ids) != len(baseline):
        raise ValueError(
            "Baseline contains duplicate query IDs"
        )

    missing_baseline = (
        baseline_ids
        - set(eligible_by_id)
    )

    if missing_baseline:
        raise ValueError(
            "Baseline contains query IDs "
            "not present in eligible qrels"
        )

    remaining_ids = sorted(
        (
            qid
            for qid in eligible_by_id
            if qid not in baseline_ids
        ),
        key=int,
    )

    if len(remaining_ids) < (
        DEV_SIZE + HOLDOUT_SIZE
    ):
        raise ValueError(
            "Not enough eligible queries "
            "for requested splits"
        )

    rng = random.Random(SEED)
    rng.shuffle(remaining_ids)

    dev_ids = set(
        remaining_ids[:DEV_SIZE]
    )

    holdout_ids = set(
        remaining_ids[
            DEV_SIZE:
            DEV_SIZE + HOLDOUT_SIZE
        ]
    )

    assert baseline_ids.isdisjoint(dev_ids)
    assert baseline_ids.isdisjoint(holdout_ids)
    assert dev_ids.isdisjoint(holdout_ids)

    development = [
        eligible_by_id[qid]
        for qid in sorted(
            dev_ids,
            key=int,
        )
    ]

    holdout = [
        eligible_by_id[qid]
        for qid in sorted(
            holdout_ids,
            key=int,
        )
    ]

    SPLIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dev_path = (
        SPLIT_DIR
        / "development_1000.jsonl"
    )

    holdout_path = (
        SPLIT_DIR
        / "holdout_500.jsonl"
    )

    write_jsonl(
        dev_path,
        development,
    )

    write_jsonl(
        holdout_path,
        holdout,
    )

    manifest = {
        "seed": SEED,
        "eligible_query_count":
            len(eligible),
        "baseline_query_count":
            len(baseline_ids),
        "development_query_count":
            len(dev_ids),
        "holdout_query_count":
            len(holdout_ids),
        "baseline_source":
            str(BASELINE_PATH),
        "development_file":
            str(dev_path),
        "holdout_file":
            str(holdout_path),
        "pairwise_disjoint": True,
    }

    manifest_path = (
        SPLIT_DIR
        / "manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 70)
    print("RELEVANCE SPLITS FROZEN")
    print("=" * 70)
    print(
        f"Eligible:    {len(eligible):,}"
    )
    print(
        f"Baseline:    {len(baseline_ids):,}"
    )
    print(
        f"Development: {len(dev_ids):,}"
    )
    print(
        f"Holdout:     {len(holdout_ids):,}"
    )
    print()
    print(
        "Baseline ∩ Development: "
        f"{len(baseline_ids & dev_ids)}"
    )
    print(
        "Baseline ∩ Holdout:     "
        f"{len(baseline_ids & holdout_ids)}"
    )
    print(
        "Development ∩ Holdout:  "
        f"{len(dev_ids & holdout_ids)}"
    )
    print()
    print(f"Saved: {dev_path}")
    print(f"Saved: {holdout_path}")
    print(f"Saved: {manifest_path}")


if __name__ == "__main__":
    main()
