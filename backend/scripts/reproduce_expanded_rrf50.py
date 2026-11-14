"""Recompute the paired 600-query RRF50 evaluation from saved measurements."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from backend.scripts.benchmark_candidate_budgets import FIELDS, MANIFEST, run


ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts/relevance/rrf50_expanded"
STRATEGIES = ("rrf50", "union")


def summarize(observations, protocol_bytes):
    protocol = json.loads(protocol_bytes)
    groups = {name: {} for name in STRATEGIES}
    for row in csv.DictReader(observations.decode().splitlines()):
        values = groups[row["strategy"]]
        index = int(row["row"])
        if index in values:
            raise ValueError(f"Duplicate observation {index}, {row['strategy']}")
        values[index] = [float(row[field]) for field in FIELDS[2:]]

    count = protocol["measured_count"]
    indices = range(1, count + 1)
    strategies = {}
    scores = {}
    for name, group in groups.items():
        if group.keys() != set(indices):
            raise ValueError(f"Expected {count} paired rows for {name}")
        values = np.asarray([group[index] for index in indices])
        ndcg, pairs, tokens, http_ms, inference_ms = values.T
        if (
            not np.isfinite(values).all()
            or np.any((ndcg < 0) | (ndcg > 1))
            or np.any(tokens <= 0)
            or np.any((pairs < 50) | (pairs > (50 if name == "rrf50" else 100)))
            or np.any((http_ms < 0) | (inference_ms < 0))
        ):
            raise ValueError(f"Invalid measurements for {name}")
        scores[name] = ndcg
        strategies[name] = {
            "mean_ndcg10": float(ndcg.mean()),
            "mean_pairs": float(pairs.mean()),
            "mean_unpadded_tokens": float(tokens.mean()),
            "http_p95_ms": float(np.percentile(http_ms, 95)),
            "inference_p95_ms": float(np.percentile(inference_ms, 95)),
        }

    difference = scores["rrf50"] - scores["union"]
    rng = np.random.default_rng(protocol["bootstrap_seed"])
    draws = rng.integers(0, count, size=(protocol["bootstrap_samples"], count))
    limits = np.percentile(difference[draws].mean(axis=1), [2.5, 97.5]).tolist()
    margin = protocol["nDCG10_loss_margin"]
    rrf50, union = (strategies[name] for name in STRATEGIES)
    return {
        "queries": count,
        "api_commit": protocol["api_commit"],
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "observations_sha256": hashlib.sha256(observations).hexdigest(),
        "strategy": strategies,
        "ndcg10_rrf50_minus_union": float(difference.mean()),
        "paired_95ci": limits,
        "nDCG10_loss_margin": margin,
        "quality_gate_passed": limits[0] >= -margin,
        "rrf50_token_reduction": 1 - rrf50["mean_unpadded_tokens"] / union["mean_unpadded_tokens"],
        "rrf50_http_p95_reduction": 1 - rrf50["http_p95_ms"] / union["http_p95_ms"],
        "wins": int((difference > 1e-12).sum()),
        "ties": int((np.abs(difference) <= 1e-12).sum()),
        "losses": int((difference < -1e-12).sum()),
    }


def verify_reference(actual, expected, field="summary"):
    if isinstance(expected, dict) and isinstance(actual, dict) and actual.keys() == expected.keys():
        for key, value in expected.items():
            verify_reference(actual[key], value, f"{field}.{key}")
    elif isinstance(expected, list) and isinstance(actual, list) and len(actual) == len(expected):
        for index, (value, target) in enumerate(zip(actual, expected, strict=True)):
            verify_reference(value, target, f"{field}[{index}]")
    elif isinstance(expected, float) and isinstance(actual, float):
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise SystemExit(f"{field}: expected {expected}, got {actual}")
    elif type(actual) is not type(expected) or actual != expected:
        raise SystemExit(f"{field}: expected {expected}, got {actual}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--observations", type=Path, help="Read metrics from a replay CSV.")
    source.add_argument("--base-url", help="Measure against the loopback benchmark API.")
    parser.add_argument("--output", type=Path, help="Save the observations from a live replay.")
    parser.add_argument("--check", action="store_true", help="Verify the archived summary and hashes.")
    args = parser.parse_args()

    protocol_path = ARTIFACTS / "protocol.json"
    protocol_bytes = protocol_path.read_bytes()
    observations = args.observations or ARTIFACTS / "observations.csv"
    if args.output and not args.base_url:
        parser.error("--output requires --base-url.")
    if args.base_url:
        if not args.output or args.check:
            parser.error("--base-url requires --output and cannot be combined with --check.")
        if args.output.exists():
            parser.error(f"Output already exists: {args.output}")
        rows, metadata = run(args.base_url, MANIFEST, protocol_path, strategies=STRATEGIES)
        protocol = json.loads(protocol_bytes)
        if metadata != {
            "device": protocol["device"], "model_batch_size": protocol["cross_encoder_batch_size"],
        }:
            raise SystemExit(f"Model settings differ from protocol: {metadata}")
        observations = args.output
        observations.parent.mkdir(parents=True, exist_ok=True)
        with observations.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    result = summarize(observations.read_bytes(), protocol_bytes)
    if args.check:
        protocol = json.loads(protocol_bytes)
        if hashlib.sha256(MANIFEST.read_bytes()).hexdigest() != protocol["manifest_sha256"]:
            raise SystemExit("Query manifest hash differs from the archived protocol.")
        reference = json.loads((ARTIFACTS / "reference.json").read_bytes())
        verify_reference(result, reference)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
