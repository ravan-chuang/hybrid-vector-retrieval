"""Compare four Cross-Encoder candidate budgets with matched dev-small queries."""

import argparse
import csv
import hashlib
import json
import math
from itertools import permutations
from pathlib import Path
from time import perf_counter
from urllib.parse import urlsplit

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts/relevance/candidate_budgets"
MANIFEST = ROOT / "artifacts/msmarco_full/dev_small_manifest.json"
STRATEGIES = ("rrf40", "quota30+10", "rrf50", "union")
BOOTSTRAP_SEED = 5013
BOOTSTRAP_SAMPLES = 10_000
MARGIN = 0.005
FIELDS = ("row", "strategy", "ndcg10", "pairs", "tokens", "http_ms", "inference_ms")


def ndcg(ranking, relevant):
    discounts = [1 / math.log2(rank + 2) for rank in range(10)]
    return (
        sum(discount for pid, discount in zip(ranking, discounts) if pid in relevant)
        / sum(discounts[:len(relevant)])
    )


def summarize(rows):
    groups = {strategy: {} for strategy in STRATEGIES}
    for row in rows:
        group = groups[row["strategy"]]
        index = int(row["row"])
        if index in group:
            raise ValueError(f"Duplicate observation for {index}, {row['strategy']}")
        group[index] = row
    keys = sorted(groups["union"])
    expected = set(range(1, len(keys) + 1))
    if not keys or any(expected != group.keys() for group in groups.values()):
        raise ValueError("Each sample row must have one observation for every strategy.")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(keys), size=(BOOTSTRAP_SAMPLES, len(keys)))
    union = np.array([float(groups["union"][i]["ndcg10"]) for i in keys])
    rrf50 = np.array([float(groups["rrf50"][i]["ndcg10"]) for i in keys])
    output = {}
    for strategy, group in groups.items():
        values = [group[i] for i in keys]
        scores = np.array([float(row["ndcg10"]) for row in values])
        result = {"ndcg10": float(scores.mean())}
        for name, baseline in (("union", union), ("rrf50", rrf50)):
            difference = scores - baseline
            limits = np.quantile(difference[draws].mean(axis=1), [0.025, 0.975])
            result[f"delta_vs_{name}"] = float(difference.mean())
            result[f"ci95_vs_{name}"] = limits.tolist()
        result["mean_pairs"] = float(np.mean([int(row["pairs"]) for row in values]))
        result["mean_tokens"] = float(np.mean([int(row["tokens"]) for row in values]))
        for field in ("http_ms", "inference_ms"):
            result[f"p95_{field}"] = float(np.quantile([float(row[field]) for row in values], 0.95))
        output[strategy] = result
    return output


def run(base_url, manifest_path, sample_path):
    import httpx

    if urlsplit(base_url).hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Use the local benchmark server.")
    spec = json.loads(sample_path.read_text(encoding="utf-8"))
    raw_manifest = manifest_path.read_bytes()
    if hashlib.sha256(raw_manifest).hexdigest() != spec["manifest_sha256"]:
        raise ValueError("Query manifest hash differs from the recorded sample.")
    queries = json.loads(raw_manifest)["queries"]
    indices = spec["manifest_row_indices"]
    if len(set(indices)) != len(indices) or any(not 0 <= i < len(queries) for i in indices):
        raise ValueError("Sample indices must be distinct positions in the query manifest.")
    sample = [queries[i] for i in indices]
    warmup = spec["warmup_count"]
    if not 0 <= warmup < len(sample):
        raise ValueError("The sample must include measured queries after warmup.")
    orders = list(permutations(STRATEGIES))
    observed = []
    metadata = {}

    with httpx.Client(base_url=base_url, timeout=60, trust_env=False) as client:
        def call(path, payload):
            started = perf_counter()
            response = client.post(path, json=payload)
            elapsed = (perf_counter() - started) * 1000
            response.raise_for_status()
            return response.json(), elapsed

        for index, row in enumerate(sample):
            query = row["query"]
            order = orders[(index - warmup) % len(orders)]
            responses = {
                strategy: call("/_budget/compare", {"query": query, "strategy": strategy})
                for strategy in order
            }
            baseline = responses["union"][0]
            same_candidates = all(
                entry["lexical"] == baseline["lexical"] and entry["dense"] == baseline["dense"]
                for entry, _ in responses.values()
            )
            if not same_candidates:
                raise RuntimeError(f"Candidate retrieval differed within sample row {index}.")
            if index >= warmup:
                observed.append(responses)
                settings = {
                    "device": baseline["device"],
                    "model_batch_size": baseline["model_batch_size"],
                }
                if metadata and settings != metadata:
                    raise RuntimeError("Model settings changed during the benchmark.")
                metadata = settings
            if (index + 1) % 24 == 0:
                print(f"Timed {index + 1}/{len(sample)} queries", flush=True)

        records = []
        for index, (row, responses) in enumerate(zip(sample[warmup:], observed, strict=True), 1):
            tokens, _ = call("/_budget/tokens", {"query": row["query"]})
            for strategy in STRATEGIES:
                entry, elapsed = responses[strategy]
                if tokens[strategy]["candidates"] != entry["candidates"]:
                    raise RuntimeError(f"Candidate IDs changed during token counting on row {index}.")
                records.append({
                    "row": index,
                    "strategy": strategy,
                    "ndcg10": ndcg(entry["top10"], set(row["relevant_document_ids"])),
                    "pairs": len(entry["candidates"]),
                    "tokens": tokens[strategy]["tokens"],
                    "http_ms": elapsed,
                    "inference_ms": entry["inference_ms"],
                })
    return records, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--base-url", help="URL of the loopback-only benchmark server.")
    source.add_argument("--summarize", type=Path, help="Recompute metrics from a numeric observations CSV.")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--sample", type=Path, default=ARTIFACTS / "sample.json")
    parser.add_argument("--output", type=Path, help="Write numeric observations from a live run.")
    args = parser.parse_args()

    if args.summarize:
        with args.summarize.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        metadata = {}
    else:
        if not args.output:
            parser.error("--base-url requires --output")
        rows, metadata = run(args.base_url, args.manifest, args.sample)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({
        "queries": len(rows) // len(STRATEGIES),
        "provisional_margin": MARGIN,
        **metadata,
        "strategies": summarize(rows),
    }, indent=2))


if __name__ == "__main__":
    main()
