"""Replay paired relevance inference on the four historical dev-small pipelines."""

import argparse
import gzip
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from backend.scripts.reproduce_candidate_contribution import verify
from backend.scripts.reproduce_candidate_coverage import checked_bytes, rank_metrics


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "artifacts/evaluations/dev-small-bootstrap"
MANIFEST = ROOT / "artifacts/msmarco_full/dev_small_manifest.json"
METRICS = ("mrr_at_10", "ndcg_at_10", "recall_at_10")
METHODS = ("bm25", "dense", "rrf", "hybrid_ce")
METHOD_PAIRS = tuple(combinations(range(len(METHODS)), 2))
BATCH = 100


def load_scores(source):
    manifest = json.loads(checked_bytes(MANIFEST, source["manifest_sha256"]))
    qrels = {
        str(row["query_id"]): len(set(row["relevant_document_ids"]))
        for row in manifest["queries"]
    }
    if len(manifest["queries"]) != len(qrels) or len(qrels) != source["query_count"]:
        raise ValueError("Manifest query count or IDs differ from the archived protocol")
    if not all(qrels.values()) or tuple(source["methods"]) != METHODS:
        raise ValueError("Invalid relevance judgments or method order")

    scores = {}
    packed = checked_bytes(ARCHIVE / "hit_ranks.jsonl.gz", source["hit_ranks_sha256"])
    for line in gzip.decompress(packed).splitlines():
        query_id, *rankings = json.loads(line)
        if query_id not in qrels or query_id in scores or len(rankings) != len(METHODS):
            raise ValueError(f"Invalid observation for query {query_id}")
        relevant = qrels[query_id]
        metrics = []
        for hits in rankings:
            if (len(hits) > relevant or hits != sorted(set(hits))
                    or any(type(rank) is not int or not 1 <= rank <= 10 for rank in hits)):
                raise ValueError(f"Invalid hit ranks for query {query_id}")
            row = rank_metrics(hits, relevant)
            metrics.append([row[metric] for metric in METRICS])
        scores[query_id] = metrics

    if scores.keys() != qrels.keys():
        raise ValueError("Observations do not cover every manifest query")
    return scores


def differences(scores, query_ids):
    matrix = np.asarray([scores[query_id] for query_id in query_ids])
    deltas = np.column_stack([
        matrix[:, a, metric] - matrix[:, b, metric]
        for metric in range(len(METRICS))
        for a, b in METHOD_PAIRS
    ])
    return matrix, deltas


def bootstrap_intervals(deltas, resamples, seed):
    rng = np.random.default_rng(seed)
    n, count = deltas.shape
    samples = np.empty((resamples, count), dtype=np.float64)
    for start in range(0, resamples, BATCH):
        end = min(start + BATCH, resamples)
        indices = rng.integers(n, size=(end - start, n), dtype=np.int64)
        for j in range(count):
            samples[start:end, j] = deltas[:, j][indices].mean(axis=1)
    return np.quantile(samples, [0.025, 0.975], axis=0).T


def permutation_p_values(deltas, resamples, seed):
    rng = np.random.default_rng(seed)
    n = len(deltas)
    extremes = []
    for delta in deltas.T:
        observed = abs(float(delta.mean()))
        extreme = 0
        for start in range(0, resamples, BATCH):
            size = min(BATCH, resamples - start)
            signs = 2 * rng.integers(0, 2, size=(size, n), dtype=np.int8) - 1
            permuted = (signs @ delta) / n
            extreme += int(np.count_nonzero(np.abs(permuted) >= observed - 1e-15))
        extremes.append(extreme)
    p_values = (np.asarray(extremes) + 1) / (resamples + 1)
    return extremes, p_values


def holm_adjust(p_values):
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    scaled = p_values[order] * np.arange(len(order), 0, -1)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(scaled))
    return adjusted


def replay(source):
    scores = load_scores(source)
    bootstrap = source["bootstrap"]
    permutation = source["permutation"]
    if (bootstrap["query_order"] != "lexicographic"
            or permutation["query_order"] != "numeric" or permutation["holm_family"] != "all_18"):
        raise ValueError("Incompatible query ordering or multiple-testing family")
    matrix, boot_deltas = differences(scores, sorted(scores))
    _, permutation_deltas = differences(scores, sorted(scores, key=int))
    intervals = bootstrap_intervals(boot_deltas, bootstrap["resamples"], bootstrap["seed"])
    extremes, p_values = permutation_p_values(
        permutation_deltas, permutation["resamples"], permutation["seed"],
    )
    corrected = holm_adjust(p_values)
    comparisons = [
        {"metric": metric, "method_a": METHODS[a], "method_b": METHODS[b]}
        for metric in METRICS for a, b in METHOD_PAIRS
    ]
    deltas = boot_deltas.mean(axis=0)
    for j, row in enumerate(comparisons):
        row.update({
            "delta": float(deltas[j]),
            "ci95": intervals[j].tolist(),
            "extreme_permutations": extremes[j],
            "permutation_p": float(p_values[j]),
            "holm_p": float(corrected[j]),
        })
    means = {
        method: dict(zip(METRICS, values))
        for method, values in zip(METHODS, matrix.mean(axis=0).tolist())
    }
    return {"query_count": len(scores), "means": means, "comparisons": comparisons}


def check_summary(actual, expected):
    for key in ("query_count", "means"):
        verify(actual[key], expected[key], path=key)
    if len(actual["comparisons"]) != len(expected["comparisons"]):
        raise ValueError("Comparison count differs from the archived reference")
    for index, (row, reference) in enumerate(zip(actual["comparisons"], expected["comparisons"])):
        # The shared verifier compares dictionaries recursively and floats with a tolerance.
        actual_fields = {**row, "ci95": dict(enumerate(row["ci95"]))}
        expected_fields = {**reference, "ci95": dict(enumerate(reference["ci95"]))}
        verify(actual_fields, expected_fields, path=f"comparison {index}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify the 18 archived comparisons")
    args = parser.parse_args()
    source = json.loads((ARCHIVE / "source.json").read_text())
    actual = replay(source)
    if args.check:
        expected = json.loads((ARCHIVE / "expected.json").read_text())
        check_summary(actual, expected)
        print(f"Verified {len(actual['comparisons'])} paired comparisons across {actual['query_count']:,} queries.")
    else:
        print(json.dumps(actual, indent=2))


if __name__ == "__main__":
    main()
