import argparse
import json
import random
import re
import statistics
import time
from pathlib import Path

import numpy as np

from backend import retrieval_cpp
from backend.scripts.benchmark_stopword_ablation_v069 import (
    load_queries,
    load_qrels,
    mrr_at_10,
    ndcg_at_10,
    recall_at_k,
)

BASE = Path("results/msmarco_full/ablation/stopwords_v069")
OLD_PATH = BASE / "per_query.json"
OUT_PATH = BASE / "normalization_v069b.json"
SUMMARY_PATH = BASE / "comparison_v069b.json"

CORPUS_PATH = Path("cpp/data/msmarco_full.bin")
TOP_K = 50


def normalize_query(query):
    tokens = re.findall(
        r"[A-Za-z0-9]+",
        query.lower(),
    )
    return " ".join(tokens)


def search(engine, query):
    start = time.perf_counter_ns()

    if query.strip():
        matches = engine.search_dense(query, TOP_K)
        ids = [int(x.doc_id) for x in matches]
    else:
        ids = []

    elapsed_ms = (
        time.perf_counter_ns() - start
    ) / 1_000_000.0

    return ids, elapsed_ms


def summarize_latency(values):
    a = np.asarray(values, dtype=float)

    return {
        "count": len(a),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
    }


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def quality(ids, relevant):
    return {
        "top50": ids,
        "mrr10": mrr_at_10(ids, relevant),
        "ndcg10": ndcg_at_10(ids, relevant),
        "recall10": recall_at_k(ids, relevant, 10),
        "recall50": recall_at_k(ids, relevant, 50),
    }


def bootstrap_delta(rows, left, right, metric, seed=42):
    a = np.asarray(
        [row[left][metric] for row in rows],
        dtype=float,
    )
    b = np.asarray(
        [row[right][metric] for row in rows],
        dtype=float,
    )

    delta = b - a
    rng = np.random.default_rng(seed)
    means = np.empty(10000, dtype=float)

    for i in range(len(means)):
        sample = rng.integers(0, len(delta), len(delta))
        means[i] = delta[sample].mean()

    low, high = np.percentile(means, [2.5, 97.5])

    return {
        "left": left,
        "right": right,
        "metric": metric,
        "left_mean": float(a.mean()),
        "right_mean": float(b.mean()),
        "delta": float(delta.mean()),
        "ci95": [float(low), float(high)],
        "wins": int((delta > 0).sum()),
        "ties": int((delta == 0).sum()),
        "losses": int((delta < 0).sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--limit", type=int, default=6980)
    args = parser.parse_args()

    if args.repeats < 1 or args.limit < 1:
        parser.error("--repeats and --limit must be >= 1")

    if not OLD_PATH.exists():
        raise FileNotFoundError(
            f"Missing v0.6.9 results: {OLD_PATH}"
        )

    old_rows = json.loads(OLD_PATH.read_text(encoding="utf-8"))
    old_by_id = {
        str(row["query_id"]): row
        for row in old_rows
    }

    queries = load_queries()[:args.limit]
    qrels = load_qrels()

    if len(old_by_id) < len(queries):
        raise RuntimeError(
            "Existing v0.6.9 results are incomplete."
        )

    print("=" * 76)
    print("v0.6.9b — NORMALIZATION VS STOPWORD ABLATION")
    print("=" * 76)

    print("\nLoading C++ BM25 corpus...")
    engine = retrieval_cpp.SearchEngine()
    engine.load_corpus(str(CORPUS_PATH))

    print(f"Corpus documents: {engine.document_count():,}")
    print("Warming up...")

    for item in queries[:args.warmup]:
        search(engine, normalize_query(item["query"]))

    rng = random.Random(42)
    rows = []

    for index, item in enumerate(queries, 1):
        qid = str(item["query_id"])
        original = item["query"]
        normalized = normalize_query(original)

        old = old_by_id[qid]

        if old["query"] != original:
            raise RuntimeError(
                f"Query mismatch for QID={qid}"
            )

        relevant = qrels.get(qid, set())

        times = []
        ids = None

        for _ in range(args.repeats):
            current_ids, latency = search(engine, normalized)
            times.append(latency)

            if ids is None:
                ids = current_ids
            elif ids != current_ids:
                raise RuntimeError(
                    f"Non-deterministic results: QID={qid}"
                )

        normalized_result = quality(ids, relevant)
        normalized_result["latencies_ms"] = times

        row = {
            "query_id": qid,
            "query": original,
            "normalized_query": normalized,
            "original": old["off"],
            "normalized": normalized_result,
            "stopwords": old["on"],
            "overlap_original_normalized": len(
                set(old["off"]["top50"]) & set(ids)
            ),
            "overlap_normalized_stopwords": len(
                set(ids) & set(old["on"]["top50"])
            ),
        }

        rows.append(row)

        if index % 250 == 0 or index == len(queries):
            print(
                f"[{index}/{len(queries)}] completed",
                flush=True,
            )
            save_json(OUT_PATH, rows)

    metrics = ["mrr10", "ndcg10", "recall10", "recall50"]

    summary = {
        "protocol": {
            "dataset": "MS MARCO Dev Small",
            "queries": len(rows),
            "corpus_documents": engine.document_count(),
            "top_k": TOP_K,
            "new_repeats": args.repeats,
            "old_results": str(OLD_PATH),
            "new_results": str(OUT_PATH),
        },
        "variants": {},
        "comparisons": {},
        "mean_overlap_original_normalized": statistics.fmean(
            row["overlap_original_normalized"] for row in rows
        ),
        "mean_overlap_normalized_stopwords": statistics.fmean(
            row["overlap_normalized_stopwords"] for row in rows
        ),
    }

    for variant in ["original", "normalized", "stopwords"]:
        summary["variants"][variant] = {
            metric: statistics.fmean(
                row[variant][metric] for row in rows
            )
            for metric in metrics
        }

        latency_values = [
            latency
            for row in rows
            for latency in row[variant]["latencies_ms"]
        ]

        summary["variants"][variant]["latency_ms"] = (
            summarize_latency(latency_values)
        )

    for left, right in [
        ("original", "normalized"),
        ("normalized", "stopwords"),
        ("original", "stopwords"),
    ]:
        key = f"{left}_to_{right}"

        summary["comparisons"][key] = {
            metric: bootstrap_delta(
                rows, left, right, metric
            )
            for metric in metrics
        }

    save_json(SUMMARY_PATH, summary)

    print("\n" + "=" * 76)
    print("FINAL RESULTS")
    print("=" * 76)

    for variant, data in summary["variants"].items():
        latency = data["latency_ms"]

        print(f"\n[{variant.upper()}]")
        for metric in metrics:
            print(f"{metric:10s}: {data[metric]:.8f}")

        print(f"Mean latency: {latency['mean']:.4f} ms")
        print(f"P50 latency:  {latency['p50']:.4f} ms")
        print(f"P95 latency:  {latency['p95']:.4f} ms")
        print(f"P99 latency:  {latency['p99']:.4f} ms")

    print("\nPAIRED BOOTSTRAP 95% CI")

    for name, comparisons in summary["comparisons"].items():
        print(f"\n{name}")

        for metric, result in comparisons.items():
            print(
                f"  {metric:10s} "
                f"delta={result['delta']:+.8f} "
                f"CI=[{result['ci95'][0]:+.8f}, "
                f"{result['ci95'][1]:+.8f}] "
                f"W/T/L={result['wins']}/"
                f"{result['ties']}/"
                f"{result['losses']}"
            )

    print(f"\nSaved: {OUT_PATH}")
    print(f"Saved: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
