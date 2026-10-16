#!/usr/bin/env python3

import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path

from backend import retrieval_cpp


TOP_K = 10
CANDIDATE_K = 50

RRF_K = 60
LEXICAL_WEIGHT = 0.1
DENSE_WEIGHT = 0.9

CORPUS_PATH = Path("cpp/data/msmarco_full.bin")

DATASETS = {
    "500": {
        "manifest": Path(
            "artifacts/msmarco_full/eval_manifest.json"
        ),
        "candidates": Path(
            "results/msmarco_full/relevance/ce_candidates.json"
        ),
    },
    "10k": {
        "manifest": Path(
            "artifacts/msmarco_full/eval_manifest_10k.json"
        ),
        "candidates": Path(
            "results/msmarco_full/relevance/ce_candidates_10k.json"
        ),
    },
}


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def document_ids(candidates):
    return [
        int(item["document_id"])
        for item in candidates[:CANDIDATE_K]
    ]


def weighted_rrf(lexical_ids, dense_ids):
    scores = {}

    for rank, doc_id in enumerate(lexical_ids, 1):
        scores[doc_id] = (
            scores.get(doc_id, 0.0)
            + LEXICAL_WEIGHT / (RRF_K + rank)
        )

    for rank, doc_id in enumerate(dense_ids, 1):
        scores[doc_id] = (
            scores.get(doc_id, 0.0)
            + DENSE_WEIGHT / (RRF_K + rank)
        )

    ranked = sorted(
        scores,
        key=lambda doc_id: (-scores[doc_id], doc_id),
    )

    return ranked[:TOP_K]


def evaluate(retrieved, relevant):
    relevant = set(relevant)
    retrieved = retrieved[:TOP_K]

    rr = 0.0
    dcg = 0.0

    for rank, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant:
            if rr == 0.0:
                rr = 1.0 / rank

            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(relevant), TOP_K)

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    hits = len(set(retrieved) & relevant)

    return {
        "mrr_at_10": rr,
        "ndcg_at_10": dcg / idcg if idcg else 0.0,
        "recall_at_10": (
            hits / len(relevant) if relevant else 0.0
        ),
    }


def summarize(rows, method):
    metrics = [
        "mrr_at_10",
        "ndcg_at_10",
        "recall_at_10",
    ]

    return {
        metric: statistics.fmean(
            row[method][metric]
            for row in rows
        )
        for metric in metrics
    }


def percentile(values, p):
    values = sorted(values)

    if not values:
        return None

    position = (len(values) - 1) * p

    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return values[lower]

    weight = position - lower

    return (
        values[lower] * (1.0 - weight)
        + values[upper] * weight
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        default="500",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    config = DATASETS[args.dataset]

    manifest = load_json(config["manifest"])
    cache = load_json(config["candidates"])

    queries = manifest["queries"]

    if args.limit is not None:
        queries = queries[:args.limit]

    cache_queries = {
        str(item["query_id"]): item
        for item in cache["queries"]
    }

    if cache["protocol"]["candidate_k"] != CANDIDATE_K:
        raise RuntimeError("Candidate-K mismatch")

    if cache["protocol"]["queries"] != len(manifest["queries"]):
        raise RuntimeError("Query count mismatch")

    print("Loading C++ BM25 corpus...")

    engine = retrieval_cpp.SearchEngine()
    engine.load_corpus(str(CORPUS_PATH))

    print(f"Dataset: {args.dataset}")
    print(f"Queries: {len(queries)}")

    rows = []

    for index, query in enumerate(queries, 1):
        qid = str(query["query_id"])

        cached = cache_queries.get(qid)

        if cached is None:
            raise RuntimeError(
                f"Missing cached query: {qid}"
            )

        if cached["query"] != query["query"]:
            raise RuntimeError(
                f"Query text mismatch: {qid}"
            )

        relevant = set(
            int(x)
            for x in query["relevant_document_ids"]
        )

        pg_ids = document_ids(
            cached["lexical"]["candidates"]
        )

        dense_ids = document_ids(
            cached["dense"]["candidates"]
        )

        start = time.perf_counter()

        cpp_matches = engine.search_dense(
            query["query"],
            CANDIDATE_K,
        )

        cpp_ms = (
            time.perf_counter() - start
        ) * 1000

        cpp_ids = [
            int(match.doc_id)
            for match in cpp_matches
        ]

        pg_hybrid = weighted_rrf(
            pg_ids,
            dense_ids,
        )

        cpp_hybrid = weighted_rrf(
            cpp_ids,
            dense_ids,
        )

        rows.append({
            "query_id": qid,
            "query": query["query"],
            "postgresql_fts": evaluate(
                pg_ids[:TOP_K], relevant
            ),
            "cpp_bm25": evaluate(
                cpp_ids[:TOP_K], relevant
            ),
            "dense": evaluate(
                dense_ids[:TOP_K], relevant
            ),
            "postgresql_hybrid": evaluate(
                pg_hybrid, relevant
            ),
            "cpp_hybrid": evaluate(
                cpp_hybrid, relevant
            ),
            "cpp_latency_ms": cpp_ms,
        })

        if index % 50 == 0 or index == len(queries):
            print(
                f"[{index}/{len(queries)}] "
                f"C++ BM25={cpp_ms:.2f} ms"
            )

    methods = [
        "postgresql_fts",
        "cpp_bm25",
        "dense",
        "postgresql_hybrid",
        "cpp_hybrid",
    ]

    summary = {
        method: summarize(rows, method)
        for method in methods
    }

    latencies = [
        row["cpp_latency_ms"]
        for row in rows
    ]

    summary["cpp_latency"] = {
        "p50_ms": percentile(latencies, 0.50),
        "p95_ms": percentile(latencies, 0.95),
        "mean_ms": statistics.fmean(latencies),
    }

    output_dir = Path(
        "results/msmarco_full/relevance/cpp_v064"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = (
        f"{args.dataset}_{len(queries)}q"
    )

    json_path = output_dir / f"summary_{suffix}.json"
    csv_path = output_dir / f"per_query_{suffix}.csv"

    output = {
        "protocol": {
            "dataset": args.dataset,
            "queries": len(queries),
            "candidate_k": CANDIDATE_K,
            "top_k": TOP_K,
            "rrf_k": RRF_K,
            "lexical_weight": LEXICAL_WEIGHT,
            "dense_weight": DENSE_WEIGHT,
            "dense_cache_protocol": cache["protocol"],
            "qrels_source": manifest["qrels_source"],
        },
        "summary": summary,
    }

    json_path.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "query_id",
            "method",
            "mrr_at_10",
            "ndcg_at_10",
            "recall_at_10",
            "cpp_latency_ms",
        ])

        for row in rows:
            for method in methods:
                metrics = row[method]

                writer.writerow([
                    row["query_id"],
                    method,
                    metrics["mrr_at_10"],
                    metrics["ndcg_at_10"],
                    metrics["recall_at_10"],
                    row["cpp_latency_ms"]
                    if method == "cpp_bm25"
                    else "",
                ])

    print("\n=== Retrieval Quality ===")

    for method in methods:
        s = summary[method]

        print(
            f"{method:20s} "
            f"MRR@10={s['mrr_at_10']:.4f} "
            f"nDCG@10={s['ndcg_at_10']:.4f} "
            f"Recall@10={s['recall_at_10']:.4f}"
        )

    print("\nC++ BM25 Latency:")
    print(summary["cpp_latency"])

    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
