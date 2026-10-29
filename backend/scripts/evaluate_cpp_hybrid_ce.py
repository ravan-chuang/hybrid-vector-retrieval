#!/usr/bin/env python3

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path

import numpy as np
import psycopg
import torch
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

from backend import retrieval_cpp


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CORPUS_PATH = Path("cpp/data/msmarco_full.bin")
OUTPUT_DIR = Path("results/msmarco_full/relevance/cpp_v065")

TOP_K = 10
CANDIDATE_K = 50
CE_BATCH_SIZE = 64

RRF_K = 60
LEXICAL_WEIGHT = 0.1
DENSE_WEIGHT = 0.9

DATASETS = {
    "500": {
        "manifest": Path(
            "artifacts/msmarco_full/eval_manifest.json"
        ),
        "cache": Path(
            "results/msmarco_full/relevance/ce_candidates.json"
        ),
    },
    "10k": {
        "manifest": Path(
            "artifacts/msmarco_full/eval_manifest_10k.json"
        ),
        "cache": Path(
            "results/msmarco_full/relevance/ce_candidates_10k.json"
        ),
    },
    "dev": {
        "manifest": Path(
            "artifacts/msmarco_full/dev_small_manifest.json"
        ),
        "cache": Path(
            "results/msmarco_full/relevance/"
            "dev_v067/ce_candidates_dev.json"
        ),
    },
}


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(path.suffix + ".tmp")

    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    temporary.replace(path)


def candidate_ids(candidates):
    return [
        int(item["document_id"])
        for item in candidates[:CANDIDATE_K]
    ]


def union_candidates(dense_ids, cpp_ids):
    return list(dict.fromkeys(dense_ids + cpp_ids))


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

    return sorted(
        scores,
        key=lambda doc_id: (-scores[doc_id], doc_id),
    )[:TOP_K]


def evaluate(ranked, relevant):
    relevant = set(relevant)
    ranked = ranked[:TOP_K]

    reciprocal_rank = 0.0
    dcg = 0.0

    for rank, doc_id in enumerate(ranked, 1):
        if doc_id in relevant:
            if reciprocal_rank == 0.0:
                reciprocal_rank = 1.0 / rank

            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(relevant), TOP_K)

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    hits = len(set(ranked) & relevant)

    return {
        "mrr_at_10": reciprocal_rank,
        "ndcg_at_10": dcg / idcg if idcg else 0.0,
        "recall_at_10": (
            hits / len(relevant) if relevant else 0.0
        ),
    }


def percentile(values, p):
    if not values:
        return 0.0

    return float(np.percentile(values, p))


def summarize(rows, method):
    metrics = [
        "mrr_at_10",
        "ndcg_at_10",
        "recall_at_10",
    ]

    return {
        metric: statistics.fmean(
            row[method]["metrics"][metric]
            for row in rows
        )
        for metric in metrics
    }


def fetch_missing_documents(conn, ids):
    if not ids:
        return {}

    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT document_id, content
            FROM public.msmarco_full_documents
            WHERE document_id = ANY(%s)
            """,
            (ids,),
        )

        return {
            int(doc_id): content
            for doc_id, content in cursor.fetchall()
        }


def synchronize_mps(device):
    if device == "mps":
        torch.mps.synchronize()


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

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    if args.checkpoint_every <= 0:
        parser.error("--checkpoint-every must be positive")

    load_dotenv(".env")

    config = DATASETS[args.dataset]

    manifest = load_json(config["manifest"])
    cache = load_json(config["cache"])

    queries = manifest["queries"]

    if args.limit is not None:
        queries = queries[:args.limit]

    cached_queries = {
        str(item["query_id"]): item
        for item in cache["queries"]
    }

    if cache["protocol"]["candidate_k"] != CANDIDATE_K:
        raise RuntimeError("Candidate-K mismatch")

    if cache["protocol"]["queries"] != len(manifest["queries"]):
        raise RuntimeError("Cache query count mismatch")

    for query in queries:
        qid = str(query["query_id"])

        cached = cached_queries.get(qid)

        if cached is None:
            raise RuntimeError(f"Missing cached query: {qid}")

        if cached["query"] != query["query"]:
            raise RuntimeError(f"Query text mismatch: {qid}")

        if set(cached["relevant_document_ids"]) != set(
            query["relevant_document_ids"]
        ):
            raise RuntimeError(f"Qrels mismatch: {qid}")

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print("=" * 72)
    print("v0.6.5 — C++ HYBRID + CROSS-ENCODER")
    print("=" * 72)
    print(f"Dataset       : {args.dataset}")
    print(f"Queries       : {len(queries)}")
    print(f"Cross-Encoder : {MODEL_NAME}")
    print(f"Device        : {device}")
    print(f"CE Batch Size : {CE_BATCH_SIZE}")
    print(f"Dense ef      : {cache['protocol']['effective_ef_search']}")

    output_dir = (
        Path("results/msmarco_full/relevance/dev_v067/cpp_ce")
        if args.dataset == "dev"
        else OUTPUT_DIR
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"{args.dataset}_{len(queries)}q"

    checkpoint_path = (
        output_dir / f"checkpoint_{suffix}.json"
    )

    result_path = (
        output_dir / f"results_{suffix}.json"
    )

    protocol = {
        "name": "cpp-hybrid-union-ce-v065",
        "dataset": args.dataset,
        "query_count": len(queries),
        "query_ids": [str(q["query_id"]) for q in queries],
        "candidate_k": CANDIDATE_K,
        "top_k": TOP_K,
        "fusion": "dense_top50_union_cpp_bm25_top50",
        "cross_encoder": MODEL_NAME,
        "ce_batch_size": CE_BATCH_SIZE,
        "dense_cache_protocol": cache["protocol"],
        "rrf_k": RRF_K,
        "lexical_weight": LEXICAL_WEIGHT,
        "dense_weight": DENSE_WEIGHT,
    }

    if checkpoint_path.exists():
        checkpoint = load_json(checkpoint_path)

        if checkpoint["protocol"] != protocol:
            raise RuntimeError(
                "Checkpoint protocol mismatch. "
                "Do not resume with changed parameters."
            )

        completed = checkpoint["completed"]
        print(f"Resuming: {len(completed)} queries")

    else:
        completed = {}

    pending = [
        query
        for query in queries
        if str(query["query_id"]) not in completed
    ]

    if pending:
        print("\nLoading C++ BM25 corpus...")

        engine = retrieval_cpp.SearchEngine()
        engine.load_corpus(str(CORPUS_PATH))

        print("Loading Cross-Encoder...")

        model = CrossEncoder(
            MODEL_NAME,
            device=device,
        )

        print("Warming up Cross-Encoder...")

        model.predict(
            [["warmup query", "warmup passage"]],
            batch_size=1,
            show_progress_bar=False,
        )

        synchronize_mps(device)

        print("Warm-up complete.\n")

        with psycopg.connect(
            os.environ["DATABASE_URL"]
        ) as conn:

            for index, query in enumerate(pending, 1):
                qid = str(query["query_id"])
                text = query["query"]

                cached = cached_queries[qid]

                dense_candidates = cached["dense"]["candidates"]

                dense_ids = candidate_ids(dense_candidates)

                start_cpp = time.perf_counter()

                cpp_matches = engine.search_dense(
                    text,
                    CANDIDATE_K,
                )

                cpp_ms = (
                    time.perf_counter() - start_cpp
                ) * 1000.0

                cpp_ids = [
                    int(item.doc_id)
                    for item in cpp_matches
                ]

                union_ids = union_candidates(
                    dense_ids,
                    cpp_ids,
                )

                rrf_top10 = weighted_rrf(
                    cpp_ids,
                    dense_ids,
                )

                documents = {}

                # Reuse frozen dense candidate contents.
                for candidate in dense_candidates:
                    doc_id = int(candidate["document_id"])
                    content = candidate.get("content")

                    if isinstance(content, str):
                        documents[doc_id] = content

                missing_ids = [
                    doc_id
                    for doc_id in union_ids
                    if doc_id not in documents
                ]

                start_fetch = time.perf_counter()

                documents.update(
                    fetch_missing_documents(
                        conn,
                        missing_ids,
                    )
                )

                fetch_ms = (
                    time.perf_counter() - start_fetch
                ) * 1000.0

                still_missing = [
                    doc_id
                    for doc_id in union_ids
                    if doc_id not in documents
                ]

                if still_missing:
                    raise RuntimeError(
                        f"Missing documents for {qid}: "
                        f"{still_missing[:10]}"
                    )

                pairs = [
                    [text, documents[doc_id]]
                    for doc_id in union_ids
                ]

                synchronize_mps(device)

                start_ce = time.perf_counter()

                scores = model.predict(
                    pairs,
                    batch_size=CE_BATCH_SIZE,
                    show_progress_bar=False,
                )

                synchronize_mps(device)

                ce_ms = (
                    time.perf_counter() - start_ce
                ) * 1000.0

                scores = np.asarray(scores).reshape(-1)

                if len(scores) != len(union_ids):
                    raise RuntimeError(
                        f"CE output length mismatch: {qid}"
                    )

                if not np.isfinite(scores).all():
                    raise RuntimeError(
                        f"Non-finite CE scores: {qid}"
                    )

                ce_ranked = sorted(
                    zip(union_ids, scores),
                    key=lambda item: (
                        -float(item[1]),
                        item[0],
                    ),
                )

                ce_top10 = [
                    doc_id
                    for doc_id, _ in ce_ranked[:TOP_K]
                ]

                relevant = [
                    int(doc_id)
                    for doc_id in query["relevant_document_ids"]
                ]

                completed[qid] = {
                    "query_id": qid,
                    "query": text,
                    "relevant_document_ids": relevant,
                    "candidate_counts": {
                        "cpp_bm25": len(cpp_ids),
                        "dense": len(dense_ids),
                        "union": len(union_ids),
                        "overlap": (
                            len(cpp_ids)
                            + len(dense_ids)
                            - len(union_ids)
                        ),
                    },
                    "rrf": {
                        "top10": rrf_top10,
                        "metrics": evaluate(
                            rrf_top10,
                            relevant,
                        ),
                    },
                    "union_ce": {
                        "top10": ce_top10,
                        "metrics": evaluate(
                            ce_top10,
                            relevant,
                        ),
                    },
                    "latency": {
                        "cpp_bm25_ms": cpp_ms,
                        "document_fetch_ms": fetch_ms,
                        "cross_encoder_ms": ce_ms,
                    },
                }

                done = len(completed)

                if (
                    index % args.checkpoint_every == 0
                    or index == len(pending)
                ):
                    save_json_atomic(
                        checkpoint_path,
                        {
                            "protocol": protocol,
                            "completed": completed,
                        },
                    )

                    print(
                        f"[{done}/{len(queries)}] "
                        f"Union={len(union_ids)} "
                        f"BM25={cpp_ms:.1f}ms "
                        f"Fetch={fetch_ms:.1f}ms "
                        f"CE={ce_ms:.1f}ms"
                    )

    rows = [
        completed[str(query["query_id"])]
        for query in queries
    ]

    summary = {
        "cpp_hybrid_rrf": summarize(rows, "rrf"),
        "cpp_hybrid_union_ce": summarize(rows, "union_ce"),
    }

    for name, key in [
        ("cpp_bm25", "cpp_bm25_ms"),
        ("document_fetch", "document_fetch_ms"),
        ("cross_encoder", "cross_encoder_ms"),
    ]:
        values = [
            row["latency"][key]
            for row in rows
        ]

        summary[f"{name}_latency"] = {
            "mean_ms": statistics.fmean(values),
            "p50_ms": percentile(values, 50),
            "p95_ms": percentile(values, 95),
        }

    summary["candidate_union"] = {
        "mean_count": statistics.fmean(
            row["candidate_counts"]["union"]
            for row in rows
        ),
        "mean_overlap": statistics.fmean(
            row["candidate_counts"]["overlap"]
            for row in rows
        ),
    }

    output = {
        "protocol": protocol,
        "summary": summary,
        "per_query": rows,
    }

    save_json_atomic(result_path, output)

    print("\n=== Retrieval Quality ===")

    for name in [
        "cpp_hybrid_rrf",
        "cpp_hybrid_union_ce",
    ]:
        metrics = summary[name]

        print(
            f"{name:22s} "
            f"MRR@10={metrics['mrr_at_10']:.4f} "
            f"nDCG@10={metrics['ndcg_at_10']:.4f} "
            f"Recall@10={metrics['recall_at_10']:.4f}"
        )

    print("\n=== Latency ===")

    for name in [
        "cpp_bm25",
        "document_fetch",
        "cross_encoder",
    ]:
        latency = summary[f"{name}_latency"]

        print(
            f"{name:22s} "
            f"P50={latency['p50_ms']:.2f}ms "
            f"P95={latency['p95_ms']:.2f}ms"
        )

    print("\n=== Candidate Union ===")
    print(summary["candidate_union"])

    print(f"\nSaved: {result_path}")


if __name__ == "__main__":
    main()
