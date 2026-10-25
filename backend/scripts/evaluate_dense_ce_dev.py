#!/usr/bin/env python3

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

MANIFEST_PATH = Path(
    "artifacts/msmarco_full/dev_small_manifest.json"
)

CACHE_PATH = Path(
    "results/msmarco_full/relevance/"
    "dev_v067/ce_candidates_dev.json"
)

OUTPUT_DIR = Path(
    "results/msmarco_full/relevance/dev_dense_ce"
)

CHECKPOINT_PATH = OUTPUT_DIR / "checkpoint.json"
RESULT_PATH = OUTPUT_DIR / "results_dev_6980q.json"

TOP_K = 10
CANDIDATE_K = 50
BATCH_SIZE = 64


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(path.suffix + ".tmp")

    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    temporary.replace(path)


def evaluate(ranked, relevant):
    relevant = set(relevant)
    ranked = ranked[:TOP_K]

    hits = [
        rank
        for rank, doc_id in enumerate(ranked, 1)
        if doc_id in relevant
    ]

    mrr = 1.0 / hits[0] if hits else 0.0

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in hits
    )

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(
            1,
            min(len(relevant), TOP_K) + 1,
        )
    )

    recall = (
        len(set(ranked) & relevant) / len(relevant)
        if relevant else 0.0
    )

    return {
        "mrr_at_10": mrr,
        "ndcg_at_10": dcg / idcg if idcg else 0.0,
        "recall_at_10": recall,
    }


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()


def main():
    parser = argparse.ArgumentParser()

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

    manifest = load_json(MANIFEST_PATH)
    cache = load_json(CACHE_PATH)

    queries = manifest["queries"]

    cached_queries = {
        str(row["query_id"]): row
        for row in cache["queries"]
    }

    if len(queries) != 6980:
        raise RuntimeError(
            f"Expected 6980 queries, got {len(queries)}"
        )

    if len(cached_queries) != 6980:
        raise RuntimeError(
            "Dense cache query count mismatch"
        )

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    protocol = {
        "name": "msmarco-dev-small-dense-ce-v1",
        "dataset": "msmarco_passage_dev_small",
        "queries": len(queries),
        "candidate_source": str(CACHE_PATH),
        "candidate_k": CANDIDATE_K,
        "top_k": TOP_K,
        "cross_encoder": MODEL_NAME,
        "batch_size": BATCH_SIZE,
        "device": device,
        "dense_cache_protocol": cache["protocol"],
        "document_id_mapping": "msmarco_pid_plus_1",
    }

    if CHECKPOINT_PATH.exists():
        checkpoint = load_json(CHECKPOINT_PATH)

        if checkpoint["protocol"] != protocol:
            raise RuntimeError(
                "Checkpoint protocol mismatch"
            )

        completed = checkpoint["completed"]

        print(
            f"Resuming {len(completed)} queries"
        )

    else:
        completed = {}

    pending = [
        row
        for row in queries
        if str(row["query_id"]) not in completed
    ]

    if args.limit is not None:
        pending = pending[:args.limit]

    print("Device:", device)
    print("Total queries:", len(queries))
    print("Completed:", len(completed))
    print("Pending this run:", len(pending))

    if pending:
        print("Loading Cross-Encoder...")

        model = CrossEncoder(
            MODEL_NAME,
            device=device,
        )

        model.predict(
            [["warmup query", "warmup passage"]],
            batch_size=1,
            show_progress_bar=False,
        )

        synchronize(device)

        for index, query in enumerate(pending, 1):
            qid = str(query["query_id"])
            text = query["query"]

            cached = cached_queries[qid]

            relevant = set(
                map(int, query["relevant_document_ids"])
            )

            cached_relevant = set(
                map(int, cached["relevant_document_ids"])
            )

            if relevant != cached_relevant:
                raise RuntimeError(
                    f"Qrels mismatch: {qid}"
                )

            candidates = (
                cached["dense"]["candidates"][:CANDIDATE_K]
            )

            ids = [
                int(row["document_id"])
                for row in candidates
            ]

            if len(ids) != CANDIDATE_K:
                raise RuntimeError(
                    f"Candidate count mismatch: {qid}"
                )

            if len(set(ids)) != len(ids):
                raise RuntimeError(
                    f"Duplicate candidate IDs: {qid}"
                )

            pairs = []

            for candidate in candidates:
                content = candidate.get("content")

                if not isinstance(content, str):
                    raise RuntimeError(
                        f"Missing content: {qid}"
                    )

                pairs.append([text, content])

            synchronize(device)

            start = time.perf_counter()

            scores = model.predict(
                pairs,
                batch_size=BATCH_SIZE,
                show_progress_bar=False,
            )

            synchronize(device)

            elapsed_ms = (
                time.perf_counter() - start
            ) * 1000.0

            scores = np.asarray(scores).reshape(-1)

            if len(scores) != len(ids):
                raise RuntimeError(
                    f"Score count mismatch: {qid}"
                )

            if not np.isfinite(scores).all():
                raise RuntimeError(
                    f"Non-finite scores: {qid}"
                )

            ranked = sorted(
                zip(ids, scores),
                key=lambda item: (
                    -float(item[1]),
                    item[0],
                ),
            )

            top10 = [
                doc_id
                for doc_id, _ in ranked[:TOP_K]
            ]

            completed[qid] = {
                "query_id": qid,
                "query": text,
                "relevant_document_ids": sorted(relevant),
                "candidate_count": len(ids),
                "dense_ce": {
                    "top10": top10,
                    "metrics": evaluate(
                        top10,
                        relevant,
                    ),
                },
                "cross_encoder_ms": elapsed_ms,
            }

            done = len(completed)

            if (
                done % args.checkpoint_every == 0
                or index == len(pending)
            ):
                save_atomic(
                    CHECKPOINT_PATH,
                    {
                        "protocol": protocol,
                        "completed": completed,
                    },
                )

                print(
                    f"Completed: {done}/{len(queries)} "
                    f"| Last CE: {elapsed_ms:.1f} ms",
                    flush=True,
                )

    if len(completed) != len(queries):
        print(
            f"Partial run: {len(completed)}/{len(queries)}"
        )
        return

    rows = [
        completed[str(row["query_id"])]
        for row in queries
    ]

    summary = {}

    for metric in (
        "mrr_at_10",
        "ndcg_at_10",
        "recall_at_10",
    ):
        summary[metric] = statistics.fmean(
            row["dense_ce"]["metrics"][metric]
            for row in rows
        )

    summary["cross_encoder_mean_ms"] = (
        statistics.fmean(
            row["cross_encoder_ms"]
            for row in rows
        )
    )

    result = {
        "protocol": protocol,
        "summary": summary,
        "per_query": rows,
    }

    save_atomic(RESULT_PATH, result)

    print("\n===== DENSE + CE DEV SMALL =====")
    print("Queries:", len(rows))

    for key, value in summary.items():
        print(f"{key}: {value:.6f}")

    print("Saved:", RESULT_PATH)


if __name__ == "__main__":
    main()
