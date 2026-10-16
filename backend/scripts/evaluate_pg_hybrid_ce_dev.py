import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TOP_K = 10
CANDIDATE_K = 50
BATCH_SIZE = 64

MANIFEST_PATH = Path(
    "artifacts/msmarco_full/dev_small_manifest.json"
)

CACHE_PATH = Path(
    "results/msmarco_full/relevance/dev_v067/"
    "ce_candidates_dev.json"
)

OUTPUT_DIR = Path(
    "results/msmarco_full/relevance/dev_v067/pg_ce"
)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def evaluate(ranked_ids, relevant_ids):
    relevant = set(map(int, relevant_ids))
    ranked = list(map(int, ranked_ids[:TOP_K]))

    if not relevant:
        raise ValueError("Query has no positive qrels")

    first_rank = next(
        (
            i
            for i, doc_id in enumerate(ranked, 1)
            if doc_id in relevant
        ),
        None,
    )

    mrr = 0.0 if first_rank is None else 1.0 / first_rank

    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, doc_id in enumerate(ranked, 1)
        if doc_id in relevant
    )

    ideal_hits = min(len(relevant), TOP_K)

    idcg = sum(
        1.0 / math.log2(i + 1)
        for i in range(1, ideal_hits + 1)
    )

    return {
        "mrr_at_10": mrr,
        "ndcg_at_10": dcg / idcg,
        "recall_at_10": len(set(ranked) & relevant) / len(relevant),
    }


def percentile(values, p):
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lo = math.floor(position)
    hi = math.ceil(position)

    if lo == hi:
        return ordered[lo]

    fraction = position - lo
    return (
        ordered[lo] * (1 - fraction)
        + ordered[hi] * fraction
    )


def summarize(rows):
    names = ("mrr_at_10", "ndcg_at_10", "recall_at_10")

    return {
        name: statistics.fmean(
            row["union_ce"]["metrics"][name]
            for row in rows
        )
        for name in names
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    if args.checkpoint_every <= 0:
        parser.error("--checkpoint-every must be positive")

    manifest = load_json(MANIFEST_PATH)
    cache = load_json(CACHE_PATH)

    manifest_rows = manifest["queries"]
    cache_rows = cache["queries"]

    if len(manifest_rows) != 6980 or len(cache_rows) != 6980:
        raise RuntimeError("Expected exactly 6980 Dev queries")

    protocol = cache["protocol"]

    if protocol["candidate_k"] != CANDIDATE_K:
        raise RuntimeError("Candidate K mismatch")

    if protocol["effective_ef_search"] != 160:
        raise RuntimeError("Dense ef_search mismatch")

    for expected, actual in zip(manifest_rows, cache_rows):
        if str(expected["query_id"]) != str(actual["query_id"]):
            raise RuntimeError("Query ID mismatch")

        if expected["query"] != actual["query"]:
            raise RuntimeError("Query text mismatch")

        if set(map(int, expected["relevant_document_ids"])) != set(
            map(int, actual["relevant_document_ids"])
        ):
            raise RuntimeError("Qrels mismatch")

    queries = cache_rows[:args.limit]

    output_protocol = {
        "dataset": "msmarco_passage_dev_small",
        "queries": len(queries),
        "candidate_k": CANDIDATE_K,
        "effective_ef_search": 160,
        "cross_encoder": MODEL_NAME,
        "batch_size": BATCH_SIZE,
        "top_k": TOP_K,
        "pipeline": "postgresql_fts_dense_union_ce",
    }

    suffix = f"dev_{len(queries)}q"

    checkpoint_path = (
        OUTPUT_DIR / f"checkpoint_{suffix}.json"
    )

    result_path = (
        OUTPUT_DIR / f"results_{suffix}.json"
    )

    completed = {}

    if checkpoint_path.exists():
        saved = load_json(checkpoint_path)

        if saved["protocol"] != output_protocol:
            raise RuntimeError("Checkpoint protocol mismatch")

        completed = saved["completed"]

    expected_ids = {str(row["query_id"]) for row in queries}

    if not set(completed).issubset(expected_ids):
        raise RuntimeError("Unexpected query IDs in checkpoint")

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 72)
    print("v0.6.7d — POSTGRESQL HYBRID + CROSS-ENCODER")
    print("=" * 72)
    print(f"Queries       : {len(queries)}")
    print(f"Cross-Encoder : {MODEL_NAME}")
    print(f"Device        : {device}")
    print(f"CE Batch Size : {BATCH_SIZE}")
    print(f"Cached        : {len(completed)}")

    model = CrossEncoder(MODEL_NAME, device=device)

    model.predict(
        [("test query", "test passage")],
        batch_size=1,
        show_progress_bar=False,
    )

    synchronize(device)

    for i, item in enumerate(queries, 1):
        qid = str(item["query_id"])

        if qid in completed:
            continue

        query = item["query"]
        relevant = item["relevant_document_ids"]

        dense_candidates = item["dense"]["candidates"][:CANDIDATE_K]
        lexical_candidates = item["lexical"]["candidates"][:CANDIDATE_K]

        dense_ids = [
            int(x["document_id"])
            for x in dense_candidates
        ]

        lexical_ids = [
            int(x["document_id"])
            for x in lexical_candidates
        ]

        document_map = {
            int(x["document_id"]): x["content"]
            for x in dense_candidates
        }

        document_map.update({
            int(x["document_id"]): x["content"]
            for x in lexical_candidates
        })

        union_ids = list(dict.fromkeys(dense_ids + lexical_ids))

        if not union_ids:
            raise RuntimeError(f"Empty candidates: query {qid}")

        pairs = [
            (query, document_map[doc_id])
            for doc_id in union_ids
        ]

        synchronize(device)
        start = time.perf_counter()

        scores = model.predict(
            pairs,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
        )

        synchronize(device)

        ce_ms = (time.perf_counter() - start) * 1000.0

        ranked = [
            doc_id
            for doc_id, score in sorted(
                zip(union_ids, scores),
                key=lambda x: (-float(x[1]), x[0]),
            )
        ]

        top10 = ranked[:TOP_K]

        completed[qid] = {
            "query_id": qid,
            "query": query,
            "relevant_document_ids": relevant,
            "candidate_counts": {
                "dense": len(dense_ids),
                "postgresql_fts": len(lexical_ids),
                "overlap": len(set(dense_ids) & set(lexical_ids)),
                "union": len(union_ids),
            },
            "union_ce": {
                "top10": top10,
                "metrics": evaluate(top10, relevant),
                "latency_ms": ce_ms,
            },
        }

        if i % args.checkpoint_every == 0 or i == len(queries):
            save_json_atomic(
                checkpoint_path,
                {
                    "protocol": output_protocol,
                    "completed": completed,
                },
            )

            print(
                f"[{i}/{len(queries)}] "
                f"Union={len(union_ids)} "
                f"CE={ce_ms:.1f}ms",
                flush=True,
            )

    rows = [
        completed[str(item["query_id"])]
        for item in queries
    ]

    quality = summarize(rows)

    latencies = [
        row["union_ce"]["latency_ms"]
        for row in rows
    ]

    union_counts = [
        row["candidate_counts"]["union"]
        for row in rows
    ]

    overlap_counts = [
        row["candidate_counts"]["overlap"]
        for row in rows
    ]

    summary = {
        "quality": {
            "pg_hybrid_union_ce": quality,
        },
        "latency": {
            "cross_encoder": {
                "p50_ms": percentile(latencies, 0.50),
                "p95_ms": percentile(latencies, 0.95),
            }
        },
        "candidate_union": {
            "mean_count": statistics.fmean(union_counts),
            "mean_overlap": statistics.fmean(overlap_counts),
        },
    }

    result = {
        "protocol": output_protocol,
        "summary": summary,
        "per_query": rows,
    }

    save_json_atomic(result_path, result)

    print("\n=== Retrieval Quality ===")
    for name, value in quality.items():
        print(f"{name}: {value:.6f}")

    print("\n=== Latency ===")
    print(
        f"Cross-Encoder P50={summary['latency']['cross_encoder']['p50_ms']:.2f}ms "
        f"P95={summary['latency']['cross_encoder']['p95_ms']:.2f}ms"
    )

    print("\n=== Candidate Union ===")
    print(summary["candidate_union"])

    print(f"\nSaved: {result_path}")


if __name__ == "__main__":
    main()
