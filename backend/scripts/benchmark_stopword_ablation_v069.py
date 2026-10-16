import argparse
import json
import random
import re
import statistics
import time
from pathlib import Path

import numpy as np

from backend import retrieval_cpp


BASE = Path(
    "results/msmarco_full/ablation/stopwords_v069"
)

CORPUS_PATH = Path(
    "cpp/data/msmarco_full.bin"
)

QUERY_PATH = Path(
    "data/msmarco/dev/queries.dev.small.tsv"
)

QRELS_PATH = Path(
    "data/msmarco/dev/qrels.dev.small.tsv"
)

TOP_K = 50

STOPWORDS = {
    "a", "an", "the",
    "and", "or", "but",
    "is", "are", "was", "were",
    "be", "been", "being",
    "am",
    "of", "to", "in", "on",
    "at", "by", "for", "from",
    "with", "as",
    "it", "its",
    "this", "that", "these", "those",
    "what", "which", "who", "whom",
    "where", "when", "why", "how",
    "do", "does", "did",
    "can", "could", "would", "should",
    "will", "shall",
    "has", "have", "had",
    "i", "you", "he", "she",
    "we", "they",
    "my", "your", "his", "her",
    "our", "their",
}


def load_queries():
    rows = []

    with QUERY_PATH.open(encoding="utf-8") as f:
        for line in f:
            qid, query = line.rstrip("\n").split(
                "\t", 1
            )

            rows.append({
                "query_id": qid,
                "query": query,
            })

    assert len(rows) == 6980, len(rows)

    return rows


def load_qrels():
    qrels = {}

    with QRELS_PATH.open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) != 4:
                continue

            qid, _, docid, relevance = parts

            if int(relevance) <= 0:
                continue

            qrels.setdefault(qid, set()).add(
                int(docid)
            )

    return qrels


def filter_stopwords(query):
    tokens = re.findall(
        r"[A-Za-z0-9]+",
        query.lower(),
    )

    filtered = [
        token
        for token in tokens
        if token not in STOPWORDS
    ]

    return " ".join(filtered), len(tokens), len(filtered)


def search(engine, query):
    start = time.perf_counter_ns()

    if query.strip():
        matches = engine.search_dense(
            query,
            TOP_K,
        )

        ids = [
            int(item.doc_id)
            for item in matches
        ]
    else:
        ids = []

    elapsed_ms = (
        time.perf_counter_ns() - start
    ) / 1_000_000.0

    return ids, elapsed_ms


def mrr_at_10(ids, relevant):
    for rank, docid in enumerate(ids[:10], 1):
        if docid in relevant:
            return 1.0 / rank

    return 0.0


def ndcg_at_10(ids, relevant):
    import math

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, docid in enumerate(ids[:10], 1)
        if docid in relevant
    )

    ideal_hits = min(10, len(relevant))

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    return dcg / idcg if idcg else 0.0


def recall_at_k(ids, relevant, k):
    if not relevant:
        return 0.0

    return len(
        set(ids[:k]) & relevant
    ) / len(relevant)


def summarize(values):
    values = list(values)

    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": max(values),
    }


def save_json(name, data):
    BASE.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = BASE / name

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=6980,
    )

    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be >= 1")

    if args.limit < 1:
        parser.error("--limit must be >= 1")

    rng = random.Random(42)

    queries = load_queries()[:args.limit]
    qrels = load_qrels()

    print("=" * 72)
    print("v0.6.9 — C++ BM25 STOPWORD ABLATION")
    print("=" * 72)

    print("\nLoading C++ BM25 corpus...")

    engine = retrieval_cpp.SearchEngine()

    engine.load_corpus(
        str(CORPUS_PATH)
    )

    print("Corpus loaded.")

    print("\nWarming up...")

    for item in queries[:args.warmup]:
        search(engine, item["query"])

    print("Warm-up complete.")

    results = []

    total = len(queries)

    for index, item in enumerate(queries, 1):
        qid = item["query_id"]
        original = item["query"]

        filtered, original_len, filtered_len = (
            filter_stopwords(original)
        )

        relevant = qrels.get(qid, set())

        original_ids = None
        filtered_ids = None

        original_times = []
        filtered_times = []

        for _ in range(args.repeats):
            conditions = [
                ("off", original),
                ("on", filtered),
            ]

            rng.shuffle(conditions)

            for condition, query_text in conditions:
                ids, elapsed = search(
                    engine,
                    query_text,
                )

                if condition == "off":
                    original_times.append(elapsed)

                    if original_ids is None:
                        original_ids = ids
                    elif original_ids != ids:
                        raise RuntimeError(
                            f"Non-deterministic OFF result: {qid}"
                        )
                else:
                    filtered_times.append(elapsed)

                    if filtered_ids is None:
                        filtered_ids = ids
                    elif filtered_ids != ids:
                        raise RuntimeError(
                            f"Non-deterministic ON result: {qid}"
                        )

        overlap = len(
            set(original_ids) & set(filtered_ids)
        )

        results.append({
            "query_id": qid,
            "query": original,
            "filtered_query": filtered,
            "original_tokens": original_len,
            "filtered_tokens": filtered_len,
            "empty_after_filtering": (
                filtered_len == 0
            ),
            "relevant_document_count": len(relevant),
            "off": {
                "top50": original_ids,
                "latencies_ms": original_times,
                "mrr10": mrr_at_10(
                    original_ids, relevant
                ),
                "ndcg10": ndcg_at_10(
                    original_ids, relevant
                ),
                "recall10": recall_at_k(
                    original_ids, relevant, 10
                ),
                "recall50": recall_at_k(
                    original_ids, relevant, 50
                ),
            },
            "on": {
                "top50": filtered_ids,
                "latencies_ms": filtered_times,
                "mrr10": mrr_at_10(
                    filtered_ids, relevant
                ),
                "ndcg10": ndcg_at_10(
                    filtered_ids, relevant
                ),
                "recall10": recall_at_k(
                    filtered_ids, relevant, 10
                ),
                "recall50": recall_at_k(
                    filtered_ids, relevant, 50
                ),
            },
            "candidate_overlap50": overlap,
        })

        if index % 250 == 0 or index == total:
            print(
                f"[{index}/{total}] completed",
                flush=True,
            )

            save_json(
                "per_query.json",
                results,
            )

    summary = {
        "protocol": {
            "dataset": "MS MARCO Dev Small",
            "query_count": len(queries),
            "corpus_size": engine.document_count(),
            "top_k": TOP_K,
            "repeats": args.repeats,
            "filtering": "query_only",
            "seed": 42,
            "stopwords": sorted(STOPWORDS),
        },
        "empty_queries": sum(
            x["empty_after_filtering"]
            for x in results
        ),
    }

    for condition in ["off", "on"]:
        summary[condition] = {}

        for metric in [
            "mrr10",
            "ndcg10",
            "recall10",
            "recall50",
        ]:
            summary[condition][metric] = (
                statistics.fmean(
                    x[condition][metric]
                    for x in results
                )
            )

        latency_values = [
            elapsed
            for x in results
            for elapsed in x[condition]["latencies_ms"]
        ]

        summary[condition]["latency_ms"] = (
            summarize(latency_values)
        )

    summary["mean_candidate_overlap50"] = (
        statistics.fmean(
            x["candidate_overlap50"]
            for x in results
        )
    )

    save_json(
        "summary.json",
        summary,
    )

    print("\n" + "=" * 72)
    print("FINAL RESULTS")
    print("=" * 72)

    print(json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    ))

    print(f"\nSaved to: {BASE}")


if __name__ == "__main__":
    main()
