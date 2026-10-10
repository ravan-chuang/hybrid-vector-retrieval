import json
import math
import random
import statistics
import time
from pathlib import Path

import psycopg
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


DATABASE_URL = (
    "postgresql://retrieval:retrieval123@127.0.0.1:5433/retrieval_db"
)

DATASET_NAME = "sentence-transformers/msmarco"
DATASET_CONFIG = "queries"
DATASET_SPLIT = "train"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOTAL_DOCS = 8_841_823
N_QUERIES = 100
TOP_K = 10
SEED = 42

HNSW_M = 16
HNSW_EF_CONSTRUCTION = 64
EF_VALUES = [10, 20, 40, 80, 160]

OUT_DIR = Path("results/msmarco_full/ann")
QUERY_FILE = OUT_DIR / "queries.json"
GT_FILE = OUT_DIR / "exact_ground_truth.json"
RESULT_FILE = OUT_DIR / "ann_benchmark.json"


def percentile(values, p):
    if not values:
        return 0.0

    xs = sorted(values)
    pos = (len(xs) - 1) * p
    lower = math.floor(pos)
    upper = math.ceil(pos)

    if lower == upper:
        return xs[lower]

    fraction = pos - lower
    return xs[lower] + (xs[upper] - xs[lower]) * fraction


def summarize(latencies):
    return {
        "mean_ms": statistics.mean(latencies),
        "p50_ms": percentile(latencies, 0.50),
        "p95_ms": percentile(latencies, 0.95),
        "qps": 1000.0 / statistics.mean(latencies),
    }


def save_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")

    tmp.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    tmp.replace(path)


def load_or_select_queries():
    if QUERY_FILE.exists():
        print(f"Loading frozen queries: {QUERY_FILE}")
        return json.loads(QUERY_FILE.read_text(encoding="utf-8"))

    print("Streaming MS MARCO queries...")

    ds = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split=DATASET_SPLIT,
        streaming=True,
    )

    # Reservoir sampling gives every streamed query equal probability
    # without materializing the full query dataset in memory.
    rng = random.Random(SEED)
    reservoir = []

    for i, row in enumerate(ds):
        item = {
            "query_id": str(row["query_id"]),
            "query": row["query"],
        }

        if i < N_QUERIES:
            reservoir.append(item)
        else:
            j = rng.randint(0, i)
            if j < N_QUERIES:
                reservoir[j] = item

    # Freeze deterministic order for all later runs.
    reservoir.sort(key=lambda x: int(x["query_id"]))

    save_json(QUERY_FILE, reservoir)

    print(f"Frozen {len(reservoir)} queries -> {QUERY_FILE}")

    return reservoir


def encode_queries(queries):
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"\nLoading model: {MODEL_NAME}")
    print(f"Device       : {device}")

    model = SentenceTransformer(
        MODEL_NAME,
        device=device,
    )

    texts = [q["query"] for q in queries]

    print(f"Encoding {len(texts)} queries...")

    embeddings = model.encode(
        texts,
        batch_size=128,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    vectors = {}

    for query, embedding in zip(queries, embeddings):
        # pgvector accepts textual vector representation.
        vector_text = "[" + ",".join(
            format(float(x), ".9g") for x in embedding
        ) + "]"

        vectors[query["query_id"]] = vector_text

    return vectors


def exact_search(conn, vector_text):
    with conn.cursor() as cur:
        # Force exact search. Query vector is already client-side, so
        # disabling index scans does NOT cause a second corpus scan.
        cur.execute("SET LOCAL enable_indexscan = off")
        cur.execute("SET LOCAL enable_bitmapscan = off")
        cur.execute("SET LOCAL max_parallel_workers_per_gather = 4")

        start = time.perf_counter()

        cur.execute(
            """
            SELECT document_id
            FROM msmarco_full_documents
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (vector_text, TOP_K),
        )

        rows = cur.fetchall()

        elapsed_ms = (time.perf_counter() - start) * 1000.0

    return [int(row[0]) for row in rows], elapsed_ms


def hnsw_search(conn, vector_text, ef_search):
    with conn.cursor() as cur:
        # Force ANN-capable planner path for benchmark consistency.
        cur.execute("SET LOCAL enable_seqscan = off")
        cur.execute(
            f"SET LOCAL hnsw.ef_search = {int(ef_search)}"
        )

        start = time.perf_counter()

        cur.execute(
            """
            SELECT document_id
            FROM msmarco_full_documents
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (vector_text, TOP_K),
        )

        rows = cur.fetchall()

        elapsed_ms = (time.perf_counter() - start) * 1000.0

    return [int(row[0]) for row in rows], elapsed_ms


def load_ground_truth():
    if not GT_FILE.exists():
        return {}

    return json.loads(GT_FILE.read_text(encoding="utf-8"))


def build_exact_ground_truth(conn, queries, vectors):
    gt = load_ground_truth()

    if gt:
        print(
            f"\nResuming exact ground truth: "
            f"{len(gt)}/{len(queries)} already complete"
        )
    else:
        print("\nBuilding exact ground truth...")

    for i, query in enumerate(queries, 1):
        qid = query["query_id"]

        if qid in gt:
            print(
                f"[Exact {i:03d}/{len(queries)}] "
                f"qid={qid} cached"
            )
            continue

        ids, latency = exact_search(
            conn,
            vectors[qid],
        )

        conn.commit()

        gt[qid] = {
            "query": query["query"],
            "document_ids": ids,
            "latency_ms": latency,
        }

        # Save after every expensive exact query.
        save_json(GT_FILE, gt)

        print(
            f"[Exact {i:03d}/{len(queries)}] "
            f"qid={qid} "
            f"{latency:.1f} ms"
        )

    return gt


def run_hnsw(conn, queries, vectors, gt):
    all_results = {}

    for ef in EF_VALUES:
        print(f"\nHNSW ef_search={ef}")

        per_query = []
        latencies = []
        recalls = []

        for i, query in enumerate(queries, 1):
            qid = query["query_id"]

            ids, latency = hnsw_search(
                conn,
                vectors[qid],
                ef,
            )

            conn.commit()

            exact_ids = gt[qid]["document_ids"]

            recall = (
                len(set(exact_ids).intersection(ids))
                / TOP_K
            )

            latencies.append(latency)
            recalls.append(recall)

            per_query.append(
                {
                    "query_id": qid,
                    "query": query["query"],
                    "document_ids": ids,
                    "latency_ms": latency,
                    "recall_at_10": recall,
                }
            )

            print(
                f"[ef={ef:3d} "
                f"{i:03d}/{len(queries)}] "
                f"qid={qid} "
                f"recall={recall:.2f} "
                f"{latency:.1f} ms"
            )

        summary = summarize(latencies)
        summary["recall_at_10"] = statistics.mean(recalls)

        all_results[str(ef)] = {
            "summary": summary,
            "per_query": per_query,
        }

    return all_results


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Full MS MARCO PostgreSQL / pgvector ANN Benchmark")
    print("=" * 72)

    print(f"Corpus documents : {TOTAL_DOCS:,}")
    print(f"Queries          : {N_QUERIES}")
    print(f"Top-k            : {TOP_K}")
    print(f"Seed             : {SEED}")
    print(f"ef_search        : {EF_VALUES}")

    queries = load_or_select_queries()

    if len(queries) != N_QUERIES:
        raise RuntimeError(
            f"Expected {N_QUERIES} queries, got {len(queries)}"
        )

    vectors = encode_queries(queries)

    with psycopg.connect(
        DATABASE_URL,
        autocommit=False,
    ) as conn:

        gt = build_exact_ground_truth(
            conn,
            queries,
            vectors,
        )

        exact_latencies = [
            gt[q["query_id"]]["latency_ms"]
            for q in queries
        ]

        exact_summary = summarize(exact_latencies)

        print("\nExact summary")
        print(json.dumps(exact_summary, indent=2))

        hnsw_results = run_hnsw(
            conn,
            queries,
            vectors,
            gt,
        )

    # Add speedup only after exact summary is known.
    for ef in EF_VALUES:
        summary = hnsw_results[str(ef)]["summary"]

        summary["speedup_p50"] = (
            exact_summary["p50_ms"]
            / summary["p50_ms"]
        )

    output = {
        "protocol": {
            "corpus": "MS MARCO full passage corpus",
            "documents": TOTAL_DOCS,
            "query_dataset": DATASET_NAME,
            "query_config": DATASET_CONFIG,
            "query_split": DATASET_SPLIT,
            "query_sampling": "reservoir sampling",
            "queries": N_QUERIES,
            "seed": SEED,
            "embedding_model": MODEL_NAME,
            "dimensions": 384,
            "normalized": True,
            "distance": "cosine",
            "top_k": TOP_K,
            "hnsw_m": HNSW_M,
            "hnsw_ef_construction": HNSW_EF_CONSTRUCTION,
            "ef_search": EF_VALUES,
            "timing_excludes_query_embedding": True,
            "ann_recall_definition": (
                "Overlap of HNSW Top-10 with exact dense Top-10"
            ),
        },
        "exact": {
            "summary": exact_summary,
            "per_query": [
                {
                    "query_id": q["query_id"],
                    **gt[q["query_id"]],
                }
                for q in queries
            ],
        },
        "hnsw": hnsw_results,
    }

    save_json(RESULT_FILE, output)

    print("\n" + "=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)

    print(
        f"Exact  "
        f"mean={exact_summary['mean_ms']:.1f} ms  "
        f"P50={exact_summary['p50_ms']:.1f} ms  "
        f"P95={exact_summary['p95_ms']:.1f} ms  "
        f"QPS={exact_summary['qps']:.2f}"
    )

    for ef in EF_VALUES:
        s = hnsw_results[str(ef)]["summary"]

        print(
            f"HNSW ef={ef:<3d} "
            f"Recall@10={s['recall_at_10']:.4f}  "
            f"Mean={s['mean_ms']:.1f} ms  "
            f"P50={s['p50_ms']:.1f} ms  "
            f"P95={s['p95_ms']:.1f} ms  "
            f"QPS={s['qps']:.2f}  "
            f"P50 speedup={s['speedup_p50']:.1f}x"
        )

    print(f"\nSaved: {RESULT_FILE}")


if __name__ == "__main__":
    main()
