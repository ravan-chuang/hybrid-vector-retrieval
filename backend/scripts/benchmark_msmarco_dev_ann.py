import argparse
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path

import psycopg
import torch
from sentence_transformers import SentenceTransformer


BASE = Path(__file__).resolve().parents[2]
MANIFEST = BASE / "artifacts/msmarco_full/dev_small_manifest.json"
OUT_DIR = BASE / "results/msmarco_full/ann_dev_6980"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 10
EF_VALUES = [10, 20, 40, 80, 160]
EXPECTED_QUERIES = 6980

DATABASE_URL = os.environ["DATABASE_URL"]


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(path)


def load_json(path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def percentile(values, p):
    xs = sorted(values)
    pos = (len(xs) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def summarize(values):
    return {
        "mean_ms": statistics.mean(values),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "qps": 1000.0 / statistics.mean(values),
    }


def load_queries(limit):
    manifest = load_json(MANIFEST)
    protocol = manifest["protocol"]
    queries = manifest["queries"]

    if protocol["dataset"] != "msmarco_passage_dev_small":
        raise RuntimeError("Unexpected dataset")

    if len(queries) != EXPECTED_QUERIES:
        raise RuntimeError("Expected 6980 Dev queries")

    if len({str(q["query_id"]) for q in queries}) != len(queries):
        raise RuntimeError("Duplicate query IDs")

    selected = [
        {
            "query_id": str(q["query_id"]),
            "query": q["query"],
        }
        for q in queries[:limit]
    ]

    return selected


def fingerprint(queries):
    payload = json.dumps(
        queries, ensure_ascii=False, sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def encode_queries(queries):
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"Embedding device: {device}")

    model = SentenceTransformer(MODEL_NAME, device=device)

    embeddings = model.encode(
        [q["query"] for q in queries],
        batch_size=128,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    return {
        q["query_id"]: "[" + ",".join(
            format(float(x), ".9g") for x in emb
        ) + "]"
        for q, emb in zip(queries, embeddings)
    }


def search(conn, vector_text, exact=False, ef=40):
    with conn.cursor() as cur:
        if exact:
            cur.execute("SET LOCAL enable_indexscan = off")
            cur.execute("SET LOCAL enable_bitmapscan = off")
            cur.execute(
                "SET LOCAL max_parallel_workers_per_gather = 4"
            )
        else:
            cur.execute("SET LOCAL enable_seqscan = off")
            cur.execute(
                "SELECT set_config('hnsw.ef_search', %s, true)",
                (str(ef),),
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

        ids = [int(row[0]) for row in cur.fetchall()]
        latency = (time.perf_counter() - start) * 1000

    conn.commit()
    return ids, latency


def verify_plan(conn, vector_text, exact=False, ef=40):
    with conn.cursor() as cur:
        if exact:
            cur.execute("SET LOCAL enable_indexscan = off")
            cur.execute("SET LOCAL enable_bitmapscan = off")
        else:
            cur.execute("SET LOCAL enable_seqscan = off")
            cur.execute(
                "SELECT set_config('hnsw.ef_search', %s, true)",
                (str(ef),),
            )

        cur.execute(
            """
            EXPLAIN (FORMAT TEXT)
            SELECT document_id
            FROM msmarco_full_documents
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (vector_text, TOP_K),
        )

        plan = "\n".join(row[0] for row in cur.fetchall())

    conn.commit()

    uses_hnsw = "idx_msmarco_full_embedding_hnsw" in plan

    if exact and uses_hnsw:
        raise RuntimeError("Exact query unexpectedly uses HNSW")

    if not exact and not uses_hnsw:
        raise RuntimeError(f"HNSW index not used:\n{plan}")

    print(
        f"Plan verified: {'Exact' if exact else f'HNSW ef={ef}'}"
    )


def validate_checkpoint(data, selected, full_queries):
    allowed = {q["query_id"]: q["query"] for q in full_queries}

    for qid, row in data.items():
        if qid not in allowed:
            raise RuntimeError(f"Unknown checkpoint query: {qid}")

        if row.get("query") != allowed[qid]:
            raise RuntimeError(f"Query mismatch: {qid}")

        ids = row.get("document_ids")

        if not isinstance(ids, list) or len(ids) != TOP_K:
            raise RuntimeError(f"Invalid Top-{TOP_K}: {qid}")



def verify_experiment_config():
    path = OUT_DIR / "experiment_config.json"

    expected = {
        "dataset": "msmarco_passage_dev_small",
        "manifest": "artifacts/msmarco_full/dev_small_manifest.json",
        "corpus": "msmarco_full_documents",
        "expected_documents": 8841823,
        "embedding_model": MODEL_NAME,
        "embedding_normalized": True,
        "embedding_dimensions": 384,
        "distance": "cosine",
        "top_k": TOP_K,
        "hnsw_index": "idx_msmarco_full_embedding_hnsw",
        "hnsw_m": 16,
        "hnsw_ef_construction": 64,
        "ef_search": EF_VALUES,
        "query_order": "manifest_order",
        "document_id_mapping": "msmarco_pid_plus_1",
    }

    if not path.exists():
        raise RuntimeError(
            "Missing experiment_config.json; "
            "refusing to reuse checkpoints."
        )

    actual = load_json(path)

    if actual != expected:
        raise RuntimeError(
            "Experiment configuration mismatch; "
            "refusing to reuse checkpoints."
        )

    print("Experiment configuration: PASS")

def run(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    verify_experiment_config()

    full_queries = load_queries(EXPECTED_QUERIES)
    queries = full_queries[:args.limit]

    gt_path = OUT_DIR / "exact_ground_truth.json"
    hnsw_path = OUT_DIR / "hnsw_checkpoint.json"

    gt = load_json(gt_path, {})
    hnsw = load_json(hnsw_path, {})

    validate_checkpoint(gt, queries, full_queries)

    for ef in EF_VALUES:
        hnsw.setdefault(str(ef), {})
        validate_checkpoint(
            hnsw[str(ef)], queries, full_queries
        )

    print(f"Dev queries selected: {len(queries)}")
    print(f"Exact cached: {sum(q['query_id'] in gt for q in queries)}")

    vectors = encode_queries(queries)

    with psycopg.connect(DATABASE_URL, autocommit=False) as conn:
        verify_plan(conn, vectors[queries[0]["query_id"]], exact=True)
        verify_plan(conn, vectors[queries[0]["query_id"]], ef=40)

        for i, q in enumerate(queries, 1):
            qid = q["query_id"]

            if qid in gt:
                continue

            ids, latency = search(
                conn, vectors[qid], exact=True
            )

            gt[qid] = {
                "query": q["query"],
                "document_ids": ids,
                "latency_ms": latency,
            }

            save_json(gt_path, gt)

            print(
                f"[Exact {i}/{len(queries)}] "
                f"{latency:.1f} ms",
                flush=True,
            )

        for ef in EF_VALUES:
            print(f"\nHNSW ef={ef}")

            for i, q in enumerate(queries, 1):
                qid = q["query_id"]

                if qid in hnsw[str(ef)]:
                    continue

                ids, latency = search(
                    conn, vectors[qid], ef=ef
                )

                exact_ids = gt[qid]["document_ids"]
                recall = len(
                    set(ids) & set(exact_ids)
                ) / TOP_K

                hnsw[str(ef)][qid] = {
                    "query": q["query"],
                    "document_ids": ids,
                    "latency_ms": latency,
                    "recall_at_10": recall,
                }

                save_json(hnsw_path, hnsw)

                print(
                    f"[ef={ef} {i}/{len(queries)}] "
                    f"recall={recall:.2f} "
                    f"{latency:.1f} ms",
                    flush=True,
                )

    exact_times = [
        gt[q["query_id"]]["latency_ms"] for q in queries
    ]
    exact_summary = summarize(exact_times)

    results = {}

    for ef in EF_VALUES:
        rows = [
            hnsw[str(ef)][q["query_id"]]
            for q in queries
        ]

        summary = summarize(
            [row["latency_ms"] for row in rows]
        )
        summary["recall_at_10"] = statistics.mean(
            row["recall_at_10"] for row in rows
        )
        summary["speedup_p50"] = (
            exact_summary["p50_ms"] / summary["p50_ms"]
        )

        results[str(ef)] = {
            "summary": summary,
            "per_query": [
                {
                    "query_id": q["query_id"],
                    **hnsw[str(ef)][q["query_id"]],
                }
                for q in queries
            ],
        }

    output = {
        "protocol": {
            "dataset": "msmarco_passage_dev_small",
            "documents": 8_841_823,
            "queries": len(queries),
            "query_selection": "first N manifest queries",
            "query_fingerprint": fingerprint(queries),
            "embedding_model": MODEL_NAME,
            "normalized": True,
            "distance": "cosine",
            "top_k": TOP_K,
            "ef_search": EF_VALUES,
            "timing_excludes_embedding": True,
            "ann_recall_definition": (
                "HNSW Top-10 overlap with Exact Dense Top-10"
            ),
            "latency_repetitions": 1,
        },
        "exact": {
            "summary": exact_summary,
        },
        "hnsw": results,
    }

    output_path = OUT_DIR / f"ann_benchmark_{len(queries)}q.json"
    save_json(output_path, output)

    print("\n===== FINAL SUMMARY =====")
    print("Exact:", exact_summary)

    for ef, result in results.items():
        print(f"HNSW ef={ef}:", result["summary"])

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        choices=[10, 100, 1000, 6980],
    )

    run(parser.parse_args())
