import json
import os
import time
from pathlib import Path

import psycopg
import torch
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "results/msmarco_full/ann_dev_6980"
OUTPUT = DATA / "hnsw_failure_diagnostics.json"

QUERY_IDS = ["15441", "23285", "15382", "17884"]
EF_VALUES = [160, 320, 640, 1000]
TOP_K = 10

DB_URL = os.environ["DATABASE_URL"]

gt = json.loads((DATA / "exact_ground_truth.json").read_text())

device = "mps" if torch.backends.mps.is_available() else "cpu"

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device=device,
)

vectors = model.encode(
    [gt[qid]["query"] for qid in QUERY_IDS],
    normalize_embeddings=True,
    convert_to_numpy=True,
)

results = {
    "experiment": "HNSW Failure Analysis",
    "query_count": len(QUERY_IDS),
    "ef_values": EF_VALUES,
    "queries": {},
}

with psycopg.connect(DB_URL) as conn:
    for qid, vector in zip(QUERY_IDS, vectors):
        query = gt[qid]["query"]
        exact_ids = gt[qid]["document_ids"]
        exact_set = set(exact_ids)

        vector_text = "[" + ",".join(
            format(float(x), ".9g") for x in vector
        ) + "]"

        print(f"\nQuery {qid}: {query}")

        query_results = {
            "query": query,
            "exact_top10": exact_ids,
            "runs": {},
        }

        for ef in EF_VALUES:
            with conn.cursor() as cur:
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
                latency_ms = (time.perf_counter() - start) * 1000

            conn.commit()

            recall = len(set(ids) & exact_set) / TOP_K

            query_results["runs"][str(ef)] = {
                "document_ids": ids,
                "recall_at_10": recall,
                "latency_ms": latency_ms,
                "missed_exact_ids": [
                    doc for doc in exact_ids if doc not in ids
                ],
            }

            print(
                f"  ef={ef:<4} "
                f"Recall@10={recall:.2f} "
                f"Latency={latency_ms:.2f} ms"
            )

        results["queries"][qid] = query_results

OUTPUT.write_text(
    json.dumps(results, indent=2, ensure_ascii=False)
)

print(f"\nSaved: {OUTPUT}")
