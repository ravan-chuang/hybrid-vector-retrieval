#!/usr/bin/env python3

import json
import math
import os
import statistics
import time
from pathlib import Path

import psycopg
import torch
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 10
CANDIDATE_K = 50

EF_SEARCH = 160

RRF_K = 60
LEXICAL_WEIGHT = 0.1
DENSE_WEIGHT = 0.9

MANIFEST_PATH = Path(
    "artifacts/msmarco_full/eval_manifest_10k.json"
)

OUT_DIR = Path(
    "results/msmarco_full/relevance"
)

RESULT_PATH = OUT_DIR / "first_stage_results_10k.json"

CHECKPOINT_PATH = (
    OUT_DIR / "first_stage_checkpoint_10k.json"
)


def get_database_url():
    url = os.environ.get("DATABASE_URL")

    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. "
            "Run: set -a; source .env; set +a"
        )

    return url


def load_manifest():
    data = json.loads(
        MANIFEST_PATH.read_text(
            encoding="utf-8"
        )
    )

    queries = data["queries"]

    if len(queries) != 10_000:
        raise RuntimeError(
            f"Expected frozen 10,000 queries, "
            f"got {len(queries)}"
        )

    return data


def choose_device():
    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def vector_to_pgvector(vector):
    return "[" + ",".join(
        f"{float(x):.8f}"
        for x in vector
    ) + "]"


def encode_queries(queries):
    device = choose_device()

    print(f"Loading model : {MODEL_NAME}")
    print(f"Device        : {device}")

    model = SentenceTransformer(
        MODEL_NAME,
        device=device,
    )

    texts = [
        q["query"]
        for q in queries
    ]

    print(
        f"Encoding {len(texts):,} "
        f"frozen queries..."
    )

    vectors = model.encode(
        texts,
        batch_size=128,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    result = {}

    for query, vector in zip(
        queries,
        vectors,
    ):
        result[
            query["query_id"]
        ] = vector_to_pgvector(vector)

    return result


def lexical_search(conn, query_text):
    sql = """
        SELECT
            document_id,
            ts_rank_cd(
                to_tsvector(
                    'english',
                    content
                ),
                websearch_to_tsquery(
                    'english',
                    %s
                )
            ) AS score
        FROM msmarco_full_documents
        WHERE
            to_tsvector(
                'english',
                content
            )
            @@
            websearch_to_tsquery(
                'english',
                %s
            )
        ORDER BY score DESC,
                 document_id ASC
        LIMIT %s
    """

    started = time.perf_counter()

    with conn.cursor() as cur:
        cur.execute(
            "SET LOCAL jit = off"
        )

        cur.execute(
            sql,
            (
                query_text,
                query_text,
                CANDIDATE_K,
            ),
        )

        rows = cur.fetchall()

    conn.commit()

    latency_ms = (
        time.perf_counter() - started
    ) * 1000.0

    return [
        {
            "document_id": row[0],
            "score": float(row[1]),
        }
        for row in rows
    ], latency_ms


def dense_search(conn, vector_text):
    sql = """
        SELECT
            document_id,
            1 - (
                embedding <=> %s::vector
            ) AS score
        FROM msmarco_full_documents
        ORDER BY
            embedding <=> %s::vector
        LIMIT %s
    """

    started = time.perf_counter()

    with conn.cursor() as cur:
        cur.execute(
            "SET LOCAL jit = off"
        )

        # This experiment explicitly evaluates
        # the frozen HNSW operating point.
        cur.execute(
            "SET LOCAL enable_seqscan = off"
        )

        cur.execute(
            "SELECT set_config("
            "'hnsw.ef_search', %s, true"
            ")",
            (str(EF_SEARCH),),
        )

        cur.execute(
            sql,
            (
                vector_text,
                vector_text,
                CANDIDATE_K,
            ),
        )

        rows = cur.fetchall()

    conn.commit()

    latency_ms = (
        time.perf_counter() - started
    ) * 1000.0

    return [
        {
            "document_id": row[0],
            "score": float(row[1]),
        }
        for row in rows
    ], latency_ms


def weighted_rrf(
    lexical,
    dense,
):
    scores = {}

    for rank, item in enumerate(
        lexical,
        start=1,
    ):
        doc_id = item["document_id"]

        scores[doc_id] = (
            scores.get(doc_id, 0.0)
            + LEXICAL_WEIGHT
            / (RRF_K + rank)
        )

    for rank, item in enumerate(
        dense,
        start=1,
    ):
        doc_id = item["document_id"]

        scores[doc_id] = (
            scores.get(doc_id, 0.0)
            + DENSE_WEIGHT
            / (RRF_K + rank)
        )

    ranked = sorted(
        scores.items(),
        key=lambda x: (
            -x[1],
            x[0],
        ),
    )

    return [
        {
            "document_id": doc_id,
            "score": score,
        }
        for doc_id, score
        in ranked[:TOP_K]
    ]


def top_ids(results):
    return [
        x["document_id"]
        for x in results[:TOP_K]
    ]


def reciprocal_rank(
    retrieved,
    relevant,
):
    relevant = set(relevant)

    for rank, doc_id in enumerate(
        retrieved[:TOP_K],
        start=1,
    ):
        if doc_id in relevant:
            return 1.0 / rank

    return 0.0


def dcg_at_k(
    retrieved,
    relevant,
):
    relevant = set(relevant)

    score = 0.0

    for rank, doc_id in enumerate(
        retrieved[:TOP_K],
        start=1,
    ):
        if doc_id in relevant:
            score += (
                1.0
                / math.log2(rank + 1)
            )

    return score


def ndcg_at_k(
    retrieved,
    relevant,
):
    relevant = set(relevant)

    dcg = dcg_at_k(
        retrieved,
        relevant,
    )

    ideal_hits = min(
        len(relevant),
        TOP_K,
    )

    if ideal_hits == 0:
        return 0.0

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(
            1,
            ideal_hits + 1,
        )
    )

    return dcg / idcg


def recall_at_k(
    retrieved,
    relevant,
):
    relevant = set(relevant)

    if not relevant:
        return 0.0

    hits = len(
        set(retrieved[:TOP_K])
        .intersection(relevant)
    )

    return hits / len(relevant)


def evaluate_ranking(
    retrieved,
    relevant,
):
    return {
        "mrr_at_10":
            reciprocal_rank(
                retrieved,
                relevant,
            ),
        "ndcg_at_10":
            ndcg_at_k(
                retrieved,
                relevant,
            ),
        "recall_at_10":
            recall_at_k(
                retrieved,
                relevant,
            ),
    }


def summarize_metrics(
    per_query,
    method,
):
    rows = [
        q[method]["metrics"]
        for q in per_query
    ]

    return {
        "mrr_at_10":
            statistics.fmean(
                r["mrr_at_10"]
                for r in rows
            ),
        "ndcg_at_10":
            statistics.fmean(
                r["ndcg_at_10"]
                for r in rows
            ),
        "recall_at_10":
            statistics.fmean(
                r["recall_at_10"]
                for r in rows
            ),
    }


def percentile(values, p):
    if not values:
        return None

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    position = (
        (len(values) - 1)
        * p
    )

    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return values[lower]

    fraction = position - lower

    return (
        values[lower]
        * (1.0 - fraction)
        + values[upper]
        * fraction
    )


def summarize_latency(
    per_query,
    method,
):
    values = [
        q[method]["latency_ms"]
        for q in per_query
        if "latency_ms"
        in q[method]
    ]

    return {
        "mean_ms":
            statistics.fmean(values),
        "p50_ms":
            percentile(values, 0.50),
        "p95_ms":
            percentile(values, 0.95),
        "qps":
            1000.0
            / statistics.fmean(values),
    }


def save_json(path, obj):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    tmp.replace(path)


def main():
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = load_manifest()
    queries = manifest["queries"]

    vectors = encode_queries(
        queries
    )

    checkpoint = {}

    if CHECKPOINT_PATH.exists():
        checkpoint = json.loads(
            CHECKPOINT_PATH.read_text(
                encoding="utf-8"
            )
        )

        print(
            f"Resuming checkpoint: "
            f"{len(checkpoint):,}/"
            f"{len(queries):,}"
        )

    database_url = (
        get_database_url()
    )

    started_all = time.time()

    with psycopg.connect(
        database_url
    ) as conn:

        for i, query in enumerate(
            queries,
            start=1,
        ):
            qid = query["query_id"]

            if qid in checkpoint:
                continue

            query_text = query["query"]

            relevant = query[
                "relevant_document_ids"
            ]

            lexical, lexical_ms = (
                lexical_search(
                    conn,
                    query_text,
                )
            )

            dense, dense_ms = (
                dense_search(
                    conn,
                    vectors[qid],
                )
            )

            hybrid = weighted_rrf(
                lexical,
                dense,
            )

            lexical_ids = top_ids(
                lexical
            )

            dense_ids = top_ids(
                dense
            )

            hybrid_ids = top_ids(
                hybrid
            )

            checkpoint[qid] = {
                "query_id": qid,
                "query": query_text,
                "relevant_document_ids":
                    relevant,

                "lexical": {
                    "document_ids":
                        lexical_ids,
                    "latency_ms":
                        lexical_ms,
                    "metrics":
                        evaluate_ranking(
                            lexical_ids,
                            relevant,
                        ),
                },

                "dense": {
                    "document_ids":
                        dense_ids,
                    "latency_ms":
                        dense_ms,
                    "metrics":
                        evaluate_ranking(
                            dense_ids,
                            relevant,
                        ),
                },

                "hybrid": {
                    "document_ids":
                        hybrid_ids,
                    "metrics":
                        evaluate_ranking(
                            hybrid_ids,
                            relevant,
                        ),
                },
            }

            save_json(
                CHECKPOINT_PATH,
                checkpoint,
            )

            if (
                i % 10 == 0
                or i == 1
                or i == len(queries)
            ):
                elapsed = (
                    time.time()
                    - started_all
                )

                print(
                    f"[{i:>3}/{len(queries)}] "
                    f"qid={qid}  "
                    f"lex={lexical_ms:.1f}ms  "
                    f"dense={dense_ms:.1f}ms  "
                    f"elapsed={elapsed:.1f}s"
                )

    # Preserve manifest order.
    per_query = [
        checkpoint[
            q["query_id"]
        ]
        for q in queries
    ]

    lexical_summary = (
        summarize_metrics(
            per_query,
            "lexical",
        )
    )

    dense_summary = (
        summarize_metrics(
            per_query,
            "dense",
        )
    )

    hybrid_summary = (
        summarize_metrics(
            per_query,
            "hybrid",
        )
    )

    output = {
        "protocol": {
            "name":
                "msmarco-full-relevance-v1",

            "corpus":
                "Full MS MARCO passage corpus",

            "documents":
                8_841_823,

            "qrels_source":
                (
                    "sentence-transformers/"
                    "msmarco labeled-list "
                    "training split"
                ),

            "positive_qrel_coverage":
                1.0,

            "evaluation_queries":
                len(queries),

            "seed":
                manifest["seed"],

            "embedding_model":
                MODEL_NAME,

            "dimensions":
                384,

            "normalized":
                True,

            "distance":
                "cosine",

            "hnsw_m":
                16,

            "hnsw_ef_construction":
                64,

            "hnsw_ef_search":
                EF_SEARCH,

            "top_k":
                TOP_K,

            "candidate_k":
                CANDIDATE_K,

            "lexical":
                (
                    "PostgreSQL FTS "
                    "websearch_to_tsquery "
                    "+ GIN"
                ),

            "hybrid":
                "weighted reciprocal rank fusion",

            "rrf_k":
                RRF_K,

            "lexical_weight":
                LEXICAL_WEIGHT,

            "dense_weight":
                DENSE_WEIGHT,

            "weight_selection":
                (
                    "Frozen from prior "
                    "500K development experiment"
                ),
        },

        "summary": {
            "lexical":
                lexical_summary,

            "dense":
                dense_summary,

            "weighted_rrf":
                hybrid_summary,

            "latency": {
                "lexical":
                    summarize_latency(
                        per_query,
                        "lexical",
                    ),

                "dense":
                    summarize_latency(
                        per_query,
                        "dense",
                    ),
            },
        },

        "per_query":
            per_query,
    }

    save_json(
        RESULT_PATH,
        output,
    )

    print(
        "\n=== Full MS MARCO "
        "First-stage Results ==="
    )

    for name, summary in [
        (
            "Lexical",
            lexical_summary,
        ),
        (
            "Dense",
            dense_summary,
        ),
        (
            "Weighted RRF",
            hybrid_summary,
        ),
    ]:
        print(
            f"{name:<13} "
            f"MRR@10="
            f"{summary['mrr_at_10']:.4f}  "
            f"nDCG@10="
            f"{summary['ndcg_at_10']:.4f}  "
            f"Recall@10="
            f"{summary['recall_at_10']:.4f}"
        )

    print("\nLatency:")

    for method in [
        "lexical",
        "dense",
    ]:
        s = output[
            "summary"
        ]["latency"][method]

        print(
            f"{method:<8} "
            f"mean={s['mean_ms']:.1f} ms  "
            f"P50={s['p50_ms']:.1f} ms  "
            f"P95={s['p95_ms']:.1f} ms  "
            f"QPS={s['qps']:.2f}"
        )

    print(
        f"\nSaved: {RESULT_PATH}"
    )


if __name__ == "__main__":
    main()
