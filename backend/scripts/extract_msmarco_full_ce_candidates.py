import json
import os
import time
from pathlib import Path

import psycopg
from sentence_transformers import SentenceTransformer


# ============================================================
# Frozen protocol
# ============================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CANDIDATE_K = 50

# Requested operating point from the ANN study was ef=40.
# Top-50 retrieval requires ef_search >= LIMIT, so the
# effective value is 50.
REQUESTED_EF_SEARCH = 40
EFFECTIVE_EF_SEARCH = max(
    REQUESTED_EF_SEARCH,
    CANDIDATE_K,
)

MANIFEST_PATH = Path(
    "artifacts/msmarco_full/eval_manifest.json"
)

OUT_DIR = Path(
    "results/msmarco_full/relevance"
)

CHECKPOINT_PATH = (
    OUT_DIR / "ce_candidates_checkpoint.json"
)

RESULT_PATH = (
    OUT_DIR / "ce_candidates.json"
)


# ============================================================
# Utilities
# ============================================================

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


def vector_to_pg(vector):
    return (
        "["
        + ",".join(
            f"{float(x):.8f}"
            for x in vector
        )
        + "]"
    )


def load_manifest():
    data = json.loads(
        MANIFEST_PATH.read_text(
            encoding="utf-8"
        )
    )

    return data["queries"]


def load_checkpoint():
    if not CHECKPOINT_PATH.exists():
        return {}

    data = json.loads(
        CHECKPOINT_PATH.read_text(
            encoding="utf-8"
        )
    )

    if isinstance(data, dict):
        return data

    raise RuntimeError(
        "Unexpected checkpoint format."
    )


# ============================================================
# Retrieval
# ============================================================

def lexical_search(conn, query):
    sql = """
        SELECT
            document_id,
            content,
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
        ORDER BY
            score DESC,
            document_id
        LIMIT %s
    """

    start = time.perf_counter()

    with conn.cursor() as cur:
        cur.execute(
            sql,
            (
                query,
                query,
                CANDIDATE_K,
            ),
        )

        rows = cur.fetchall()

    latency_ms = (
        time.perf_counter() - start
    ) * 1000.0

    return [
        {
            "document_id": int(row[0]),
            "content": row[1],
            "score": float(row[2]),
        }
        for row in rows
    ], latency_ms


def dense_search(
    conn,
    vector_text,
):
    start = time.perf_counter()

    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SET LOCAL hnsw.ef_search =
                {EFFECTIVE_EF_SEARCH}
                """
            )

            cur.execute(
                """
                SELECT
                    document_id,
                    content,
                    1 - (
                        embedding <=> %s::vector
                    ) AS score
                FROM msmarco_full_documents
                ORDER BY
                    embedding <=> %s::vector,
                    document_id
                LIMIT %s
                """,
                (
                    vector_text,
                    vector_text,
                    CANDIDATE_K,
                ),
            )

            rows = cur.fetchall()

    latency_ms = (
        time.perf_counter() - start
    ) * 1000.0

    return [
        {
            "document_id": int(row[0]),
            "content": row[1],
            "score": float(row[2]),
        }
        for row in rows
    ], latency_ms


# ============================================================
# Main
# ============================================================

def main():
    queries = load_manifest()

    checkpoint = load_checkpoint()

    print(
        f"Frozen queries      : "
        f"{len(queries):,}"
    )
    print(
        f"Already cached      : "
        f"{len(checkpoint):,}"
    )
    print(
        f"Candidate depth     : "
        f"{CANDIDATE_K}"
    )
    print(
        f"Requested ef_search : "
        f"{REQUESTED_EF_SEARCH}"
    )
    print(
        f"Effective ef_search : "
        f"{EFFECTIVE_EF_SEARCH}"
    )

    print(
        f"\nLoading model: "
        f"{MODEL_NAME}"
    )

    model = SentenceTransformer(
        MODEL_NAME,
        device="mps",
    )

    remaining = [
        q
        for q in queries
        if str(q["query_id"])
        not in checkpoint
    ]

    if remaining:
        print(
            f"Encoding "
            f"{len(remaining):,} "
            f"remaining queries..."
        )

        embeddings = model.encode(
            [
                q["query"]
                for q in remaining
            ],
            batch_size=128,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
    else:
        embeddings = []

    database_url = os.environ[
        "DATABASE_URL"
    ]

    started = time.perf_counter()

    with psycopg.connect(
        database_url
    ) as conn:

        for i, (
            query,
            embedding,
        ) in enumerate(
            zip(
                remaining,
                embeddings,
            ),
            1,
        ):
            qid = str(
                query["query_id"]
            )

            text = query["query"]

            vector_text = vector_to_pg(
                embedding
            )

            lexical, lexical_ms = (
                lexical_search(
                    conn,
                    text,
                )
            )

            dense, dense_ms = (
                dense_search(
                    conn,
                    vector_text,
                )
            )

            lexical_ids = {
                x["document_id"]
                for x in lexical
            }

            dense_ids = {
                x["document_id"]
                for x in dense
            }

            union_ids = (
                lexical_ids
                | dense_ids
            )

            checkpoint[qid] = {
                "query_id": qid,
                "query": text,
                "relevant_document_ids": (
                    query.get(
                        "relevant_document_ids",
                        [],
                    )
                ),
                "lexical": {
                    "latency_ms": (
                        lexical_ms
                    ),
                    "candidates": lexical,
                },
                "dense": {
                    "latency_ms": (
                        dense_ms
                    ),
                    "candidates": dense,
                },
                "union": {
                    "candidate_count": (
                        len(union_ids)
                    ),
                    "document_ids": (
                        sorted(union_ids)
                    ),
                },
            }

            # Save every query so the
            # extraction is fully resumable.
            save_json(
                CHECKPOINT_PATH,
                checkpoint,
            )

            if (
                i == 1
                or i % 10 == 0
                or i == len(remaining)
            ):
                elapsed = (
                    time.perf_counter()
                    - started
                )

                print(
                    f"[{i:>3}/"
                    f"{len(remaining)}] "
                    f"qid={qid}  "
                    f"lex={len(lexical):>2} "
                    f"({lexical_ms:.1f}ms)  "
                    f"dense={len(dense):>2} "
                    f"({dense_ms:.1f}ms)  "
                    f"union={len(union_ids):>3}  "
                    f"elapsed={elapsed:.1f}s"
                )

    # Keep the final file separate from
    # the resumable checkpoint.
    output = {
        "protocol": {
            "name": (
                "msmarco-full-ce-candidates-v1"
            ),
            "corpus_documents": 8_841_823,
            "queries": len(queries),
            "embedding_model": MODEL_NAME,
            "candidate_k": CANDIDATE_K,
            "requested_ef_search": (
                REQUESTED_EF_SEARCH
            ),
            "effective_ef_search": (
                EFFECTIVE_EF_SEARCH
            ),
            "lexical": (
                "PostgreSQL FTS + "
                "websearch_to_tsquery"
            ),
            "dense": (
                "pgvector HNSW cosine"
            ),
        },
        "queries": [
            checkpoint[
                str(q["query_id"])
            ]
            for q in queries
        ],
    }

    save_json(
        RESULT_PATH,
        output,
    )

    lexical_counts = [
        len(
            item["lexical"][
                "candidates"
            ]
        )
        for item in output["queries"]
    ]

    dense_counts = [
        len(
            item["dense"][
                "candidates"
            ]
        )
        for item in output["queries"]
    ]

    union_counts = [
        item["union"][
            "candidate_count"
        ]
        for item in output["queries"]
    ]

    print(
        "\n=== CE Candidate Cache ==="
    )

    print(
        f"Queries             : "
        f"{len(output['queries']):,}"
    )

    print(
        f"Lexical candidates  : "
        f"{sum(lexical_counts):,}"
    )

    print(
        f"Dense candidates    : "
        f"{sum(dense_counts):,}"
    )

    print(
        f"Union candidates    : "
        f"{sum(union_counts):,}"
    )

    print(
        f"Mean union/query    : "
        f"{sum(union_counts) / len(union_counts):.2f}"
    )

    print(
        f"\nSaved:"
        f"\n  {CHECKPOINT_PATH}"
        f"\n  {RESULT_PATH}"
    )


if __name__ == "__main__":
    main()
