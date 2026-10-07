import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import psycopg
from sentence_transformers import SentenceTransformer

import os


TABLE_NAME = "scalability_large_documents"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CANDIDATE_K = 50
EF_SEARCH = 50


def vector_to_pg(vector):
    return "[" + ",".join(
        f"{float(x):.8f}" for x in vector
    ) + "]"


def pid_from_external_id(external_id):
    prefix = "msmarco-"

    if not external_id.startswith(prefix):
        raise ValueError(
            f"Unexpected external_id: {external_id}"
        )

    return external_id[len(prefix):]


def lexical_search_with_scores(conn, query, limit):
    sql = f"""
        WITH q AS (
            SELECT websearch_to_tsquery(
                'english',
                %s
            ) AS query
        )
        SELECT
            d.external_id,
            ts_rank_cd(
                d.search_vector,
                q.query
            ) AS score
        FROM {TABLE_NAME} AS d
        CROSS JOIN q
        WHERE d.search_vector @@ q.query
        ORDER BY
            score DESC,
            d.document_id ASC
        LIMIT %s
    """

    start = time.perf_counter()

    with conn.cursor() as cur:
        cur.execute(sql, (query, limit))
        rows = cur.fetchall()

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000.0

    return [
        (
            pid_from_external_id(row[0]),
            float(row[1]),
        )
        for row in rows
    ], elapsed_ms


def vector_search_with_scores(
    conn,
    query_vector,
    limit,
    ef_search,
):
    query_pg = vector_to_pg(query_vector)

    sql = f"""
        SELECT
            external_id,
            embedding <=> %s::vector AS distance
        FROM {TABLE_NAME}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """

    start = time.perf_counter()

    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute(
                f"SET LOCAL hnsw.ef_search = "
                f"{int(ef_search)}"
            )
            cur.execute(
                sql,
                (
                    query_pg,
                    query_pg,
                    limit,
                ),
            )
            rows = cur.fetchall()

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000.0

    # pgvector cosine distance:
    # similarity = 1 - distance
    return [
        (
            pid_from_external_id(row[0]),
            1.0 - float(row[1]),
        )
        for row in rows
    ], elapsed_ms


def encode_pairs(pairs):
    return "|".join(
        f"{pid}:{score:.10g}"
        for pid, score in pairs
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query-file",
        required=True,
    )
    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    query_path = Path(args.query_file)
    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    queries = []

    with query_path.open() as f:
        for line in f:
            line = line.strip()

            if line:
                queries.append(
                    json.loads(line)
                )

    print("=" * 72)
    print("SCORE-FUSION CANDIDATE EXTRACTION")
    print("=" * 72)
    print(f"Queries:      {len(queries)}")
    print(f"Candidate K:  {CANDIDATE_K}")
    print(f"ef_search:    {EF_SEARCH}")
    print(f"Model:        {MODEL_NAME}")
    print()

    model = SentenceTransformer(MODEL_NAME)

    query_texts = [
        row["query"]
        for row in queries
    ]

    print("Encoding queries...")

    embeddings = model.encode(
        query_texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    rows_out = []

    with psycopg.connect(
        os.environ["DATABASE_URL"]
    ) as conn:

        for i, (row, embedding) in enumerate(
            zip(queries, embeddings),
            start=1,
        ):
            lexical, lexical_ms = (
                lexical_search_with_scores(
                    conn,
                    row["query"],
                    CANDIDATE_K,
                )
            )

            vector, vector_ms = (
                vector_search_with_scores(
                    conn,
                    embedding,
                    CANDIDATE_K,
                    EF_SEARCH,
                )
            )

            rows_out.append({
                "query_id": row["query_id"],
                "query": row["query"],

                "lexical_candidates":
                    "|".join(
                        pid for pid, _ in lexical
                    ),

                "lexical_scores":
                    "|".join(
                        f"{score:.10g}"
                        for _, score in lexical
                    ),

                "vector_candidates":
                    "|".join(
                        pid for pid, _ in vector
                    ),

                "vector_scores":
                    "|".join(
                        f"{score:.10g}"
                        for _, score in vector
                    ),

                # Convenient self-contained representation.
                "lexical_pid_scores":
                    encode_pairs(lexical),

                "vector_pid_scores":
                    encode_pairs(vector),

                "lexical_latency_ms":
                    lexical_ms,

                "vector_latency_ms":
                    vector_ms,
            })

            if (
                i == 1
                or i % 100 == 0
                or i == len(queries)
            ):
                print(
                    f"[{i:4d}/{len(queries)}] "
                    f"L={len(lexical):2d} "
                    f"V={len(vector):2d}"
                )

    fieldnames = list(
        rows_out[0].keys()
    )

    with output_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows_out)

    print()
    print("=" * 72)
    print("EXTRACTION COMPLETE")
    print("=" * 72)
    print(f"Queries: {len(rows_out)}")
    print(f"Saved:   {output_path}")


if __name__ == "__main__":
    main()
