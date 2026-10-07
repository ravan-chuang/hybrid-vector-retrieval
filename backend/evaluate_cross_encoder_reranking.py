import argparse
import json
import math
import os
import time
from pathlib import Path

import pandas as pd
import psycopg
import torch
from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K = 10


def split_ids(value):
    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value or value == "nan":
        return []

    return [
        x
        for x in value.split("|")
        if x
    ]


def external_id(pid):
    return f"msmarco-{pid}"


def mrr_at_k(ranked, relevant):
    relevant = set(relevant)

    for rank, pid in enumerate(
        ranked[:TOP_K],
        start=1,
    ):
        if pid in relevant:
            return 1.0 / rank

    return 0.0


def recall_at_k(ranked, relevant):
    relevant = set(relevant)

    if not relevant:
        return 0.0

    return (
        len(set(ranked[:TOP_K]) & relevant)
        / len(relevant)
    )


def ndcg_at_k(ranked, relevant):
    relevant = set(relevant)

    if not relevant:
        return 0.0

    dcg = 0.0

    for rank, pid in enumerate(
        ranked[:TOP_K],
        start=1,
    ):
        if pid in relevant:
            dcg += (
                1.0
                / math.log2(rank + 1)
            )

    ideal_hits = min(
        len(relevant),
        TOP_K,
    )

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(
            1,
            ideal_hits + 1,
        )
    )

    return dcg / idcg


def fetch_documents(conn, pids):
    if not pids:
        return {}

    ids = [
        external_id(pid)
        for pid in pids
    ]

    sql = """
        SELECT
            external_id,
            content
        FROM scalability_large_documents
        WHERE external_id = ANY(%s)
    """

    with conn.cursor() as cur:
        cur.execute(sql, (ids,))
        rows = cur.fetchall()

    result = {}

    for ext_id, content in rows:
        pid = ext_id.removeprefix(
            "msmarco-"
        )
        result[pid] = content

    return result


def rerank(
    model,
    query,
    candidate_ids,
    documents,
):
    usable = [
        pid
        for pid in candidate_ids
        if pid in documents
    ]

    if not usable:
        return [], 0.0

    pairs = [
        [query, documents[pid]]
        for pid in usable
    ]

    start = time.perf_counter()

    scores = model.predict(
        pairs,
        batch_size=32,
        show_progress_bar=False,
    )

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000.0

    ranked = sorted(
        zip(usable, scores),
        key=lambda x: (
            -float(x[1]),
            int(x[0]),
        ),
    )

    return [
        pid
        for pid, _ in ranked
    ], elapsed_ms


def mean(values):
    if not values:
        return 0.0

    return sum(values) / len(values)


def percentile(values, p):
    if not values:
        return 0.0

    values = sorted(values)

    index = (
        len(values) - 1
    ) * p

    lower = math.floor(index)
    upper = math.ceil(index)

    if lower == upper:
        return values[lower]

    fraction = index - lower

    return (
        values[lower]
        * (1.0 - fraction)
        + values[upper]
        * fraction
    )


def aggregate(rows, prefix):
    return {
        "MRR@10": mean(
            [
                r[f"{prefix}_mrr@10"]
                for r in rows
            ]
        ),
        "nDCG@10": mean(
            [
                r[f"{prefix}_ndcg@10"]
                for r in rows
            ]
        ),
        "Recall@10": mean(
            [
                r[f"{prefix}_recall@10"]
                for r in rows
            ]
        ),
        "mean_rerank_latency_ms":
            mean(
                [
                    r[
                        f"{prefix}_latency_ms"
                    ]
                    for r in rows
                ]
            ),
        "p50_rerank_latency_ms":
            percentile(
                [
                    r[
                        f"{prefix}_latency_ms"
                    ]
                    for r in rows
                ],
                0.50,
            ),
        "p95_rerank_latency_ms":
            percentile(
                [
                    r[
                        f"{prefix}_latency_ms"
                    ]
                    for r in rows
                ],
                0.95,
            ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input)

    if args.limit is not None:
        df = df.head(args.limit)

    print("=" * 72)
    print(
        "CROSS-ENCODER RERANKING "
        "EVALUATION"
    )
    print("=" * 72)

    print(f"Queries: {len(df)}")
    print(f"Model:   {MODEL_NAME}")
    print(
        "MPS:     "
        f"{torch.backends.mps.is_available()}"
    )

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Device:  {device}")
    print()

    model = CrossEncoder(
        MODEL_NAME,
        device=device,
    )

    print("Warming up Cross-Encoder...")

    _ = model.predict(
        [
            [
                "warmup query",
                "warmup passage for model initialization",
            ]
        ],
        batch_size=1,
        show_progress_bar=False,
    )

    if device == "mps":
        torch.mps.synchronize()

    print("Warm-up complete.")
    print()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows_out = []

    with psycopg.connect(
        os.environ["DATABASE_URL"]
    ) as conn:

        for i, row in df.iterrows():
            lexical = split_ids(
                row["lexical_candidates"]
            )

            vector = split_ids(
                row["vector_candidates"]
            )

            # Preserve deterministic candidate order:
            # Dense first, then lexical-only docs.
            union = list(vector)

            seen = set(vector)

            for pid in lexical:
                if pid not in seen:
                    union.append(pid)
                    seen.add(pid)

            documents = fetch_documents(
                conn,
                union,
            )

            relevant = split_ids(
                row["relevant_doc_ids"]
            )

            dense_ranked, dense_ms = rerank(
                model,
                row["query"],
                vector,
                documents,
            )

            union_ranked, union_ms = rerank(
                model,
                row["query"],
                union,
                documents,
            )

            rows_out.append({
                "query_id":
                    str(row["query_id"]),

                "query":
                    row["query"],

                "relevant_doc_ids":
                    row["relevant_doc_ids"],

                "dense_candidate_count":
                    len(vector),

                "union_candidate_count":
                    len(union),

                "dense_reranked_top10":
                    "|".join(
                        dense_ranked[:TOP_K]
                    ),

                "union_reranked_top10":
                    "|".join(
                        union_ranked[:TOP_K]
                    ),

                "dense_rerank_mrr@10":
                    mrr_at_k(
                        dense_ranked,
                        relevant,
                    ),

                "dense_rerank_ndcg@10":
                    ndcg_at_k(
                        dense_ranked,
                        relevant,
                    ),

                "dense_rerank_recall@10":
                    recall_at_k(
                        dense_ranked,
                        relevant,
                    ),

                "union_rerank_mrr@10":
                    mrr_at_k(
                        union_ranked,
                        relevant,
                    ),

                "union_rerank_ndcg@10":
                    ndcg_at_k(
                        union_ranked,
                        relevant,
                    ),

                "union_rerank_recall@10":
                    recall_at_k(
                        union_ranked,
                        relevant,
                    ),

                "dense_rerank_latency_ms":
                    dense_ms,

                "union_rerank_latency_ms":
                    union_ms,
            })

            done = len(rows_out)

            if (
                done == 1
                or done % 50 == 0
                or done == len(df)
            ):
                print(
                    f"[{done:4d}/{len(df)}] "
                    f"D={len(vector):2d} "
                    f"U={len(union):3d} "
                    f"D_ms={dense_ms:7.1f} "
                    f"U_ms={union_ms:7.1f}"
                )

    dense_summary = aggregate(
        rows_out,
        "dense_rerank",
    )

    union_summary = aggregate(
        rows_out,
        "union_rerank",
    )

    pd.DataFrame(
        rows_out
    ).to_csv(
        output_dir
        / "per_query_results.csv",
        index=False,
    )

    summary = {
        "protocol": {
            "model": MODEL_NAME,
            "device": device,
            "top_k": TOP_K,
            "dense_candidate_source":
                "vector top-50",
            "union_candidate_source":
                "vector top-50 union "
                "lexical top-50",
        },
        "dense_rerank":
            dense_summary,
        "union_rerank":
            union_summary,
    }

    with (
        output_dir
        / "summary.json"
    ).open("w") as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("=" * 72)
    print("RESULTS")
    print("=" * 72)

    for name, result in [
        (
            "DENSE TOP-50 -> CE",
            dense_summary,
        ),
        (
            "UNION -> CE",
            union_summary,
        ),
    ]:
        print()
        print(name)

        print(
            f"MRR@10:    "
            f"{result['MRR@10']:.4f}"
        )

        print(
            f"nDCG@10:   "
            f"{result['nDCG@10']:.4f}"
        )

        print(
            f"Recall@10: "
            f"{result['Recall@10']:.4f}"
        )

        print(
            f"P50 CE:    "
            f"{result['p50_rerank_latency_ms']:.1f} ms"
        )

        print(
            f"P95 CE:    "
            f"{result['p95_rerank_latency_ms']:.1f} ms"
        )

    print()
    print(
        "Union Δ nDCG@10 vs Dense CE: "
        f"{union_summary['nDCG@10'] - dense_summary['nDCG@10']:+.4f}"
    )

    print(
        "Union Δ Recall@10 vs Dense CE: "
        f"{union_summary['Recall@10'] - dense_summary['Recall@10']:+.4f}"
    )

    print()
    print("Saved:", output_dir)


if __name__ == "__main__":
    main()
