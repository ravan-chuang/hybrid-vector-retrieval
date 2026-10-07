import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import psycopg
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


TABLE_NAME = "scalability_large_documents"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 10
CANDIDATE_K = 50
EF_SEARCH = 50
RRF_K = 60
SEED = 42

RESULTS_DIR = Path("results/relevance")
QRELS_CACHE_PATH = RESULTS_DIR / "eligible_qrels.jsonl"
COVERAGE_CACHE_PATH = RESULTS_DIR / "coverage.json"


def vector_to_pg(vector):
    return "[" + ",".join(str(float(x)) for x in vector) + "]"


def pid_from_external_id(external_id):
    prefix = "msmarco-"
    if not external_id.startswith(prefix):
        raise ValueError(f"Unexpected external_id: {external_id}")
    return external_id[len(prefix):]


def reciprocal_rank_at_k(ranked, relevant, k=10):
    relevant = set(relevant)
    for rank, pid in enumerate(ranked[:k], start=1):
        if pid in relevant:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranked, relevant, k=10):
    relevant = set(relevant)
    if not relevant:
        return 0.0
    hits = len(set(ranked[:k]) & relevant)
    return hits / len(relevant)


def ndcg_at_k(ranked, relevant, k=10):
    relevant = set(relevant)

    dcg = 0.0
    for rank, pid in enumerate(ranked[:k], start=1):
        if pid in relevant:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    return dcg / idcg


def save_qrels_cache(eligible, metadata):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with QRELS_CACHE_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in eligible:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    COVERAGE_CACHE_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"Saved qrels cache: {QRELS_CACHE_PATH}"
    )
    print(
        f"Saved coverage cache: {COVERAGE_CACHE_PATH}"
    )


def load_qrels_cache():
    if not (
        QRELS_CACHE_PATH.exists()
        and COVERAGE_CACHE_PATH.exists()
    ):
        return None

    eligible = []

    with QRELS_CACHE_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            line = line.strip()
            if line:
                eligible.append(json.loads(line))

    metadata = json.loads(
        COVERAGE_CACHE_PATH.read_text(
            encoding="utf-8"
        )
    )

    print(
        f"Loaded {len(eligible):,} eligible queries "
        f"from cache"
    )

    return eligible, metadata


def load_query_texts():
    print("Loading MS MARCO query texts...")

    dataset = load_dataset(
        "sentence-transformers/msmarco",
        "queries",
        split="train",
        streaming=True,
    )

    queries = {}

    for row in dataset:
        queries[str(row["query_id"])] = row["query"]

    print(f"Loaded {len(queries):,} query texts")
    return queries


def load_local_pids(conn):
    print("Loading local MS MARCO passage IDs from PostgreSQL...")

    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT external_id
            FROM {TABLE_NAME}
            """
        )
        pids = {
            pid_from_external_id(row[0])
            for row in cur
        }

    print(f"Loaded {len(pids):,} local passage IDs")
    return pids


def build_eligible_queries(query_texts, local_pids):
    print("Scanning labeled-list and filtering qrels...")

    dataset = load_dataset(
        "sentence-transformers/msmarco",
        "labeled-list",
        split="train",
        streaming=True,
    )

    eligible = []

    labeled_query_count = 0
    original_relevant_count = 0
    local_relevant_count = 0

    for row in dataset:
        labeled_query_count += 1

        qid = str(row["query_id"])

        relevant_all = {
            str(doc_id)
            for doc_id, label in zip(
                row["doc_ids"],
                row["labels"],
            )
            if int(label) > 0
        }

        original_relevant_count += len(relevant_all)

        relevant_local = relevant_all & local_pids
        local_relevant_count += len(relevant_local)

        query = query_texts.get(qid)

        if relevant_local and query:
            eligible.append(
                {
                    "query_id": qid,
                    "query": query,
                    "relevant_doc_ids": sorted(
                        relevant_local,
                        key=int,
                    ),
                    "original_relevant_count": len(
                        relevant_all
                    ),
                    "local_relevant_count": len(
                        relevant_local
                    ),
                }
            )

    coverage = (
        local_relevant_count / original_relevant_count
        if original_relevant_count
        else 0.0
    )

    metadata = {
        "labeled_query_count": labeled_query_count,
        "eligible_query_count": len(eligible),
        "excluded_query_count": (
            labeled_query_count - len(eligible)
        ),
        "original_relevant_judgments": (
            original_relevant_count
        ),
        "local_relevant_judgments": local_relevant_count,
        "relevant_judgment_coverage": coverage,
        "local_corpus_size": len(local_pids),
    }

    print()
    print("QRELS COVERAGE")
    print(json.dumps(metadata, indent=2))

    return eligible, metadata


def lexical_search(conn, query, limit):
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

    results = [
        pid_from_external_id(row[0])
        for row in rows
    ]

    return results, elapsed_ms


def vector_search(
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
                f"SET LOCAL hnsw.ef_search = {int(ef_search)}"
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

    results = [
        pid_from_external_id(row[0])
        for row in rows
    ]

    return results, elapsed_ms


def rrf_fuse(
    lexical_results,
    vector_results,
    top_k,
    rrf_k,
):
    scores = {}

    for rank, pid in enumerate(
        lexical_results,
        start=1,
    ):
        scores[pid] = scores.get(pid, 0.0) + (
            1.0 / (rrf_k + rank)
        )

    for rank, pid in enumerate(
        vector_results,
        start=1,
    ):
        scores[pid] = scores.get(pid, 0.0) + (
            1.0 / (rrf_k + rank)
        )

    ranked = sorted(
        scores.items(),
        key=lambda item: (-item[1], int(item[0])),
    )

    return [
        pid
        for pid, _ in ranked[:top_k]
    ]


def mean(values):
    return sum(values) / len(values) if values else 0.0


def percentile(values, p):
    if not values:
        return 0.0

    values = sorted(values)

    index = (len(values) - 1) * p
    lower = math.floor(index)
    upper = math.ceil(index)

    if lower == upper:
        return values[lower]

    fraction = index - lower

    return (
        values[lower] * (1 - fraction)
        + values[upper] * fraction
    )


def aggregate(rows, method):
    return {
        "MRR@10": mean(
            [r[f"{method}_mrr@10"] for r in rows]
        ),
        "nDCG@10": mean(
            [r[f"{method}_ndcg@10"] for r in rows]
        ),
        "Recall@10": mean(
            [r[f"{method}_recall@10"] for r in rows]
        ),
        "mean_latency_ms": mean(
            [r[f"{method}_latency_ms"] for r in rows]
        ),
        "p50_latency_ms": percentile(
            [r[f"{method}_latency_ms"] for r in rows],
            0.50,
        ),
        "p95_latency_ms": percentile(
            [r[f"{method}_latency_ms"] for r in rows],
            0.95,
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--queries",
        type=int,
        default=20,
        help="Number of eligible queries to evaluate",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    args = parser.parse_args()

    database_url = os.environ.get("DATABASE_URL")

    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is not set"
        )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("MS MARCO 500K RELEVANCE EVALUATION")
    print("=" * 70)
    print(f"top_k:       {TOP_K}")
    print(f"candidate_k: {CANDIDATE_K}")
    print(f"ef_search:   {EF_SEARCH}")
    print(f"rrf_k:       {RRF_K}")
    print(f"sample size: {args.queries}")
    print(f"seed:        {args.seed}")
    print()

    conn = psycopg.connect(database_url)
    conn.prepare_threshold = None

    try:
        cached = load_qrels_cache()

        if cached is not None:
            eligible, coverage_metadata = cached
        else:
            query_texts = load_query_texts()
            local_pids = load_local_pids(conn)

            eligible, coverage_metadata = (
                build_eligible_queries(
                    query_texts,
                    local_pids,
                )
            )

            save_qrels_cache(
                eligible,
                coverage_metadata,
            )

        if args.queries > len(eligible):
            raise ValueError(
                "Requested more queries than eligible"
            )

        rng = random.Random(args.seed)
        evaluation_queries = rng.sample(
            eligible,
            args.queries,
        )

        evaluation_queries.sort(
            key=lambda x: int(x["query_id"])
        )

        query_path = (
            RESULTS_DIR
            / "evaluation_queries.jsonl"
        )

        with query_path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            for row in evaluation_queries:
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        print()
        print("Loading embedding model...")
        model = SentenceTransformer(MODEL_NAME)

        query_strings = [
            row["query"]
            for row in evaluation_queries
        ]

        print(
            f"Encoding {len(query_strings)} queries..."
        )

        embeddings = model.encode(
            query_strings,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        rows = []

        print()
        print("Running retrieval evaluation...")

        for index, (
            item,
            embedding,
        ) in enumerate(
            zip(
                evaluation_queries,
                embeddings,
            ),
            start=1,
        ):
            relevant = set(
                item["relevant_doc_ids"]
            )

            lexical_candidates, lexical_ms = (
                lexical_search(
                    conn,
                    item["query"],
                    CANDIDATE_K,
                )
            )

            vector_candidates, vector_ms = (
                vector_search(
                    conn,
                    embedding,
                    CANDIDATE_K,
                    EF_SEARCH,
                )
            )

            fusion_start = time.perf_counter()

            hybrid_top = rrf_fuse(
                lexical_candidates,
                vector_candidates,
                TOP_K,
                RRF_K,
            )

            fusion_ms = (
                time.perf_counter()
                - fusion_start
            ) * 1000.0

            lexical_top = (
                lexical_candidates[:TOP_K]
            )
            vector_top = (
                vector_candidates[:TOP_K]
            )

            hybrid_ms = (
                lexical_ms
                + vector_ms
                + fusion_ms
            )

            row = {
                "query_id": item["query_id"],
                "query": item["query"],
                "relevant_doc_ids": "|".join(
                    sorted(relevant, key=int)
                ),
                "relevant_count": len(relevant),

                "lexical_top10": "|".join(
                    lexical_top
                ),
                "vector_top10": "|".join(
                    vector_top
                ),
                "hybrid_top10": "|".join(
                    hybrid_top
                ),

                "lexical_mrr@10":
                    reciprocal_rank_at_k(
                        lexical_top,
                        relevant,
                        TOP_K,
                    ),
                "lexical_ndcg@10":
                    ndcg_at_k(
                        lexical_top,
                        relevant,
                        TOP_K,
                    ),
                "lexical_recall@10":
                    recall_at_k(
                        lexical_top,
                        relevant,
                        TOP_K,
                    ),

                "vector_mrr@10":
                    reciprocal_rank_at_k(
                        vector_top,
                        relevant,
                        TOP_K,
                    ),
                "vector_ndcg@10":
                    ndcg_at_k(
                        vector_top,
                        relevant,
                        TOP_K,
                    ),
                "vector_recall@10":
                    recall_at_k(
                        vector_top,
                        relevant,
                        TOP_K,
                    ),

                "hybrid_mrr@10":
                    reciprocal_rank_at_k(
                        hybrid_top,
                        relevant,
                        TOP_K,
                    ),
                "hybrid_ndcg@10":
                    ndcg_at_k(
                        hybrid_top,
                        relevant,
                        TOP_K,
                    ),
                "hybrid_recall@10":
                    recall_at_k(
                        hybrid_top,
                        relevant,
                        TOP_K,
                    ),

                "lexical_latency_ms":
                    lexical_ms,
                "vector_latency_ms":
                    vector_ms,
                "hybrid_latency_ms":
                    hybrid_ms,
            }

            rows.append(row)

            print(
                f"[{index:>4}/{len(evaluation_queries)}] "
                f"qid={item['query_id']} "
                f"L={row['lexical_ndcg@10']:.3f} "
                f"V={row['vector_ndcg@10']:.3f} "
                f"H={row['hybrid_ndcg@10']:.3f}"
            )

        csv_path = (
            RESULTS_DIR
            / "per_query_results.csv"
        )

        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=rows[0].keys(),
            )
            writer.writeheader()
            writer.writerows(rows)

        summary = {
            "protocol": {
                "dataset":
                    "MS MARCO 500K subset",
                "table": TABLE_NAME,
                "model": MODEL_NAME,
                "top_k": TOP_K,
                "candidate_k": CANDIDATE_K,
                "ef_search": EF_SEARCH,
                "rrf_k": RRF_K,
                "sample_size": len(rows),
                "seed": args.seed,
            },
            "coverage": coverage_metadata,
            "lexical": aggregate(
                rows,
                "lexical",
            ),
            "vector": aggregate(
                rows,
                "vector",
            ),
            "hybrid": aggregate(
                rows,
                "hybrid",
            ),
        }

        summary_path = (
            RESULTS_DIR / "summary.json"
        )

        summary_path.write_text(
            json.dumps(
                summary,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print()
        print("=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)

        for method in [
            "lexical",
            "vector",
            "hybrid",
        ]:
            result = summary[method]

            print(
                f"\n{method.upper()}"
            )
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
                f"P50:       "
                f"{result['p50_latency_ms']:.3f} ms"
            )
            print(
                f"P95:       "
                f"{result['p95_latency_ms']:.3f} ms"
            )

        print()
        print(f"Saved: {query_path}")
        print(f"Saved: {csv_path}")
        print(f"Saved: {summary_path}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
