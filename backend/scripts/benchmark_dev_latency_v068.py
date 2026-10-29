import argparse
import json
import os
import random
import statistics
import time
from pathlib import Path

import numpy as np
import psycopg
from dotenv import load_dotenv

from backend import retrieval_cpp


load_dotenv(".env")

BASE = Path("results/msmarco_full/latency/dev_v068")
QUERY_PATH = Path("data/msmarco/dev/queries.dev.small.tsv")
CORPUS_PATH = Path("cpp/data/msmarco_full.bin")

TOP_K = 50

SQL = """
    SELECT
        document_id,
        ts_rank_cd(
            to_tsvector('english', content),
            websearch_to_tsquery('english', %s)
        ) AS score
    FROM msmarco_full_documents
    WHERE
        to_tsvector('english', content)
        @@ websearch_to_tsquery('english', %s)
    ORDER BY score DESC, document_id ASC
    LIMIT %s
"""

HIGH_FREQUENCY_QUERIES = [
    "the",
    "and",
    "information",
    "people",
    "government",
    "health",
    "school",
    "education",
    "business",
    "system",
    "computer",
    "technology",
    "medical",
    "history",
    "science",
    "information system",
    "health information",
    "government information",
    "people health information",
    "computer science",
    "education system",
    "medical information",
    "business information",
    "health care",
    "united states",
]


def load_queries():
    queries = []

    with QUERY_PATH.open(encoding="utf-8") as f:
        for line in f:
            qid, query = line.rstrip("\n").split("\t", 1)

            queries.append({
                "query_id": qid,
                "query": query,
            })

    assert len(queries) == 6980, len(queries)

    return queries


def percentile(values, p):
    return float(np.percentile(values, p))


def summarize(values):
    values = list(values)

    if not values:
        return {}

    total_seconds = sum(values) / 1000.0

    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values),
        "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
        "max_ms": max(values),
        "min_ms": min(values),
        "serial_qps": (
            len(values) / total_seconds
            if total_seconds > 0 else None
        ),
    }


def cpp_search(engine, query):
    start = time.perf_counter_ns()

    results = engine.search_dense(query, TOP_K)

    elapsed_ms = (
        time.perf_counter_ns() - start
    ) / 1_000_000.0

    return elapsed_ms, len(results)


def pg_search(conn, query):
    start = time.perf_counter_ns()

    with conn.cursor() as cur:
        cur.execute("SET LOCAL jit = off")

        cur.execute(
            SQL,
            (query, query, TOP_K),
        )

        rows = cur.fetchall()

    conn.commit()

    elapsed_ms = (
        time.perf_counter_ns() - start
    ) / 1_000_000.0

    return elapsed_ms, len(rows)


def benchmark_query(engine, conn, item, iteration, rng):
    query = item["query"]

    # Alternate execution order to reduce order bias.
    if rng.randint(0, 1) == 0:
        cpp_ms, cpp_count = cpp_search(engine, query)
        pg_ms, pg_count = pg_search(conn, query)
        order = "cpp_first"
    else:
        pg_ms, pg_count = pg_search(conn, query)
        cpp_ms, cpp_count = cpp_search(engine, query)
        order = "pg_first"

    return {
        "query_id": item["query_id"],
        "query": query,
        "iteration": iteration,
        "execution_order": order,
        "cpp_latency_ms": cpp_ms,
        "pg_latency_ms": pg_ms,
        "cpp_result_count": cpp_count,
        "pg_result_count": pg_count,
        "latency_ratio_pg_over_cpp": (
            pg_ms / cpp_ms
            if cpp_ms > 0 else None
        ),
    }


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + ".tmp")

    tmp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    tmp.replace(path)


def run_benchmark(
    engine,
    conn,
    queries,
    repeats,
    label,
    rng,
):
    results = []

    total = len(queries) * repeats

    print(f"\n=== {label} ===")
    print(f"Queries: {len(queries)}")
    print(f"Repeats: {repeats}")
    print(f"Total paired measurements: {total}")

    counter = 0

    for iteration in range(repeats):
        order = queries.copy()
        rng.shuffle(order)

        for item in order:
            counter += 1

            result = benchmark_query(
                engine,
                conn,
                item,
                iteration,
                rng,
            )

            results.append(result)

            if counter % 250 == 0 or counter == total:
                print(
                    f"[{counter}/{total}] completed",
                    flush=True,
                )

    cpp_values = [
        row["cpp_latency_ms"]
        for row in results
    ]

    pg_values = [
        row["pg_latency_ms"]
        for row in results
    ]

    summary = {
        "label": label,
        "queries": len(queries),
        "repeats": repeats,
        "cpp_bm25": summarize(cpp_values),
        "postgresql_fts": summarize(pg_values),
    }

    print("\nC++ BM25:")
    print(json.dumps(
        summary["cpp_bm25"],
        indent=2,
    ))

    print("\nPostgreSQL FTS:")
    print(json.dumps(
        summary["postgresql_fts"],
        indent=2,
    ))

    return results, summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dev-repeats",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--stress-repeats",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    rng = random.Random(42)

    queries = load_queries()

    stress_queries = [
        {
            "query_id": f"stress_{i}",
            "query": query,
        }
        for i, query in enumerate(
            HIGH_FREQUENCY_QUERIES,
            1,
        )
    ]

    print("=" * 72)
    print("v0.6.8 — LEXICAL RETRIEVAL LATENCY BENCHMARK")
    print("=" * 72)

    print("\nLoading C++ BM25 corpus...")

    engine = retrieval_cpp.SearchEngine()
    engine.load_corpus(str(CORPUS_PATH))

    print("C++ corpus ready.")

    database_url = os.environ.get("DATABASE_URL")

    if not database_url:
        raise RuntimeError(
            "DATABASE_URL missing from .env"
        )

    with psycopg.connect(database_url) as conn:
        print("\nPostgreSQL connected.")

        print(f"\nWarming up {args.warmup} queries...")

        warmup_queries = queries[:args.warmup]

        for item in warmup_queries:
            cpp_search(engine, item["query"])
            pg_search(conn, item["query"])

        print("Warm-up complete.")

        dev_rows, dev_summary = run_benchmark(
            engine,
            conn,
            queries,
            args.dev_repeats,
            "Official Dev",
            rng,
        )

        save_json(
            BASE / "official_dev_per_query.json",
            dev_rows,
        )

        save_json(
            BASE / "official_dev_summary.json",
            dev_summary,
        )

        stress_rows, stress_summary = run_benchmark(
            engine,
            conn,
            stress_queries,
            args.stress_repeats,
            "High-Frequency Stress Test",
            rng,
        )

        save_json(
            BASE / "high_frequency_per_query.json",
            stress_rows,
        )

        save_json(
            BASE / "high_frequency_summary.json",
            stress_summary,
        )

    print("\nBenchmark complete.")
    print(f"Results saved to: {BASE}")


if __name__ == "__main__":
    main()
