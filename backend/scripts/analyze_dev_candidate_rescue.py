import json
import statistics
from pathlib import Path

from backend import retrieval_cpp


BASE = Path("results/msmarco_full/relevance/dev_v067")

CACHE_PATH = BASE / "ce_candidates_dev.json"
CPP_PATH = BASE / "cpp_ce/results_dev_6980q.json"
PG_PATH = BASE / "pg_ce/results_dev_6980q.json"

CORPUS_PATH = Path("cpp/data/msmarco_full.bin")

OUTPUT_PATH = BASE / "candidate_rescue_analysis.json"

TOP_K = 50


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + ".tmp")

    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    tmp.replace(path)


def candidate_recall(candidates, relevant):
    return len(candidates & relevant) / len(relevant)


def main():
    cache_data = load_json(CACHE_PATH)
    cpp_data = load_json(CPP_PATH)
    pg_data = load_json(PG_PATH)

    cache_rows = cache_data["queries"]

    cpp_rows = {
        str(row["query_id"]): row
        for row in cpp_data["per_query"]
    }

    pg_rows = {
        str(row["query_id"]): row
        for row in pg_data["per_query"]
    }

    assert len(cache_rows) == 6980
    assert len(cpp_rows) == 6980
    assert len(pg_rows) == 6980

    print("=" * 72)
    print("v0.6.7f — CANDIDATE COMPLEMENTARITY & RESCUE")
    print("=" * 72)

    print("\nLoading C++ BM25 index...")

    engine = retrieval_cpp.SearchEngine()
    engine.load_corpus(str(CORPUS_PATH))

    print("C++ index ready.")

    results = []

    for i, item in enumerate(cache_rows, 1):
        qid = str(item["query_id"])
        query = item["query"]

        relevant = set(
            map(int, item["relevant_document_ids"])
        )

        dense = {
            int(x["document_id"])
            for x in item["dense"]["candidates"][:TOP_K]
        }

        pg_lexical = {
            int(x["document_id"])
            for x in item["lexical"]["candidates"][:TOP_K]
        }

        cpp_matches = engine.search_dense(query, TOP_K)

        cpp_lexical = {
            int(x.doc_id)
            for x in cpp_matches
        }

        pg_union = dense | pg_lexical
        cpp_union = dense | cpp_lexical

        # Verify reconstructed C++ candidates
        expected = cpp_rows[qid]["candidate_counts"]

        if len(cpp_union) != expected["union"]:
            raise RuntimeError(
                f"Candidate mismatch at query {qid}: "
                f"reconstructed={len(cpp_union)}, "
                f"original={expected['union']}"
            )

        cpp_only_relevant = (
            cpp_union - pg_union
        ) & relevant

        pg_only_relevant = (
            pg_union - cpp_union
        ) & relevant

        cpp_top10 = set(
            map(int, cpp_rows[qid]["union_ce"]["top10"])
        )

        pg_top10 = set(
            map(int, pg_rows[qid]["union_ce"]["top10"])
        )

        promoted_cpp = cpp_only_relevant & cpp_top10
        promoted_pg = pg_only_relevant & pg_top10

        results.append({
            "query_id": qid,
            "relevant_count": len(relevant),
            "pg_candidate_recall": candidate_recall(
                pg_union, relevant
            ),
            "cpp_candidate_recall": candidate_recall(
                cpp_union, relevant
            ),
            "pg_union_count": len(pg_union),
            "cpp_union_count": len(cpp_union),
            "cpp_only_relevant_count": len(cpp_only_relevant),
            "pg_only_relevant_count": len(pg_only_relevant),
            "cpp_rescued_to_top10": len(promoted_cpp),
            "pg_rescued_to_top10": len(promoted_pg),
            "cpp_only_relevant_ids": sorted(cpp_only_relevant),
            "pg_only_relevant_ids": sorted(pg_only_relevant),
        })

        if i % 500 == 0 or i == len(cache_rows):
            print(f"[{i}/{len(cache_rows)}] analyzed", flush=True)

    cpp_rescue_queries = [
        row for row in results
        if row["cpp_only_relevant_count"] > 0
    ]

    pg_rescue_queries = [
        row for row in results
        if row["pg_only_relevant_count"] > 0
    ]

    cpp_rescued_docs = sum(
        row["cpp_only_relevant_count"]
        for row in results
    )

    pg_rescued_docs = sum(
        row["pg_only_relevant_count"]
        for row in results
    )

    cpp_promoted_docs = sum(
        row["cpp_rescued_to_top10"]
        for row in results
    )

    pg_promoted_docs = sum(
        row["pg_rescued_to_top10"]
        for row in results
    )

    summary = {
        "dataset": "msmarco_passage_dev_small",
        "queries": len(results),
        "candidate_recall": {
            "postgresql": statistics.fmean(
                row["pg_candidate_recall"]
                for row in results
            ),
            "cpp": statistics.fmean(
                row["cpp_candidate_recall"]
                for row in results
            ),
        },
        "candidate_budget": {
            "postgresql_mean": statistics.fmean(
                row["pg_union_count"]
                for row in results
            ),
            "cpp_mean": statistics.fmean(
                row["cpp_union_count"]
                for row in results
            ),
        },
        "rescue": {
            "cpp_rescue_queries": len(cpp_rescue_queries),
            "pg_rescue_queries": len(pg_rescue_queries),
            "cpp_rescued_relevant_docs": cpp_rescued_docs,
            "pg_rescued_relevant_docs": pg_rescued_docs,
            "cpp_promoted_to_top10": cpp_promoted_docs,
            "pg_promoted_to_top10": pg_promoted_docs,
            "cpp_promotion_rate": (
                cpp_promoted_docs / cpp_rescued_docs
                if cpp_rescued_docs else 0.0
            ),
            "pg_promotion_rate": (
                pg_promoted_docs / pg_rescued_docs
                if pg_rescued_docs else 0.0
            ),
        },
    }

    save_json(
        OUTPUT_PATH,
        {
            "summary": summary,
            "per_query": results,
        },
    )

    print("\n=== Candidate Recall ===")
    print(
        f"PostgreSQL: "
        f"{summary['candidate_recall']['postgresql']:.6f}"
    )
    print(
        f"C++       : "
        f"{summary['candidate_recall']['cpp']:.6f}"
    )

    print("\n=== Candidate Budget ===")
    print(summary["candidate_budget"])

    print("\n=== Rescue Analysis ===")

    for key, value in summary["rescue"].items():
        print(f"{key}: {value}")

    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
