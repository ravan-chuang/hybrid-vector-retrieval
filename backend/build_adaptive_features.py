import argparse
import csv
import json
import math
import re
from pathlib import Path


def parse_ranking(value):
    if not value:
        return []
    return [x for x in value.split("|") if x]


def overlap_count(a, b, k):
    return len(set(a[:k]) & set(b[:k]))


def jaccard(a, b, k):
    sa = set(a[:k])
    sb = set(b[:k])
    union = sa | sb

    if not union:
        return 0.0

    return len(sa & sb) / len(union)


def reciprocal_rank_agreement(a, b, k=50):
    """
    Agreement score based only on rankings.

    Documents appearing near the top of both lists contribute more.
    No qrels or relevance scores are used.
    """
    rank_a = {
        doc_id: rank
        for rank, doc_id in enumerate(a[:k], start=1)
    }
    rank_b = {
        doc_id: rank
        for rank, doc_id in enumerate(b[:k], start=1)
    }

    common = set(rank_a) & set(rank_b)

    return sum(
        1.0 / (rank_a[doc_id] + rank_b[doc_id])
        for doc_id in common
    )


def mean_rank_difference(a, b, k=50):
    rank_a = {
        doc_id: rank
        for rank, doc_id in enumerate(a[:k], start=1)
    }
    rank_b = {
        doc_id: rank
        for rank, doc_id in enumerate(b[:k], start=1)
    }

    common = set(rank_a) & set(rank_b)

    if not common:
        return float(k)

    return sum(
        abs(rank_a[d] - rank_b[d])
        for d in common
    ) / len(common)


def tokenize_query(query):
    # Simple deterministic query-side features.
    return re.findall(r"\b\w+\b", query.lower())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", required=True)
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dev_path = Path(args.development)
    oracle_path = Path(args.oracle)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with oracle_path.open(newline="") as f:
        oracle_rows = {
            row["query_id"]: row
            for row in csv.DictReader(f)
        }

    rows_out = []

    with dev_path.open(newline="") as f:
        reader = csv.DictReader(f)

        required = {
            "query_id",
            "query",
            "lexical_candidates",
            "vector_candidates",
        }

        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Missing development columns: {sorted(missing)}"
            )

        for row in reader:
            qid = row["query_id"]

            if qid not in oracle_rows:
                raise ValueError(
                    f"Query {qid} missing from oracle table"
                )

            oracle = oracle_rows[qid]

            lexical = parse_ranking(
                row["lexical_candidates"]
            )
            vector = parse_ranking(
                row["vector_candidates"]
            )

            tokens = tokenize_query(row["query"])

            best_weights = (
                oracle["all_best_lexical_weights"]
                .split("|")
            )

            # Label only. This must NEVER be used as a feature.
            lexical_required = int(
                "0.0" not in best_weights
            )

            features = {
                "query_id": qid,

                # Query-only signals
                "query_token_count": len(tokens),
                "query_char_count": len(row["query"]),
                "query_avg_token_length": (
                    sum(len(t) for t in tokens) / len(tokens)
                    if tokens else 0.0
                ),

                # Candidate availability
                "lexical_candidate_count": len(lexical),
                "vector_candidate_count": len(vector),

                # Cross-retriever agreement
                "overlap_at_5":
                    overlap_count(lexical, vector, 5),
                "overlap_at_10":
                    overlap_count(lexical, vector, 10),
                "overlap_at_20":
                    overlap_count(lexical, vector, 20),
                "overlap_at_50":
                    overlap_count(lexical, vector, 50),

                "jaccard_at_5":
                    jaccard(lexical, vector, 5),
                "jaccard_at_10":
                    jaccard(lexical, vector, 10),
                "jaccard_at_20":
                    jaccard(lexical, vector, 20),
                "jaccard_at_50":
                    jaccard(lexical, vector, 50),

                "reciprocal_rank_agreement":
                    reciprocal_rank_agreement(
                        lexical,
                        vector,
                        50,
                    ),

                "mean_common_rank_difference":
                    mean_rank_difference(
                        lexical,
                        vector,
                        50,
                    ),

                # Training target
                "lexical_required": lexical_required,
            }

            rows_out.append(features)

    if len(rows_out) != len(oracle_rows):
        raise ValueError(
            "Development/oracle row-count mismatch"
        )

    fieldnames = list(rows_out[0].keys())

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows_out)

    positives = sum(
        row["lexical_required"]
        for row in rows_out
    )

    print("=" * 72)
    print("ADAPTIVE FEATURE TABLE")
    print("=" * 72)
    print(f"Queries:          {len(rows_out)}")
    print(
        f"Dense sufficient: {len(rows_out) - positives}"
    )
    print(f"Lexical required: {positives}")
    print()
    print("Features:")

    for name in fieldnames:
        if name not in {
            "query_id",
            "lexical_required",
        }:
            print(f"  {name}")

    print()
    print("Leakage guard:")
    forbidden_patterns = (
        "ndcg",
        "mrr",
        "recall",
        "relevant",
        "oracle",
        "best_weight",
    )

    feature_names = [
        x for x in fieldnames
        if x not in {
            "query_id",
            "lexical_required",
        }
    ]

    leaked = [
        name
        for name in feature_names
        if any(
            pattern in name.lower()
            for pattern in forbidden_patterns
        )
    ]

    if leaked:
        raise ValueError(
            f"Potential leakage features: {leaked}"
        )

    print("  PASS — no relevance-derived feature names")
    print()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
