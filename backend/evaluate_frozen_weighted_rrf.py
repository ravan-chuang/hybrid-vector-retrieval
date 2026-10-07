import argparse
import csv
import json
import math
from pathlib import Path


TOP_K = 10
RRF_K = 60


def parse_ranking(value):
    if not value:
        return []
    return [x for x in value.split("|") if x]


def parse_relevant(value):
    if not value:
        return set()
    return {x for x in value.split("|") if x}


def mrr_at_k(ranking, relevant, k=TOP_K):
    for rank, pid in enumerate(ranking[:k], start=1):
        if pid in relevant:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranking, relevant, k=TOP_K):
    if not relevant:
        return 0.0
    hits = len(set(ranking[:k]) & relevant)
    return hits / len(relevant)


def ndcg_at_k(ranking, relevant, k=TOP_K):
    if not relevant:
        return 0.0

    dcg = 0.0
    for rank, pid in enumerate(ranking[:k], start=1):
        if pid in relevant:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(relevant), k)
    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    return dcg / idcg if idcg > 0 else 0.0


def weighted_rrf(
    lexical,
    vector,
    lexical_weight,
    vector_weight,
    rrf_k=RRF_K,
    top_k=TOP_K,
):
    scores = {}

    if lexical_weight > 0:
        for rank, pid in enumerate(lexical, start=1):
            scores[pid] = scores.get(pid, 0.0) + (
                lexical_weight / (rrf_k + rank)
            )

    if vector_weight > 0:
        for rank, pid in enumerate(vector, start=1):
            scores[pid] = scores.get(pid, 0.0) + (
                vector_weight / (rrf_k + rank)
            )

    ranked = sorted(
        scores.items(),
        key=lambda item: (-item[1], item[0]),
    )

    return [pid for pid, _ in ranked[:top_k]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--selected-weight", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    weight_path = Path(args.selected_weight)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with weight_path.open() as f:
        selected = json.load(f)

    lexical_weight = float(selected["lexical_weight"])
    vector_weight = float(selected["vector_weight"])

    print("=" * 70)
    print("FROZEN WEIGHTED RRF HOLDOUT EVALUATION")
    print("=" * 70)
    print(f"Input:          {input_path}")
    print(f"Selected from:  {weight_path}")
    print(f"Lexical weight: {lexical_weight}")
    print(f"Vector weight:  {vector_weight}")
    print(f"RRF k:          {RRF_K}")
    print(f"Top k:          {TOP_K}")
    print()

    rows_out = []
    mrr_values = []
    ndcg_values = []
    recall_values = []

    with input_path.open(newline="") as f:
        reader = csv.DictReader(f)

        required = {
            "query_id",
            "relevant_doc_ids",
            "lexical_candidates",
            "vector_candidates",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}"
            )

        for row in reader:
            lexical = parse_ranking(row["lexical_candidates"])
            vector = parse_ranking(row["vector_candidates"])
            relevant = parse_relevant(row["relevant_doc_ids"])

            ranking = weighted_rrf(
                lexical,
                vector,
                lexical_weight,
                vector_weight,
            )

            mrr = mrr_at_k(ranking, relevant)
            ndcg = ndcg_at_k(ranking, relevant)
            recall = recall_at_k(ranking, relevant)

            mrr_values.append(mrr)
            ndcg_values.append(ndcg)
            recall_values.append(recall)

            rows_out.append(
                {
                    "query_id": row["query_id"],
                    "mrr@10": mrr,
                    "ndcg@10": ndcg,
                    "recall@10": recall,
                    "weighted_rrf_top10": "|".join(ranking),
                }
            )

    n = len(rows_out)
    if n == 0:
        raise ValueError("No evaluation rows found")

    summary = {
        "protocol": "frozen development-selected weighted RRF",
        "query_count": n,
        "lexical_weight": lexical_weight,
        "vector_weight": vector_weight,
        "rrf_k": RRF_K,
        "top_k": TOP_K,
        "mrr@10": sum(mrr_values) / n,
        "ndcg@10": sum(ndcg_values) / n,
        "recall@10": sum(recall_values) / n,
        "input": str(input_path),
        "selected_weight_source": str(weight_path),
    }

    per_query_path = output_dir / "per_query_results.csv"
    summary_path = output_dir / "summary.json"

    with per_query_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "query_id",
                "mrr@10",
                "ndcg@10",
                "recall@10",
                "weighted_rrf_top10",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_out)

    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print("FINAL FROZEN-WEIGHT RESULTS")
    print("-" * 70)
    print(f"Queries:   {n}")
    print(f"MRR@10:    {summary['mrr@10']:.4f}")
    print(f"nDCG@10:   {summary['ndcg@10']:.4f}")
    print(f"Recall@10: {summary['recall@10']:.4f}")
    print()
    print(f"Saved: {per_query_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
