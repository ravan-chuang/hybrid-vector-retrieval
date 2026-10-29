import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path("data/msmarco/dev")
OUTPUT_DIR = Path("artifacts/msmarco_full")

QUERIES_PATH = DATA_DIR / "queries.dev.small.tsv"
QRELS_PATH = DATA_DIR / "qrels.dev.small.tsv"

OUTPUT_PATH = OUTPUT_DIR / "dev_small_manifest.json"


def open_queries(path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def main():
    queries = {}

    with open_queries(QUERIES_PATH) as f:
        reader = csv.reader(f, delimiter="\t")

        for row in reader:
            if len(row) != 2:
                raise ValueError(
                    f"Invalid query row: {row}"
                )

            qid, query = row

            if qid in queries:
                raise ValueError(
                    f"Duplicate query ID: {qid}"
                )

            queries[qid] = query

    qrels = defaultdict(set)
    qrel_rows = 0

    with open(QRELS_PATH, encoding="utf-8") as f:
        for line in f:
            parts = line.split()

            if len(parts) != 4:
                raise ValueError(
                    f"Invalid qrels line: {line!r}"
                )

            qid, iteration, pid, relevance = parts

            if qid not in queries:
                raise ValueError(
                    f"Qrels query not found: {qid}"
                )

            if int(relevance) > 0:
                document_id = int(pid) + 1

                if not 1 <= document_id <= 8841823:
                    raise ValueError(
                        f"Document ID out of range: {document_id}"
                    )

                qrels[qid].add(document_id)

            qrel_rows += 1

    records = []

    for qid, query in queries.items():
        records.append({
            "query_id": qid,
            "query": query,
            "relevant_document_ids": sorted(
                qrels[qid]
            ),
        })

    summary = {
        "dataset": "msmarco_passage_dev_small",
        "query_count": len(records),
        "qrel_rows": qrel_rows,
        "queries_with_positive_qrels": sum(
            bool(r["relevant_document_ids"])
            for r in records
        ),
        "document_id_mapping": "msmarco_pid_plus_1",
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "protocol": summary,
                "queries": records,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n=== MS MARCO Dev Small ===")

    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
