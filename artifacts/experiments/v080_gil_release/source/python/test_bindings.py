from backend import retrieval_cpp


def main():
    engine = retrieval_cpp.SearchEngine()

    documents = [
        (10, "machine learning and information retrieval"),
        (20, "database indexing and query optimization"),
        (30, "neural information retrieval with machine learning"),
        (40, "distributed database systems"),
    ]

    for doc_id, text in documents:
        engine.add_document(doc_id, text)

    assert engine.document_count() == 4

    query = "machine learning"

    baseline = engine.search(query, 10)
    optimized = engine.search_dense(query, 10)

    assert len(baseline) == len(optimized)

    for left, right in zip(baseline, optimized):
        assert left.doc_id == right.doc_id
        assert abs(left.score - right.score) < 1e-9

    print("Documents:", engine.document_count())
    print("Vocabulary:", engine.vocabulary_size())
    print("Postings:", engine.posting_count())

    print("\nSearch Results:")

    for rank, result in enumerate(optimized, start=1):
        print(
            f"{rank}. "
            f"Document={result.doc_id}, "
            f"BM25={result.score:.6f}"
        )

    print("\nPython Binding Test PASSED")


if __name__ == "__main__":
    main()
