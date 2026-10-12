from unittest.mock import Mock

from backend.api.services import cross_encoder


def test_candidate_union_deduplicates():
    lexical = [
        (1, "doc-1", "Document one"),
        (2, "doc-2", "Document two"),
    ]

    vector = [
        (2, "doc-2", "Document two"),
        (3, "doc-3", "Document three"),
    ]

    candidates = cross_encoder.union_candidates(
        lexical,
        vector,
    )

    assert len(candidates) == 3

    by_id = {
        item["document_id"]: item
        for item in candidates
    }

    assert by_id[1]["lexical_rank"] == 1
    assert by_id[1]["vector_rank"] is None

    assert by_id[2]["lexical_rank"] == 2
    assert by_id[2]["vector_rank"] == 1

    assert by_id[3]["lexical_rank"] is None
    assert by_id[3]["vector_rank"] == 2


def test_cross_encoder_ranking(monkeypatch):
    model = Mock()
    model.predict.return_value = [0.2, 0.9, -0.5]

    monkeypatch.setattr(
        cross_encoder,
        "get_cross_encoder",
        lambda: model,
    )

    candidates = [
        {
            "document_id": i,
            "external_id": f"doc-{i}",
            "content": f"Document {i}",
            "lexical_rank": None,
            "vector_rank": i,
        }
        for i in range(1, 4)
    ]

    results = cross_encoder.rerank_candidates(
        query="test query",
        candidates=candidates,
        top_k=2,
    )

    assert len(results) == 2
    assert results[0]["document_id"] == 2
    assert results[1]["document_id"] == 1
    assert results[0]["cross_encoder_score"] == 0.9


def test_empty_candidates():
    assert cross_encoder.union_candidates([], []) == []

    assert cross_encoder.rerank_candidates(
        query="test",
        candidates=[],
    ) == []
