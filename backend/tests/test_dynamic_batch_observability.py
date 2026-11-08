"""Tests for dynamic batching queue-wait decomposition."""

from unittest.mock import patch

import numpy as np
import pytest

from backend.api.services.dynamic_batch import (
    DynamicBatchReranker,
)


class FakeModel:
    def predict(
        self,
        pairs,
        batch_size,
        show_progress_bar,
        convert_to_numpy,
    ):
        return np.asarray(
            [float(i) for i in range(len(pairs))],
            dtype=np.float32,
        )


def make_candidates():
    return [
        {
            "document_id": 1,
            "external_id": "doc-1",
            "content": "first document",
        },
        {
            "document_id": 2,
            "external_id": "doc-2",
            "content": "second document",
        },
    ]


def test_queue_wait_decomposition():
    with (
        patch(
            "backend.api.services.dynamic_batch.get_cross_encoder",
            return_value=FakeModel(),
        ),
        patch(
            "backend.api.services.dynamic_batch.resolve_device",
            return_value="cpu",
        ),
    ):
        scheduler = DynamicBatchReranker(
            max_requests=1,
            max_pairs=10,
            max_wait_ms=0,
            timeout_seconds=5,
        )

        try:
            ranked, profile = scheduler.submit(
                query="test query",
                candidates=make_candidates(),
                top_k=2,
            )
        finally:
            scheduler.close()

    queue_wait = profile["batch_queue_wait_ms"]
    backlog_wait = profile["batch_backlog_wait_ms"]
    formation_wait = profile["batch_formation_wait_ms"]

    assert queue_wait >= 0
    assert backlog_wait >= 0
    assert formation_wait >= 0

    assert queue_wait == pytest.approx(
        backlog_wait + formation_wait,
        abs=0.002,
    )

    assert len(ranked) == 2
    assert ranked[0]["document_id"] == 2


def test_empty_candidates_have_complete_profile():
    scheduler = DynamicBatchReranker(
        max_requests=1,
        max_pairs=10,
    )

    try:
        ranked, profile = scheduler.submit(
            query="test",
            candidates=[],
            top_k=1,
        )
    finally:
        scheduler.close()

    assert ranked == []
    assert profile["batch_queue_wait_ms"] == 0.0
    assert profile["batch_backlog_wait_ms"] == 0.0
    assert profile["batch_formation_wait_ms"] == 0.0
