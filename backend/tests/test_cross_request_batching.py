import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np

from backend.api.services.dynamic_batch import DynamicBatchReranker


class RecordingCrossEncoder:
    def __init__(self):
        self.calls = []
        self.lock = threading.Lock()

    def predict(self, pairs, **kwargs):
        with self.lock:
            self.calls.append(list(pairs))

        return np.asarray(
            [float(len(text)) for _, text in pairs],
            dtype=np.float32,
        )


def make_candidates(request_id):
    return [
        {
            "document_id": i + 1,
            "external_id": f"{request_id}-{i}",
            "content": f"request-{request_id}-document-{i}",
            "lexical_rank": i + 1,
            "vector_rank": None,
        }
        for i in range(10)
    ]


def main():
    model = RecordingCrossEncoder()
    barrier = threading.Barrier(4)

    with (
        patch(
            "backend.api.services.dynamic_batch.get_cross_encoder",
            return_value=model,
        ),
        patch(
            "backend.api.services.dynamic_batch.resolve_device",
            return_value="cpu",
        ),
    ):
        scheduler = DynamicBatchReranker(
            max_requests=4,
            max_pairs=100,
            max_wait_ms=100,
            timeout_seconds=10,
        )

        try:
            def submit(index):
                barrier.wait(timeout=5)

                return scheduler.submit(
                    query=f"query-{index}",
                    candidates=make_candidates(index),
                    top_k=5,
                )

            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(submit, range(4)))

        finally:
            scheduler.close()

    print("Number of predict() calls:", len(model.calls))
    print("Pairs per predict() call:", [len(x) for x in model.calls])

    for index, (ranked, profile) in enumerate(results):
        print(
            f"Request {index}: "
            f"results={len(ranked)}, "
            f"batch_requests={profile['batch_request_count']}, "
            f"batch_pairs={profile['batch_pair_count']}"
        )

        assert len(ranked) == 5

        expected_ids = {
            f"{index}-{i}"
            for i in range(10)
        }

        assert all(
            item["external_id"] in expected_ids
            for item in ranked
        )

        expected = sorted(
            make_candidates(index),
            key=lambda item: (
                -len(item["content"]),
                item["document_id"],
            ),
        )[:5]

        assert [
            item["document_id"] for item in ranked
        ] == [
            item["document_id"] for item in expected
        ]

    assert len(model.calls) == 1, (
        "Expected one cross-request predict() call."
    )

    assert len(model.calls[0]) == 40

    assert all(
        profile["batch_request_count"] == 4
        for _, profile in results
    )

    print("\nPASS: Four requests merged into one predict() call.")
    print("PASS: Each request received correctly ranked results.")


if __name__ == "__main__":
    main()


def test_cross_request_batching():
    main()
