"""Verify serialized Cross-Encoder initialization."""

import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock
from unittest.mock import patch

from backend.api.services import cross_encoder


def test_concurrent_initialization_only_once():
    workers = 16
    barrier = Barrier(workers)

    created = []
    creation_lock = Lock()
    model = object()

    def fake_constructor(*args, **kwargs):
        with creation_lock:
            created.append(1)

        time.sleep(0.05)
        return model

    # Ensure the test starts with an empty model cache.
    cross_encoder._load_cross_encoder.cache_clear()

    try:
        with (
            patch.object(
                cross_encoder,
                "CrossEncoder",
                side_effect=fake_constructor,
            ),
            patch.object(
                cross_encoder,
                "resolve_device",
                return_value="cpu",
            ),
        ):

            def call_model(_):
                barrier.wait()
                return cross_encoder.get_cross_encoder()

            with ThreadPoolExecutor(
                max_workers=workers
            ) as executor:
                results = list(
                    executor.map(
                        call_model,
                        range(workers),
                    )
                )

        assert len(created) == 1
        assert all(result is model for result in results)

    finally:
        cross_encoder._load_cross_encoder.cache_clear()
