import os
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_CORPUS_PATH = (
    PROJECT_ROOT / "cpp" / "data" / "msmarco_full.bin"
)


def load_cpp_engine():
    from backend import retrieval_cpp

    corpus_path = Path(
        os.getenv(
            "CPP_CORPUS_PATH",
            str(DEFAULT_CORPUS_PATH),
        )
    ).expanduser()

    if not corpus_path.is_file():
        raise FileNotFoundError(
            f"C++ corpus file not found: {corpus_path}"
        )

    print(
        f"Loading C++ BM25 corpus: {corpus_path}",
        flush=True,
    )

    engine = retrieval_cpp.SearchEngine()

    start = time.perf_counter()

    loaded = engine.load_corpus(str(corpus_path))

    elapsed = time.perf_counter() - start

    print(
        f"C++ BM25 loaded: {loaded:,} documents "
        f"in {elapsed:.3f} seconds",
        flush=True,
    )

    print(
        f"Vocabulary: {engine.vocabulary_size():,}",
        flush=True,
    )

    print(
        f"Postings: {engine.posting_count():,}",
        flush=True,
    )

    return engine
