import os
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_CORPUS_PATH = (
    PROJECT_ROOT / "cpp" / "data" / "msmarco_full.bin"
)


def load_cpp_engine():
    extension_path = os.getenv("RETRIEVAL_CPP_EXTENSION")

    if extension_path:
        import importlib.util

        extension_path = Path(extension_path).expanduser().resolve()

        if not extension_path.is_file():
            raise FileNotFoundError(
                f"C++ extension not found: {extension_path}"
            )

        spec = importlib.util.spec_from_file_location(
            "retrieval_cpp",
            str(extension_path),
        )

        if spec is None or spec.loader is None:
            raise RuntimeError(
                f"Cannot load C++ extension: {extension_path}"
            )

        retrieval_cpp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(retrieval_cpp)

        print(
            f"C++ extension override: {extension_path}",
            flush=True,
        )
    else:
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
