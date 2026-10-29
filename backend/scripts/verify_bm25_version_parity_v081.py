import argparse
import importlib.util
import json
import random
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

MANIFEST = (
    ROOT / "artifacts/msmarco_full/dev_small_manifest.json"
)

CORPUS = ROOT / "cpp/data/msmarco_full.bin"

EXTENSIONS = {
    "v080": (
        ROOT
        / "artifacts/experiments/v080_gil_release"
        / "build/python_extension"
        / "retrieval_cpp.cpython-313-darwin.so"
    ),
    "v081": (
        ROOT
        / "artifacts/experiments/v081_parallel_search"
        / "build/python_extension"
        / "retrieval_cpp.cpython-313-darwin.so"
    ),
}


def load_extension(path):
    spec = importlib.util.spec_from_file_location(
        "retrieval_cpp",
        str(path),
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load extension: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        choices=EXTENSIONS,
        required=True,
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    if args.sample_size <= 0 or args.top_k <= 0:
        parser.error("sample-size and top-k must be positive")

    manifest = json.loads(
        MANIFEST.read_text(encoding="utf-8")
    )

    queries = manifest["queries"]

    if args.sample_size > len(queries):
        parser.error("sample-size exceeds available queries")

    rng = random.Random(81)

    selected = rng.sample(
        queries,
        args.sample_size,
    )

    extension = load_extension(
        EXTENSIONS[args.version]
    )

    engine = extension.SearchEngine()

    print(
        f"Loading {args.version} corpus...",
        flush=True,
    )

    start = time.perf_counter()

    loaded = engine.load_corpus(str(CORPUS))

    print(
        f"Loaded {loaded:,} documents "
        f"in {time.perf_counter() - start:.2f}s",
        flush=True,
    )

    results = []

    for index, item in enumerate(selected, 1):
        query = item["query"]

        hits = engine.search_dense(
            query,
            args.top_k,
        )

        results.append({
            "query_id": item["query_id"],
            "query": query,
            "hits": [
                {
                    "doc_id": int(hit.doc_id),
                    "score": float(hit.score),
                }
                for hit in hits
            ],
        })

        if index % 20 == 0:
            print(
                f"Completed {index}/{len(selected)}",
                flush=True,
            )

    output = (
        ROOT
        / "results/msmarco_full/parity/v081"
        / f"{args.version}_top{args.top_k}_n{args.sample_size}.json"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            {
                "version": args.version,
                "sample_size": args.sample_size,
                "top_k": args.top_k,
                "seed": 81,
                "loaded_documents": loaded,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved: {output}", flush=True)


if __name__ == "__main__":
    main()
