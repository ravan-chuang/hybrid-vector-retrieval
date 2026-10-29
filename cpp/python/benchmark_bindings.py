import statistics
import time

from backend import retrieval_cpp


CORPUS_PATH = "cpp/data/msmarco_full.bin"

QUERIES = [
    "nature",
    "science",
    "history",
    "machine learning",
    "Manhattan Project",
    "heart disease",
    "database indexing",
]


def percentile(values, p):
    values = sorted(values)

    position = p * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower

    return (
        values[lower] * (1 - fraction)
        + values[upper] * fraction
    )


def main():
    engine = retrieval_cpp.SearchEngine()

    print("Loading corpus...", flush=True)

    start = time.perf_counter()
    engine.load_corpus(CORPUS_PATH)

    print(
        f"Load time: {time.perf_counter() - start:.3f} s",
        flush=True,
    )

    print(f"Documents: {engine.document_count()}")
    print()

    for query in QUERIES:
        for _ in range(10):
            engine.search_dense(query, 10)

        samples = []

        for _ in range(200):
            start = time.perf_counter()
            engine.search_dense(query, 10)
            elapsed_ms = (time.perf_counter() - start) * 1000
            samples.append(elapsed_ms)

        print(
            f"{query:20s} "
            f"P50={percentile(samples, 0.50):.3f} ms "
            f"P95={percentile(samples, 0.95):.3f} ms"
        )


if __name__ == "__main__":
    main()
