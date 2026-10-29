import math
import re
from collections import Counter


def tokenize(text):
    # Match the current C++ ASCII tokenizer for ASCII-only inputs.
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def bm25_reference(documents, query, top_k=10, k1=1.2, b=0.75):
    if not documents or top_k <= 0:
        return []

    tokenized = {
        doc_id: tokenize(text)
        for doc_id, text in documents.items()
    }

    N = len(tokenized)

    avgdl = sum(
        len(tokens) for tokens in tokenized.values()
    ) / N

    if avgdl == 0:
        return []

    query_terms = set(tokenize(query))

    scores = {}

    for doc_id, tokens in tokenized.items():
        frequencies = Counter(tokens)
        score = 0.0

        for term in query_terms:
            tf = frequencies.get(term, 0)

            if tf == 0:
                continue

            df = sum(
                term in doc_tokens
                for doc_tokens in tokenized.values()
            )

            idf = math.log(
                1 + (N - df + 0.5) / (df + 0.5)
            )

            dl = len(tokens)

            denominator = (
                tf + k1 * (1 - b + b * dl / avgdl)
            )

            score += (
                idf * tf * (k1 + 1) / denominator
            )

        if score > 0:
            scores[doc_id] = score

    return sorted(
        scores.items(),
        key=lambda item: (-item[1], item[0])
    )[:top_k]


if __name__ == "__main__":
    documents = {
        1: "natural disaster causes damage",
        2: "disaster recovery and prevention",
        3: "nature and environment",
    }

    results = bm25_reference(
        documents,
        "disaster recovery",
        top_k=3
    )

    print("Python BM25 reference results:")

    for doc_id, score in results:
        print(f"doc_id={doc_id} BM25={score:.6f}")
