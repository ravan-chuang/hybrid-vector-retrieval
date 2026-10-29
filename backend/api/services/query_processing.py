"""Lexical query processing."""

import re

STOPWORDS = {
    "a", "an", "the",
    "and", "or", "but",
    "is", "are", "was", "were",
    "be", "been", "being",
    "am",
    "of", "to", "in", "on",
    "at", "by", "for", "from",
    "with", "as",
    "it", "its",
    "this", "that", "these", "those",
    "what", "which", "who", "whom",
    "where", "when", "why", "how",
    "do", "does", "did",
    "can", "could", "would", "should",
    "will", "shall",
    "has", "have", "had",
    "i", "you", "he", "she",
    "we", "they",
    "my", "your", "his", "her",
    "our", "their",
}

def filter_stopwords(query: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9]+", query.lower())

    return " ".join(
        token for token in tokens
        if token not in STOPWORDS
    )
