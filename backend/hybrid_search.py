import os
import sys
import time
import psycopg
from sentence_transformers import SentenceTransformer

DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://retrieval:retrieval@127.0.0.1:5433/retrieval_db",
)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 10
CANDIDATE_K = 50
RRF_K = 60
HNSW_EF_SEARCH = 80

model = SentenceTransformer(MODEL_NAME)


def to_vector_string(vector):
    return "[" + ",".join(str(float(x)) for x in vector) + "]"


def lexical_search(cur, query, k=CANDIDATE_K):
    terms = [
        term.strip()
        for term in query.split()
        if term.strip()
    ]

    tsquery = " | ".join(terms)

    sql = """
        SELECT
            benchmark_id,
            content,
            ts_rank_cd(
                search_vector,
                to_tsquery('english', %s)
            ) AS score
        FROM benchmark_documents
        WHERE search_vector @@
              to_tsquery('english', %s)
        ORDER BY score DESC, benchmark_id
        LIMIT %s
    """

    start = time.perf_counter()

    cur.execute(
        sql,
        (tsquery, tsquery, k)
    )

    rows = cur.fetchall()

    latency_ms = (
        time.perf_counter() - start
    ) * 1000

    return rows, latency_ms

def vector_search(cur, query_vector, k=CANDIDATE_K):
    sql = """
        SELECT
            benchmark_id,
            content,
            1 - (embedding <=> %s::vector) AS similarity
        FROM benchmark_documents
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """

    start = time.perf_counter()

    cur.execute(sql, (query_vector, query_vector, k))
    rows = cur.fetchall()

    latency_ms = (time.perf_counter() - start) * 1000

    return rows, latency_ms


def reciprocal_rank_fusion(
    lexical_rows,
    vector_rows,
    rrf_k=RRF_K
):
    fused = {}

    for rank, row in enumerate(lexical_rows, start=1):
        doc_id, content, score = row

        if doc_id not in fused:
            fused[doc_id] = {
                "content": content,
                "lexical_rank": None,
                "vector_rank": None,
                "lexical_score": None,
                "vector_score": None,
                "rrf_score": 0.0,
            }

        fused[doc_id]["lexical_rank"] = rank
        fused[doc_id]["lexical_score"] = float(score)

        fused[doc_id]["rrf_score"] += (
            1.0 / (rrf_k + rank)
        )

    for rank, row in enumerate(vector_rows, start=1):
        doc_id, content, score = row

        if doc_id not in fused:
            fused[doc_id] = {
                "content": content,
                "lexical_rank": None,
                "vector_rank": None,
                "lexical_score": None,
                "vector_score": None,
                "rrf_score": 0.0,
            }

        fused[doc_id]["vector_rank"] = rank
        fused[doc_id]["vector_score"] = float(score)

        fused[doc_id]["rrf_score"] += (
            1.0 / (rrf_k + rank)
        )

    ranked = sorted(
        fused.items(),
        key=lambda item: item[1]["rrf_score"],
        reverse=True
    )

    return ranked


def hybrid_search(query):
    embedding_start = time.perf_counter()

    embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    embedding_ms = (
        time.perf_counter() - embedding_start
    ) * 1000

    query_vector = to_vector_string(embedding)

    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:

            cur.execute(
                f"SET hnsw.ef_search = {HNSW_EF_SEARCH}"
            )

            lexical_rows, lexical_ms = lexical_search(
                cur,
                query
            )

            vector_rows, vector_ms = vector_search(
                cur,
                query_vector
            )

    fusion_start = time.perf_counter()

    fused = reciprocal_rank_fusion(
        lexical_rows,
        vector_rows
    )

    fusion_ms = (
        time.perf_counter() - fusion_start
    ) * 1000

    print("\n" + "=" * 80)
    print("Hybrid Retrieval System")
    print("=" * 80)

    print(f"Query:             {query}")
    print(f"Embedding:         {embedding_ms:.3f} ms")
    print(f"Lexical retrieval: {lexical_ms:.3f} ms")
    print(f"Vector retrieval:  {vector_ms:.3f} ms")
    print(f"RRF fusion:        {fusion_ms:.3f} ms")

    print(
        f"Candidates:        "
        f"{len(lexical_rows)} lexical + "
        f"{len(vector_rows)} vector"
    )

    print("=" * 80)

    for rank, (doc_id, data) in enumerate(
        fused[:TOP_K],
        start=1
    ):
        print(f"\n[{rank}] Document {doc_id}")
        print(f"RRF score:     {data['rrf_score']:.6f}")
        print(f"Lexical rank:  {data['lexical_rank']}")
        print(f"Vector rank:   {data['vector_rank']}")

        if data["lexical_score"] is not None:
            print(
                f"Lexical score: "
                f"{data['lexical_score']:.6f}"
            )

        if data["vector_score"] is not None:
            print(
                f"Vector score:  "
                f"{data['vector_score']:.6f}"
            )

        text = data["content"].replace("\n", " ")

        if len(text) > 300:
            text = text[:300] + "..."

        print(f"Text:          {text}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]).strip()

    if not query:
        query = "technology companies and software"

    hybrid_search(query)
