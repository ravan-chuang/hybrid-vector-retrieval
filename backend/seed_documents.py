import os
import psycopg
from sentence_transformers import SentenceTransformer

DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://retrieval:retrieval@127.0.0.1:5433/retrieval_db",
)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print(f"Loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

documents = [
    {
        "title": "Introduction to Information Retrieval",
        "author": "Demo",
        "chunks": [
            "Information retrieval is the process of finding relevant information from a large collection of documents.",
            "Search engines rank documents according to their relevance to a user's query.",
            "Dense retrieval represents queries and documents as vectors in an embedding space."
        ]
    },
    {
        "title": "Database Systems",
        "author": "Demo",
        "chunks": [
            "A relational database organizes data into tables consisting of rows and columns.",
            "PostgreSQL is an open source relational database management system.",
            "Database indexes improve query performance by reducing the amount of data that must be scanned."
        ]
    },
    {
        "title": "Machine Learning",
        "author": "Demo",
        "chunks": [
            "Machine learning algorithms learn patterns from data.",
            "Neural networks consist of layers of interconnected artificial neurons.",
            "Embeddings map complex objects such as text into numerical vector representations."
        ]
    }
]

with psycopg.connect(DB_URL) as conn:
    with conn.cursor() as cur:

        # Allow this script to be executed repeatedly.
        cur.execute("""
            DELETE FROM documents
            WHERE author = 'Demo'
        """)

        for document in documents:

            cur.execute(
                """
                INSERT INTO documents (title, author)
                VALUES (%s, %s)
                RETURNING document_id
                """,
                (document["title"], document["author"])
            )

            document_id = cur.fetchone()[0]

            embeddings = model.encode(
                document["chunks"],
                normalize_embeddings=True
            )

            for chunk_index, (content, embedding) in enumerate(
                zip(document["chunks"], embeddings)
            ):
                vector_string = "[" + ",".join(
                    str(float(x)) for x in embedding
                ) + "]"

                cur.execute(
                    """
                    INSERT INTO document_chunks
                        (document_id, chunk_index, content, embedding)
                    VALUES (%s, %s, %s, %s::vector)
                    """,
                    (
                        document_id,
                        chunk_index,
                        content,
                        vector_string
                    )
                )

                print(
                    f"Inserted document={document_id}, "
                    f"chunk={chunk_index}, "
                    f"dimension={len(embedding)}"
                )

    conn.commit()

print("\nDocuments and 384-d embeddings inserted successfully.")
