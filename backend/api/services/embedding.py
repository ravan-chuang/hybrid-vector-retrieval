def vector_to_pg(vector) -> str:
    """Serialize a NumPy embedding as a PostgreSQL pgvector literal."""
    return "[" + ",".join(map(str, vector.tolist())) + "]"
