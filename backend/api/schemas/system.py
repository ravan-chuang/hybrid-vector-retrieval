from pydantic import BaseModel


class IndexStats(BaseModel):
    name: str
    size: str


class SystemStatsResponse(BaseModel):
    documents: int
    embedding_dimension: int
    embedding_model: str
    database: str
    vector_extension: str
    table_size: str
    indexes_size: str
    total_size: str
    indexes: list[IndexStats]
