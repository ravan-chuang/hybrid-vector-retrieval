from typing import Literal

from pydantic import BaseModel, Field, model_validator


SearchCorpus = Literal["benchmark", "application"]


class VectorSearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Natural-language search query",
    )
    corpus: SearchCorpus = Field(
        default="benchmark",
        description="Search corpus: benchmark or application",
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of results to return",
    )
    ef_search: int = Field(
        default=40,
        ge=10,
        le=500,
        description="HNSW search breadth",
    )


class SearchResult(BaseModel):
    document_id: int
    external_id: str
    content: str
    distance: float
    similarity: float


class VectorSearchResponse(BaseModel):
    query: str
    method: str
    top_k: int
    ef_search: int
    effective_ef_search: int
    embedding_ms: float
    retrieval_ms: float
    total_ms: float
    results: list[SearchResult]


class LexicalSearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Natural-language or keyword search query",
    )
    corpus: SearchCorpus = Field(
        default="benchmark",
        description="Search corpus: benchmark or application",
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of results to return",
    )


class LexicalSearchResult(BaseModel):
    document_id: int
    external_id: str
    content: str
    score: float


class LexicalSearchResponse(BaseModel):
    query: str
    method: str
    top_k: int
    retrieval_ms: float
    total_ms: float
    results: list[LexicalSearchResult]


class HybridSearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Natural-language hybrid search query",
    )
    corpus: SearchCorpus = Field(
        default="benchmark",
        description="Search corpus: benchmark or application",
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of final results",
    )
    candidate_k: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Candidates retrieved from each method",
    )
    ef_search: int = Field(
        default=40,
        ge=10,
        le=500,
        description="HNSW search breadth",
    )
    rrf_k: int = Field(
        default=60,
        ge=1,
        le=200,
        description="RRF rank constant",
    )

    @model_validator(mode="after")
    def validate_retrieval_parameters(self):
        if self.candidate_k < self.top_k:
            raise ValueError(
                "candidate_k must be greater than or equal to top_k"
            )

        return self


class HybridSearchResult(BaseModel):
    document_id: int
    external_id: str
    content: str
    rrf_score: float
    lexical_rank: int | None = None
    vector_rank: int | None = None


class HybridSearchResponse(BaseModel):
    query: str
    method: str
    top_k: int
    candidate_k: int
    ef_search: int
    effective_ef_search: int
    rrf_k: int
    embedding_ms: float
    lexical_ms: float
    vector_ms: float
    fusion_ms: float
    total_ms: float
    results: list[HybridSearchResult]


class SearchHistoryItem(BaseModel):
    search_id: int
    query_text: str
    search_type: str
    result_count: int | None
    latency_ms: float | None
    created_at: str
