from pydantic import BaseModel, Field


class DocumentCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=1)
    author: str | None = Field(default=None, max_length=255)
    source: str | None = Field(default=None, max_length=500)
    publication_year: int | None = Field(
        default=None,
        ge=0,
        le=2100,
    )


class DocumentUpdate(BaseModel):
    title: str | None = Field(
        default=None,
        min_length=1,
        max_length=500,
    )
    content: str | None = Field(
        default=None,
        min_length=1,
    )
    author: str | None = Field(default=None, max_length=255)
    source: str | None = Field(default=None, max_length=500)
    publication_year: int | None = Field(
        default=None,
        ge=0,
        le=2100,
    )


class DocumentResponse(BaseModel):
    document_id: int
    title: str
    content: str
    author: str | None
    source: str | None
    publication_year: int | None
    created_at: str
