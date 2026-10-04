from pydantic import BaseModel, Field


class TagCreate(BaseModel):
    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        examples=["database"],
    )


class TagResponse(BaseModel):
    tag_id: int
    name: str
