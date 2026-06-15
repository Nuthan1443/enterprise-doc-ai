from pydantic import BaseModel
from typing import Optional


class QueryRequest(BaseModel):
    question: str
    k: int = 4


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    chunks_used: int
    cached: bool = False        # populated in Module 8