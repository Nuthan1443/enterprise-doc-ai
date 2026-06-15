from pydantic import BaseModel
from typing import Optional


class IngestResponse(BaseModel):
    message: str
    source: str
    file_size_bytes: int
    pii_entities_found: int
    chunks_created: int = 0        # populated in Module 4
    collection_name: str = ""      # populated in Module 4


class IngestError(BaseModel):
    error: str
    source: Optional[str] = None