from dataclasses import dataclass
from typing import Optional


@dataclass
class Document:
    id: int
    title: str
    filename: str
    source_type: str
    status: str


@dataclass
class DocumentChunk:
    id: int
    document_id: int
    source_id: str
    title: str
    content: str
    chunk_number: Optional[int] = None
    distance: Optional[float] = None
