from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.db.database import DatabaseUnavailableError
from app.rag.retriever import search as vector_search

router = APIRouter(prefix="/search", tags=["search"])


class SearchResult(BaseModel):
    id: str
    document_id: str
    source_id: str  # правило/стаття бази знань, напр. "KB-02"
    title: str
    content: str
    distance: float


@router.post("", response_model=list[SearchResult])
async def search(question: str, limit: int = 3):
    try:
        results = vector_search(question, limit=limit)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except DatabaseUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return results
