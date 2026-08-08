from fastapi import APIRouter, UploadFile, File, HTTPException, Request

from app.db.database import DatabaseUnavailableError
from app.rag.retriever import index_file

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload")
async def upload_document(request: Request, file: UploadFile = File(...)):
    try:
        result = await index_file(request, file)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except DatabaseUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {"result": result}
