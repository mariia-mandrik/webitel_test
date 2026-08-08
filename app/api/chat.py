from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.chat.service import handle_message
from app.db.database import DatabaseUnavailableError

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    sources: list[str] = []
    escalate: bool = False


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        result = handle_message(request.session_id, request.message)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except DatabaseUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return ChatResponse(**result)
