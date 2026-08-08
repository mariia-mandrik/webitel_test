from dotenv import load_dotenv
from fastapi import FastAPI

from app.api import documents, chat

load_dotenv()

app = FastAPI()

app.include_router(documents.router)
app.include_router(chat.router)
