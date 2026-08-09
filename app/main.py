from dotenv import load_dotenv
from fastapi import FastAPI

from app.api import documents, chat, search

load_dotenv()

app = FastAPI()

app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(search.router)
