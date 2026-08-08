import io

from docx import Document as DocxDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

# chunk_size/overlap подобраны под короткие KB-статьи (по параграфу-два на факт):
# 300 симв. вкладывает 1 пункт KB целиком, overlap 100 не рвёт границу факта пополам.
CHUNK_SIZE = 300
CHUNK_OVERLAP = 100


def extract_text(filename: str, content: bytes) -> str:
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if suffix == "docx":
        document = DocxDocument(io.BytesIO(content))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)

    if suffix in ("txt", "md"):
        return content.decode("utf-8")

    raise ValueError(f"Unsupported file type: .{suffix}")


def split_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    return splitter.split_text(text)
