import logging

from psycopg2.extras import Json

from app.db.database import get_db_connection, DatabaseUnavailableError
from app.rag.chunker import extract_text, split_text
from app.rag.embeddings import embed_query, embed_documents

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_QUESTION_LENGTH = 1000

# Порог отсечки "нет ответа в базе": cosine distance выше — считаем,
# что релевантных статей нет, и не передаём их в генерацию (см. Часть A.3 задания).
NO_ANSWER_DISTANCE_THRESHOLD = 0.25


def search(question: str, limit: int = 3) -> list[dict]:
    if not question.strip():
        raise ValueError("question is required")

    if len(question) > MAX_QUESTION_LENGTH:
        raise ValueError(f"question must be less than {MAX_QUESTION_LENGTH} characters")

    prompt_vector = embed_query(question)

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    """
                    SELECT
                        dc.id,
                        dc.document_id,
                        dc.source_id,
                        dc.title,
                        dc.content,
                        dc.embedding <=> %(prompt_vector)s::vector AS distance
                    FROM document_chunks AS dc
                    INNER JOIN documents AS d
                        ON dc.document_id = d.id
                    WHERE d.status = 'indexed'
                    ORDER BY distance
                    LIMIT %(limit)s
                    """,
                    {"prompt_vector": prompt_vector, "limit": limit},
                )

                rows = cur.fetchall()

                return [
                    {
                        "id": row[0],
                        "document_id": row[1],
                        "source_id": row[2],
                        "title": row[3],
                        "content": row[4],
                        "distance": row[5],
                    }
                    for row in rows
                ]

            except DatabaseUnavailableError:
                raise
            except Exception:
                logging.error("Search query failed", exc_info=True)
                raise DatabaseUnavailableError(
                    "Temporary database failure. Please retry the request."
                )


def has_relevant_results(results: list[dict], threshold: float = NO_ANSWER_DISTANCE_THRESHOLD) -> bool:
    return bool(results) and results[0]["distance"] <= threshold


async def index_file(request, file) -> dict:
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_FILE_SIZE:
        raise ValueError(f"File too large (Header says {content_length} bytes)")

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise ValueError("File too large")

    text = extract_text(file.filename, content)
    if not text.strip():
        raise ValueError(
            "No extractable text found in file. If this is a scanned/image PDF, OCR is required (not supported)."
        )

    texts = split_text(text)
    vectors = embed_documents(texts)

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    """
                    INSERT INTO documents (title, filename, source_type, status)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id;
                    """,
                    (
                        file.filename.rsplit(".", 1)[0],
                        file.filename,
                        "pdf",
                        "indexed",
                    ),
                )

                document_id = cur.fetchone()[0]

                for i, (chunk, vector) in enumerate(zip(texts, vectors), start=1):
                    cur.execute(
                        """
                        INSERT INTO document_chunks (
                            document_id, chunk_number, source_id, title, content, embedding, metadata
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s);
                        """,
                        (
                            document_id,
                            i,
                            f"{file.filename}-{i}",
                            file.filename,
                            chunk,
                            vector,
                            Json({}),
                        ),
                    )

                conn.commit()

            except Exception:
                conn.rollback()
                raise

    return {"result": "success"}
