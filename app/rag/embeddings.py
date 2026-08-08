from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()


def embed_query(text: str) -> list[float]:
    return embeddings.embed_query(text)


def embed_documents(texts: list[str]) -> list[list[float]]:
    return embeddings.embed_documents(texts)
