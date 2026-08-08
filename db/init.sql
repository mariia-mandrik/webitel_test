CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS documents (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title       VARCHAR(255) NOT NULL,
    filename    VARCHAR(255) NOT NULL,
    source_type VARCHAR(20) NOT NULL,
    status      VARCHAR(20) NOT NULL DEFAULT 'uploaded',
    created_at  TIMESTAMP NOT NULL DEFAULT now(),
    updated_at  TIMESTAMP NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS document_chunks (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id   UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_number  INTEGER NOT NULL,
    source_id     VARCHAR(50) NOT NULL,
    title         VARCHAR(255),
    content       TEXT NOT NULL,
    embedding     VECTOR(1536),
    metadata      JSONB,
    created_at    TIMESTAMP NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_document_chunks_metadata
    ON document_chunks USING gin (metadata);

CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding
    ON document_chunks USING ivfflat (embedding vector_cosine_ops);
