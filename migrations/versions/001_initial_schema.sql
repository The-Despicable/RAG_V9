-- Migration: 001_initial_schema.sql
-- Creates the base tables for workspaces, users, documents, chunks, and query logs

-- ========== WORKSPACES ==========
CREATE TABLE IF NOT EXISTS workspaces (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_workspaces_created ON workspaces(created_at DESC);

-- ========== USERS ==========
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    is_admin BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_users_workspace ON users(workspace_id);

-- ========== DOCUMENTS ==========
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    status TEXT DEFAULT 'uploaded',
    error_message TEXT,
    extraction_method TEXT,
    batch_id UUID,
    batch_name TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    processed_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_documents_workspace ON documents(workspace_id);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_documents_created ON documents(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_documents_batch ON documents(batch_id);

-- ========== CHUNKS ==========
CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    embedding_vector TEXT,
    metadata JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);

-- ========== BM25 INDEX ==========
CREATE TABLE IF NOT EXISTS bm25_index (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    chunk_id UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    tsvector tsvector NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_bm25_workspace ON bm25_index(workspace_id);
CREATE INDEX IF NOT EXISTS idx_bm25_tsv ON bm25_index USING gin(tsvector);

-- ========== QUERY LOGS ==========
CREATE TABLE IF NOT EXISTS query_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    query TEXT NOT NULL,
    retrieved_count INT DEFAULT 0,
    response_tokens INT DEFAULT 0,
    input_tokens INT DEFAULT 0,
    response_time_ms INT DEFAULT 0,
    confidence_score FLOAT DEFAULT 0.0,
    provider_used TEXT,
    feedback TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_query_logs_workspace ON query_logs(workspace_id);
CREATE INDEX IF NOT EXISTS idx_query_logs_created ON query_logs(workspace_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_query_logs_user ON query_logs(user_id, created_at DESC);