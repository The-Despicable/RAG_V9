-- Migration: 003_add_metadata.sql
-- Adds document metadata and extraction job tracking

-- ========== DOCUMENT METADATA ==========
CREATE TABLE IF NOT EXISTS document_metadata (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL UNIQUE REFERENCES documents(id) ON DELETE CASCADE,
    total_pages INT,
    total_sheets INT,
    extraction_duration_ms INT,
    chunk_count INT DEFAULT 0,
    vector_count INT DEFAULT 0,
    source_hash TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_doc_metadata_document ON document_metadata(document_id);

-- ========== EXTRACTION JOBS ==========
CREATE TABLE IF NOT EXISTS extraction_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    celery_task_id TEXT UNIQUE,
    status TEXT DEFAULT 'pending',
    attempt INT DEFAULT 1,
    max_attempts INT DEFAULT 3,
    error_details TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_extraction_jobs_document ON extraction_jobs(document_id);
CREATE INDEX IF NOT EXISTS idx_extraction_jobs_status ON extraction_jobs(status);
CREATE INDEX IF NOT EXISTS idx_extraction_jobs_celery ON extraction_jobs(celery_task_id);