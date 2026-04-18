-- Migration: 004_add_usage_tracking.sql
-- Adds API call tracking, request tracking, and rate limit history

-- ========== API USAGE TRACKING ==========
CREATE TABLE IF NOT EXISTS api_calls (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    endpoint TEXT NOT NULL,
    method TEXT NOT NULL,
    status_code INT,
    duration_ms INT,
    error_type TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_api_calls_workspace ON api_calls(workspace_id);
CREATE INDEX IF NOT EXISTS idx_api_calls_created ON api_calls(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_api_calls_endpoint ON api_calls(endpoint, created_at DESC);

-- ========== REQUEST ID MAPPING ==========
CREATE TABLE IF NOT EXISTS request_tracking (
    request_id TEXT PRIMARY KEY,
    workspace_id UUID REFERENCES workspaces(id) ON DELETE SET NULL,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    endpoint TEXT,
    method TEXT,
    ip_address INET,
    user_agent TEXT,
    duration_ms INT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_request_tracking_workspace ON request_tracking(workspace_id);
CREATE INDEX IF NOT EXISTS idx_request_tracking_created ON request_tracking(created_at DESC);

-- ========== RATE LIMIT TRACKING ==========
CREATE TABLE IF NOT EXISTS rate_limit_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    endpoint TEXT NOT NULL,
    requests INT DEFAULT 0,
    window_start TIMESTAMP,
    window_end TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_rate_limit_user ON rate_limit_history(user_id);
CREATE INDEX IF NOT EXISTS idx_rate_limit_window ON rate_limit_history(user_id, window_start);