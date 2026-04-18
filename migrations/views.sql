-- SQL Views for Analytics Dashboards

-- ========== DAILY ACTIVE USERS ==========
CREATE OR REPLACE VIEW daily_active_users AS
SELECT 
    DATE(ql.created_at) as date,
    COUNT(DISTINCT ql.workspace_id) as active_workspaces,
    COUNT(DISTINCT ql.user_id) as active_users
FROM query_logs ql
GROUP BY DATE(ql.created_at)
ORDER BY date DESC;

-- ========== DAILY TOKEN BURN ==========
CREATE OR REPLACE VIEW daily_token_burn AS
SELECT 
    DATE(tl.created_at) as date,
    SUM(tl.tokens_deducted) as total_tokens_deducted,
    SUM(tl.input_tokens) as total_input_tokens,
    SUM(tl.output_tokens) as total_output_tokens,
    COUNT(DISTINCT tl.workspace_id) as workspaces_active
FROM token_ledger tl
GROUP BY DATE(tl.created_at)
ORDER BY date DESC;

-- ========== PROVIDER USAGE STATS ==========
CREATE OR REPLACE VIEW provider_usage_stats AS
SELECT 
    provider_used,
    COUNT(*) as query_count,
    AVG(response_time_ms) as avg_latency_ms,
    AVG(response_tokens) as avg_output_tokens,
    SUM(response_tokens) as total_output_tokens
FROM query_logs
WHERE created_at > NOW() - INTERVAL '7 days'
GROUP BY provider_used
ORDER BY query_count DESC;

-- ========== FAILED DOCUMENT UPLOADS ==========
CREATE OR REPLACE VIEW failed_uploads AS
SELECT 
    d.id,
    d.filename,
    d.error_message,
    d.created_at,
    COUNT(ej.id) as retry_attempts
FROM documents d
LEFT JOIN extraction_jobs ej ON d.id = ej.document_id
WHERE d.status = 'failed'
GROUP BY d.id, d.filename, d.error_message, d.created_at
ORDER BY d.created_at DESC;

-- ========== CONVERSION FUNNEL ==========
CREATE OR REPLACE VIEW conversion_funnel AS
SELECT 
    'free' as segment,
    COUNT(*) as count
FROM subscriptions
WHERE plan = 'free'
UNION ALL
SELECT 
    'pro' as segment,
    COUNT(*) as count
FROM subscriptions
WHERE plan = 'pro'
UNION ALL
SELECT 
    'enterprise' as segment,
    COUNT(*) as count
FROM subscriptions
WHERE plan = 'enterprise';