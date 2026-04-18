-- Migration: 002_add_billing.sql
-- Adds billing tables: subscriptions, token_ledger, razorpay_orders

-- ========== SUBSCRIPTIONS ==========
CREATE TABLE IF NOT EXISTS subscriptions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL UNIQUE REFERENCES workspaces(id) ON DELETE CASCADE,
    plan TEXT DEFAULT 'free',
    tokens_balance BIGINT DEFAULT 50000,
    tokens_limit_monthly BIGINT DEFAULT 50000,
    payment_provider TEXT,
    payment_id TEXT UNIQUE,
    renewal_date TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_subscriptions_workspace ON subscriptions(workspace_id);
CREATE INDEX IF NOT EXISTS idx_subscriptions_plan ON subscriptions(plan);
CREATE INDEX IF NOT EXISTS idx_subscriptions_renewal ON subscriptions(renewal_date);

-- ========== TOKEN LEDGER ==========
CREATE TABLE IF NOT EXISTS token_ledger (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    query_id UUID REFERENCES query_logs(id) ON DELETE SET NULL,
    tokens_deducted INT NOT NULL,
    input_tokens INT NOT NULL,
    output_tokens INT NOT NULL,
    reason TEXT,
    balance_after BIGINT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_token_ledger_workspace ON token_ledger(workspace_id);
CREATE INDEX IF NOT EXISTS idx_token_ledger_created ON token_ledger(workspace_id, created_at DESC);

-- ========== RAZORPAY ORDERS ==========
CREATE TABLE IF NOT EXISTS razorpay_orders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    razorpay_order_id TEXT UNIQUE NOT NULL,
    razorpay_payment_id TEXT UNIQUE,
    razorpay_signature TEXT,
    amount_paise INT NOT NULL,
    plan TEXT NOT NULL,
    status TEXT DEFAULT 'created',
    verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_razorpay_orders_workspace ON razorpay_orders(workspace_id);
CREATE INDEX IF NOT EXISTS idx_razorpay_orders_status ON razorpay_orders(status);