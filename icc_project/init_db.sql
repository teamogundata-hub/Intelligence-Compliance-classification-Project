-- ICC Database Initialization
-- Creates audit trail tables for 5-year retention compliance
-- Author: Team Ogun — ICC Product

-- Audit trail table for classification decisions
CREATE TABLE IF NOT EXISTS icc_audit_trail (
    id SERIAL PRIMARY KEY,
    audit_id VARCHAR(50) UNIQUE NOT NULL,
    customer_id VARCHAR(100),
    document_type VARCHAR(50),
    classification_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    kyc_tier VARCHAR(20) NOT NULL,
    kyc_confidence FLOAT,
    mapped_obligations JSONB,
    obligation_confidence FLOAT,
    risk_flag VARCHAR(50),
    risk_confidence FLOAT,
    requires_escalation BOOLEAN DEFAULT FALSE,
    raw_input_hash VARCHAR(64),
    processed_input TEXT,
    model_version VARCHAR(50) DEFAULT '1.0.0',
    processing_time_ms FLOAT,
    retention_expires_at TIMESTAMP,
    reviewed_by VARCHAR(100),
    review_timestamp TIMESTAMP,
    review_decision VARCHAR(50),
    review_notes TEXT,
    checksum VARCHAR(64)
);

-- System event log
CREATE TABLE IF NOT EXISTS icc_audit_log (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,
    event_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    actor VARCHAR(100),
    target_id VARCHAR(100),
    action VARCHAR(200),
    details JSONB,
    ip_address VARCHAR(45),
    user_agent TEXT,
    checksum VARCHAR(64)
);

-- Model performance metrics
CREATE TABLE IF NOT EXISTS icc_model_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_version VARCHAR(50),
    metric_name VARCHAR(100),
    metric_value FLOAT,
    dataset_slice VARCHAR(100),
    data_drift_score FLOAT
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_audit_customer ON icc_audit_trail(customer_id);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON icc_audit_trail(classification_timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_kyc_tier ON icc_audit_trail(kyc_tier);
CREATE INDEX IF NOT EXISTS idx_audit_risk_flag ON icc_audit_trail(risk_flag);
CREATE INDEX IF NOT EXISTS idx_audit_retention ON icc_audit_trail(retention_expires_at);
CREATE INDEX IF NOT EXISTS idx_event_timestamp ON icc_audit_log(event_timestamp);
CREATE INDEX IF NOT EXISTS idx_event_type ON icc_audit_log(event_type);

-- Partitioning for audit trail (by month)
CREATE OR REPLACE FUNCTION create_audit_partition() RETURNS void AS $$
DECLARE
    start_date DATE;
    end_date DATE;
BEGIN
    start_date := date_trunc('month', CURRENT_DATE);
    end_date := start_date + interval '1 month';
    EXECUTE format(
        'CREATE TABLE IF NOT EXISTS icc_audit_trail_%s PARTITION OF icc_audit_trail FOR VALUES FROM (%L) TO (%L)',
        to_char(start_date, 'YYYY_MM'),
        start_date,
        end_date
    );
END;
$$ LANGUAGE plpgsql;

-- Auto-purge function for expired records
CREATE OR REPLACE FUNCTION purge_expired_records() RETURNS integer AS $$
DECLARE
    purged_count integer;
BEGIN
    DELETE FROM icc_audit_trail
    WHERE retention_expires_at < CURRENT_TIMESTAMP;
    GET DIAGNOSTICS purged_count = ROW_COUNT;
    RETURN purged_count;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO icc_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO icc_user;
