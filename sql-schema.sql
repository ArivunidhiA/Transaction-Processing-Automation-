-- Transaction Processing Database Schema

-- Transactions table
CREATE TABLE transactions (
    transaction_id TEXT PRIMARY KEY,
    timestamp DATETIME,
    amount FLOAT,
    account_from TEXT,
    account_to TEXT,
    transaction_type TEXT,
    status TEXT,
    risk_score FLOAT,
    processing_time FLOAT,
    validation_status TEXT
);

-- Transaction logs for audit trail
CREATE TABLE transaction_logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id TEXT,
    timestamp DATETIME,
    action TEXT,
    status TEXT,
    message TEXT
);

-- Create index for performance
CREATE INDEX idx_transaction_timestamp 
ON transactions(timestamp);

CREATE INDEX idx_transaction_status 
ON transactions(status);

-- Views for PowerBI reporting
CREATE VIEW vw_transaction_metrics AS
SELECT 
    strftime('%Y-%m-%d', timestamp) as date,
    COUNT(*) as total_transactions,
    AVG(processing_time) as avg_processing_time,
    SUM(CASE WHEN status = 'APPROVED' THEN 1 ELSE 0 END) as approved_count,
    SUM(CASE WHEN status = 'REJECTED' THEN 1 ELSE 0 END) as rejected_count,
    AVG(risk_score) as avg_risk_score
FROM transactions
GROUP BY strftime('%Y-%m-%d', timestamp);

CREATE VIEW vw_sla_compliance AS
SELECT 
    strftime('%Y-%m-%d', timestamp) as date,
    COUNT(*) as total_transactions,
    AVG(processing_time) as avg_processing_time,
    SUM(CASE WHEN processing_time <= 5.0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as sla_compliance_rate
FROM transactions
GROUP BY strftime('%Y-%m-%d', timestamp);
