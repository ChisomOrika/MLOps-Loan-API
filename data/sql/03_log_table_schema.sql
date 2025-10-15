-- Schema for logging all incoming API requests and predictions
CREATE TABLE IF NOT EXISTS api_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    
    -- Features logged from the API request
    feature_1 DECIMAL,
    feature_2 DECIMAL,
    loan_amount INT,
    applicant_age INT,
    employment_type VARCHAR(50),
    
    -- Prediction output
    prediction INT,
    probability DECIMAL
);

-- Index on timestamp for faster monitoring queries
CREATE INDEX idx_api_logs_timestamp ON api_logs (timestamp);
