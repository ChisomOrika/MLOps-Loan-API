import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()
DB_URL = os.getenv("DB_URL", "postgresql://user:password@localhost:5432/mlops_db")

# --- Edge Case Simulation ---
def calculate_feature_drift(df_live: pd.DataFrame, feature_name: str, threshold: float = 0.2):
    """
    Simulates feature drift detection using a simple mean comparison.
    In a real system, you'd use statistical methods like Population Stability Index (PSI).
    """
    # Use a reference value (e.g., mean from training set, here hardcoded)
    if feature_name == 'applicant_age':
        reference_mean = 38.4 # Simulated training mean
    elif feature_name == 'loan_amount':
        reference_mean = 14500.0
    else:
        return "N/A", "N/A" # Skip other features

    live_mean = df_live[feature_name].mean()
    
    # Check for a significant shift in the mean (20% threshold)
    drift_percent = abs(live_mean - reference_mean) / reference_mean
    
    drift_status = "ALERT" if drift_percent > threshold else "OK"
    
    return drift_status, drift_percent

def run_monitoring_report(lookback_days: int = 7):
    """Queries recent live data and performs basic checks."""
    engine = create_engine(DB_URL)
    
    start_time = datetime.now() - timedelta(days=lookback_days)
    print(f"--- Running Monitoring Report for past {lookback_days} days (Since {start_time.strftime('%Y-%m-%d')}) ---")

    # 1. Query Live Data
    try:
        sql_query = text(f"""
            SELECT * FROM api_logs 
            WHERE timestamp >= :start_time
        """)
        
        df_live = pd.read_sql(sql_query, engine, params={'start_time': start_time})
        
        if df_live.empty:
            print("STATUS: No log data found for the period.")
            return

    except Exception as e:
        print(f"ERROR querying log data: {e}")
        return

    print(f"STATUS: {len(df_live)} predictions analyzed.")

    # 2. Prediction Drift Check (Check balance of predictions)
    pred_counts = df_live['prediction'].value_counts(normalize=True).get(1, 0)
    print(f"\n--- Prediction Metrics ---")
    print(f"Percentage of High Risk (Prediction=1): {pred_counts:.2%}")
    # Edge Case: If high risk exceeds 50%, might be an issue (or a bad month)
    if pred_counts > 0.5:
        print("WARNING: High volume of high-risk predictions detected.")

    # 3. Feature Drift Check
    print(f"\n--- Feature Drift Checks (against Training Mean) ---")
    for feature in ['applicant_age', 'loan_amount']:
        status, percent = calculate_feature_drift(df_live, feature)
        print(f"  {feature}: Mean Live={df_live[feature].mean():.2f} | Drift Status: {status} ({percent:.2%})")

    # 4. Data Quality Check (Missing Values)
    missing = df_live.isnull().sum().sum()
    print(f"\n--- Data Quality Check ---")
    print(f"Total Missing Values in Features: {missing}")
    if missing > 0:
        print("ALERT: Missing values detected in live API logs.")

if __name__ == '__main__':
    run_monitoring_report(lookback_days=365) # Run on 1 year of simulated data
