import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
import os
from typing import Dict, Any

class APILogger:
    """Handles connection to PostgreSQL and logging of API requests/predictions."""
    
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.engine = self._connect()
        
    def _connect(self):
        """Initializes and returns the SQLAlchemy engine."""
        try:
            return create_engine(self.db_url)
        except Exception as e:
            print(f"ERROR: Could not connect to PostgreSQL: {e}")
            return None

    def log_prediction(self, request_data: Dict[str, Any], prediction: int, probability: float):
        """Logs the input features, prediction, and probability."""
        if self.engine is None:
            print("WARNING: Cannot log prediction, database engine is null.")
            return

        # Prepare log data, ensuring all keys match the SQL schema
        log_data = {
            'timestamp': datetime.now(),
            'feature_1': request_data.get('feature_1'),
            'feature_2': request_data.get('feature_2'),
            'loan_amount': request_data.get('loan_amount'),
            'applicant_age': request_data.get('applicant_age'),
            'employment_type': request_data.get('employment_type'),
            'prediction': prediction,
            'probability': probability
        }
        
        # Edge Case: Use pandas to_sql for safer data type handling
        log_df = pd.DataFrame([log_data])
        
        try:
            log_df.to_sql('api_logs', self.engine, if_exists='append', index=False)
            # print("Prediction logged successfully.") # Commented out for production quietness
        except Exception as e:
            print(f"ERROR logging to database: {e}")

# Example usage is in api/main.py
