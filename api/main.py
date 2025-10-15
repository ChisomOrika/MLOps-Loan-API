import pandas as pd
from fastapi import FastAPI, HTTPException
from api.schemas import LoanApplicant, PredictionResponse
from src.data_logger import APILogger
import joblib
from dotenv import load_dotenv
import os
import numpy as np

# Load environment variables (especially DB_URL)
load_dotenv()

# --- Configuration ---
MODEL_PATH = "model/loan_model.pkl"
DB_URL = os.getenv("DB_URL", "postgresql://user:password@localhost:5432/mlops_db") # Default placeholder
MODEL_VERSION = "1.0.0"

app = FastAPI(
    title="Loan Default Prediction API",
    description="Real-time service for risk assessment using a trained ML model.",
    version=MODEL_VERSION
)

# Global resources
model_pipeline = None
api_logger = None

@app.on_event("startup")
def load_resources():
    """Load the model and initialize the database logger on startup."""
    global model_pipeline, api_logger
    
    # Load Model (Critical Resource)
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run src/train.py first!")
        model_pipeline = joblib.load(MODEL_PATH)
        print("INFO: Model pipeline loaded successfully.")
    except Exception as e:
        print(f"FATAL ERROR: Failed to load model: {e}")
        model_pipeline = None # Set to None to trigger 503 errors on predict

    # Initialize Logger (Logging is non-critical, so errors are handled by APILogger itself)
    api_logger = APILogger(db_url=DB_URL)


@app.get("/health", summary="Health Check")
def health_check():
    """Check API status and model availability."""
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    if api_logger.engine is None:
        print("WARNING: Database connection failed.")
        # Logging failure is not a 503, but is a warning
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict", response_model=PredictionResponse, summary="Get Loan Default Prediction")
def predict(applicant: LoanApplicant):
    """
    Takes loan application features and returns the model's prediction 
    and probability of default (1).
    """
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model service is unavailable.")

    # 1. Prepare Data for Model
    # Convert Pydantic model to pandas DataFrame, matching training data column order
    input_data_dict = applicant.model_dump()
    df_input = pd.DataFrame([input_data_dict])
    
    # 2. Make Prediction
    try:
        prediction_class = int(model_pipeline.predict(df_input)[0])
        # Get the probability of the POSITIVE class (1: Default)
        prediction_prob = float(model_pipeline.predict_proba(df_input)[:, 1][0])
    except Exception as e:
        # Edge Case: Model prediction failure (e.g., feature mismatch) -> 500 error
        print(f"MODEL PREDICTION ERROR: {e}")
        raise HTTPException(status_code=500, detail="Internal model prediction error.")

    # 3. Log Prediction (Asynchronous/Non-blocking logging is best practice)
    # Logging is done in a separate thread/process in production, but synchronously here for simplicity
    if api_logger:
        api_logger.log_prediction(input_data_dict, prediction_class, prediction_prob)

    # 4. Return Response
    return PredictionResponse(
        prediction=prediction_class,
        probability=round(prediction_prob, 4),
        model_version=MODEL_VERSION
    )
