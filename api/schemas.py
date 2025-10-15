from pydantic import BaseModel, Field
from typing import Literal

# Edge Case: Use Pydantic and Field validators for robust input checking
class LoanApplicant(BaseModel):
    """Schema for an incoming loan application request."""
    feature_1: float = Field(..., gt=0.0, description="Proprietary feature 1 (must be positive)")
    feature_2: float = Field(..., gt=0.0, description="Proprietary feature 2 (must be positive)")
    loan_amount: int = Field(..., gt=500, le=100000, description="Loan amount in USD")
    applicant_age: int = Field(..., gt=18, le=99, description="Age of the applicant")
    # Edge Case: Enforce known categories to prevent API errors, using Python's Literal
    employment_type: Literal['Professional', 'Student', 'Retired', 'Unemployed']

# Schema for API response
class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 for Low Risk (No Default), 1 for High Risk (Default)")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probability of the predicted class")
    model_version: str
