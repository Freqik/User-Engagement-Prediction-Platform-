from pydantic import BaseModel, Field, validator
from typing import Optional

class CustomerData(BaseModel):
    # Demographics
    gender: str = Field(..., description="Customer gender (Male/Female)")
    SeniorCitizen: int = Field(..., description="1 if senior citizen, 0 otherwise")
    Partner: str = Field(..., description="Yes/No")
    Dependents: str = Field(..., description="Yes/No")
    
    # Services
    tenure: int = Field(..., ge=0, description="Number of months of tenure")
    PhoneService: str = Field(..., description="Yes/No")
    MultipleLines: str = Field(..., description="Yes/No/No phone service")
    InternetService: str = Field(..., description="DSL/Fiber optic/No")
    OnlineSecurity: str = Field(..., description="Yes/No/No internet service")
    OnlineBackup: str = Field(..., description="Yes/No/No internet service")
    DeviceProtection: str = Field(..., description="Yes/No/No internet service")
    TechSupport: str = Field(..., description="Yes/No/No internet service")
    StreamingTV: str = Field(..., description="Yes/No/No internet service")
    StreamingMovies: str = Field(..., description="Yes/No/No internet service")
    
    # Account
    Contract: str = Field(..., description="Month-to-month/One year/Two year")
    PaperlessBilling: str = Field(..., description="Yes/No")
    PaymentMethod: str = Field(..., description="e.g. Electronic check, Mailed check...")
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: Optional[float] = Field(None, description="Total charges. If unknown/new customer, can be null or 0.")
    
    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85
            }
        }

class PredictionResponse(BaseModel):
    churn_probability: float
    risk_category: str
    
class HealthResponse(BaseModel):
    status: str
    version: str

class ModelInfoResponse(BaseModel):
    model_type: str
    version: str
    trained_date: str
    features_count: int
