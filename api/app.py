from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from src.inference.predictor import ChurnPredictor
from api.schemas import CustomerData, PredictionResponse, HealthResponse, ModelInfoResponse
import uvicorn
import os
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ChurnAPI")

app = FastAPI(
    title="Churn Prediction API",
    description="API for predicting customer churn risk.",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model_predictor = None

MODEL_PATH = "artifacts/models/best_model.joblib"

@app.on_event("startup")
def load_learner():
    global model_predictor
    logger.info("Loading model...")
    try:
        if os.path.exists(MODEL_PATH):
            model_predictor = ChurnPredictor(MODEL_PATH)
            logger.info("Model loaded successfully.")
        else:
            logger.warning(f"Model not found at {MODEL_PATH}. API will return errors for predictions.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")

@app.get("/health", response_model=HealthResponse)
def health_check():
    status = "healthy" if model_predictor and model_predictor.model else "degraded (model not loaded)"
    return {"status": status, "version": "1.0.0"}

@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    if not model_predictor or not model_predictor.model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        pipeline = model_predictor.model
        # Try to infer model type from pipeline steps
        if hasattr(pipeline, 'steps'):
            model_step = pipeline.steps[-1][1]
            model_type = model_step.__class__.__name__
        else:
            model_type = str(type(pipeline))
            
        features_count = 25 # Approximate default
        
        trained_date = datetime.now().strftime("%Y-%m-%d") 
    except:
        model_type = "Unknown"
        features_count = 0
        trained_date = "Unknown"

    return {
        "model_type": model_type,
        "version": "1.0.0",
        "trained_date": trained_date, 
        "features_count": features_count
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerData):
    if not model_predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        input_dict = data.dict()
        # Handle TotalCharges explicitly if passed as None or 0 and Tenure is 0
        # The transformers usually handle logs/divisions, but clean input is good.
        if input_dict.get("TotalCharges") is None:
             input_dict["TotalCharges"] = 0 
        
        result = model_predictor.predict_single(input_dict)
        return result
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
