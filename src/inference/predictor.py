import joblib
import pandas as pd
import logging
import os
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ChurnPredictor:
    """
    Inference class to load trained model and serve predictions.
    Separates inference logic from API and Training code.
    """
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            logger.error(f"Model file not found at {self.model_path}")
            raise FileNotFoundError(f"Model artifacts not found at {self.model_path}")
        
        logger.info(f"Loading model from {self.model_path}...")
        try:
            self.model = joblib.load(self.model_path)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e

    def predict_single(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict for a single user (passed as dict).
        Returns dict with probability and risk category.
        """
        df = pd.DataFrame([input_data])
        return self._predict_df(df)[0]

    def predict_batch(self, input_data: list) -> list:
        """
        Predict for a batch of users.
        """
        df = pd.DataFrame(input_data)
        return self._predict_df(df)

    def _predict_df(self, df: pd.DataFrame) -> list:
        if self.model is None:
            raise ValueError("Model is not loaded.")
        
        # preprocessing is included in the pipeline
        try:
            # We assume the model is a Pipeline that handles everything
            # prediction is probability of Churn="Yes" (class 1)
            probs = self.model.predict_proba(df)[:, 1]
            
            results = []
            for p in probs:
                risk = "LOW"
                if p >= 0.6:
                    risk = "HIGH"
                elif p >= 0.3:
                    risk = "MEDIUM"
                
                results.append({
                    "churn_probability": float(p),
                    "risk_category": risk
                })
            return results
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise e
