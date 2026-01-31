from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import logging

logger = logging.getLogger(__name__)

class ChallengerModel(BaseEstimator, ClassifierMixin):
    """
    XGBoost Challenger Model.
    - Uses 'scale_pos_weight' for class imbalance.
    - Optimized for mixed feature types (handled by pipeline before this, or XGBoost native support).
    - We assume preprocessing happens upstream or we can bundle it. 
      For consistency with Baseline, we'll expect preprocessed features or add a simple pipeline.
    """
    def __init__(self, random_state=42, n_estimators=100, max_depth=5, learning_rate=0.1, scale_pos_weight=1.0):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.scale_pos_weight = scale_pos_weight
        
        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            scale_pos_weight=self.scale_pos_weight,
            random_state=self.random_state,
            eval_metric='logloss',
            n_jobs=-1
        )

    def fit(self, X, y):
        logger.info(f"Training Challenger Model (XGBoost) with scale_pos_weight={self.scale_pos_weight}...")
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
