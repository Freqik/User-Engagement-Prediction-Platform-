from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
import logging

logger = logging.getLogger(__name__)

class BaselineModel(BaseEstimator, ClassifierMixin):
    """
    Logistic Regression Baseline.
    Uses 'class_weight="balanced"' to handle churn imbalance.
    Includes StandardScaling as LR is sensitive to scale.
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(class_weight='balanced', random_state=self.random_state, solver='liblinear'))
        ])

    def fit(self, X, y):
        logger.info("Training Baseline Model (LogisticRegression)...")
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
