import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import logging

logger = logging.getLogger(__name__)

class TenureBinning(BaseEstimator, TransformerMixin):
    """
    Groups customers into tenure cohorts to capture lifecycle stages.
    
    Business Logic:
    - **New (0-12 months)**: High risk of churn as they haven't formed a habit/loyalty.
    - **Early (12-24 months)**: Contract renewal periods often trigger churn here.
    - **Loyal (24-60 months)**: Stable users, lower churn probability.
    - **Very Loyal (60+ months)**: Highly unlikely to churn unless service degrades significantly.
    """
    def __init__(self, bins=[0, 12, 24, 60, np.inf], labels=['0-12', '12-24', '24-60', '60+']):
        self.bins = bins
        self.labels = labels

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if 'tenure' in X.columns:
            logger.info("Binning 'tenure' into cohorts.")
            X['tenure_group'] = pd.cut(X['tenure'], bins=self.bins, labels=self.labels, right=True)
            # Convert to string to treat as categorical for OHE later
            X['tenure_group'] = X['tenure_group'].astype(str)
        return X

class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Applies Log transformation to skewed numerical features like TotalCharges.
    
    Business Logic:
    - Financial features (Charges) often follow a power law distribution.
    - Log transform normalizes this distribution, helping linear models (Baseline) 
      and reducing the impact of extreme whales (high spenders) on distance-based metrics.
    """
    def __init__(self, columns=['TotalCharges']):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            if col in X.columns:
                logger.info(f"Applying log1p transformation to {col}")
                # Using log1p to handle 0s gracefully
                X[f'{col}_log'] = np.log1p(X[col])
        return X

class InteractionFeatures(BaseEstimator, TransformerMixin):
    """
    Creates interaction features to capture usage intensity.
    
    Business Logic:
    - **AvgChargesPerMonth (TotalCharges / tenure)**:
      - Validates if 'MonthlyCharges' matches the long-term average.
      - Discrepancies might indicate recent price hikes (churn risk) or discounts (retention).
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if 'TotalCharges' in X.columns and 'tenure' in X.columns:
            logger.info("Creating interaction feature: AvgCharges (Total / Tenure)")
            # Avoid division by zero
            X['calculated_AvgCharges'] = X['TotalCharges'] / (X['tenure'].replace(0, 1))
        return X

class DropColumns(BaseEstimator, TransformerMixin):
    """
    Drops unnecessary columns (e.g., ID, or original cols after transformation).
    """
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.drop(columns=[c for c in self.columns if c in X.columns], errors='ignore')
