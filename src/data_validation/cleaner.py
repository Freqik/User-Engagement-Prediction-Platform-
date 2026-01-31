import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DataCleaner:
    def __init__(self):
        pass

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the input DataFrame.
        - Coerces TotalCharges to numeric (errors='coerce').
        - Fills missing TotalCharges (which result from empty strings in this dataset) with 0 or mean.
          (Logic: If tenure is 0, TotalCharges is usually ' '. Imputing with 0 makes sense).
        - Standardizes categorical values if needed (e.g. 'No phone service' -> 'No').
        """
        df = df.copy()
        
        # TotalCharges: Convert to numeric, handle errors
        # The dataset has " " for some rows where tenure is 0.
        original_rows = len(df)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        nan_count = df['TotalCharges'].isna().sum()
        if nan_count > 0:
            logger.info(f"Found {nan_count} non-numeric values in TotalCharges. Filling with 0 (Assuming new customers).")
            df['TotalCharges'] = df['TotalCharges'].fillna(0.0)

        # Basic Check: Drop duplicates if any
        df = df.drop_duplicates()
        if len(df) < original_rows:
            logger.info(f"Dropped {original_rows - len(df)} duplicate rows.")

        return df
