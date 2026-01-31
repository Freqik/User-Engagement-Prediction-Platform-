import pandas as pd
from sklearn.model_selection import train_test_split
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class DataSplitter:
    def __init__(self, target_column: str, test_size: float = 0.2, random_state: int = 42):
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the data into train and test sets using Stratified Sampling.
        This preserves the percentage of samples for each class.
        """
        logger.info(f"Splitting data with test_size={self.test_size} and random_state={self.random_state}")
        
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataframe.")
            
        train_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            stratify=df[self.target_column],
            random_state=self.random_state
        )
        
        logger.info(f"Train/Test split complete. Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        
        # Log distribution
        train_dist = train_df[self.target_column].value_counts(normalize=True).to_dict()
        test_dist = test_df[self.target_column].value_counts(normalize=True).to_dict()
        logger.info(f"Train Class Dist: {train_dist}")
        logger.info(f"Test Class Dist: {test_dist}")
        
        return train_df, test_df
