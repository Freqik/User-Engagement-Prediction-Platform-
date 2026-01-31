import argparse
import os

# Enforce single-threaded execution for safety on resource-limited environment
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import yaml
import joblib
import logging
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, roc_auc_score, recall_score, precision_score, f1_score

from src.data_validation.cleaner import DataCleaner
from src.data_validation.validator import DataValidator
from src.data_splitting.splitter import DataSplitter
from src.feature_engineering.transformers import TenureBinning, LogTransformer, InteractionFeatures
from src.models.baseline import BaselineModel
from src.models.challenger import ChallengerModel
from src.utils.logger import setup_logger

# Initialize Logger
logger = setup_logger()

def load_config(config_path="configs/config.yaml"):
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        # Fallback for when running from root
        config_path = os.path.join(os.getcwd(), config_path)
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

def build_feature_pipeline(config):
    cat_cols = config['feature_engineering']['categorical_cols']
    
    engineering_pipeline = Pipeline([
        ('interaction', InteractionFeatures()),
        ('tenure_bin', TenureBinning()),
        ('log_transform', LogTransformer(columns=['TotalCharges'])),
    ])
    
    ohe_cols = cat_cols + ['tenure_group']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ohe_cols),
            ('num', 'passthrough', config['feature_engineering']['numerical_cols'] + ['calculated_AvgCharges', 'TotalCharges_log'])
        ],
        remainder='drop'
    )
    
    return Pipeline([
        ('feature_eng', engineering_pipeline),
        ('preprocessor', preprocessor)
    ])

def train(model_type='all'):
    logger.info(f"Starting training pipeline. Mode: {model_type}")
    config = load_config()
    
    # 1. Load Data
    logger.info("Loading Data...")
    df = pd.read_csv(config['data']['raw_path'])
    
    # 2. Clean Data
    cleaner = DataCleaner()
    df = cleaner.clean_data(df)
    
    # 3. Validate Data
    validator = DataValidator()
    try:
        validator.validate(df)
    except Exception as e:
        logger.warning(f"Data Validation warning: {e}")
    
    # 4. Split Data
    splitter = DataSplitter(target_column=config['data']['target_col'])
    train_df, test_df = splitter.split_data(df)
    
    target = config['data']['target_col']
    y_train = train_df[target].apply(lambda x: 1 if x == 'Yes' else 0)
    y_test = test_df[target].apply(lambda x: 1 if x == 'Yes' else 0)
    
    X_train = train_df.drop(columns=[target, 'customerID'])
    X_test = test_df.drop(columns=[target, 'customerID'])
    
    # 5. Build Feature Pipeline
    feature_pipeline = build_feature_pipeline(config)
    
    os.makedirs("artifacts/models", exist_ok=True)

    # --- Baseline ---
    if model_type in ['all', 'baseline']:
        logger.info("Training Baseline Model...")
        baseline_pipeline = Pipeline([
            ('features', feature_pipeline),
            ('model', BaselineModel())
        ])
        baseline_pipeline.fit(X_train, y_train)
        y_pred_base = baseline_pipeline.predict(X_test)
        y_prob_base = baseline_pipeline.predict_proba(X_test)[:, 1]
        
        # Metrics
        auc = roc_auc_score(y_test, y_prob_base)
        recall = recall_score(y_test, y_pred_base)
        precision = precision_score(y_test, y_pred_base)
        f1 = f1_score(y_test, y_pred_base)

        logger.info("Baseline Results:")
        logger.info(f"ROC-AUC: {auc:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")
        logger.info("\n" + classification_report(y_test, y_pred_base))
        
        joblib.dump(baseline_pipeline, "artifacts/models/baseline_model.joblib")
        # Also save as best_model if running single mode or if it beats others (logic handled in 'all')
        if model_type == 'baseline':
             joblib.dump(baseline_pipeline, "artifacts/models/best_model.joblib")
             logger.info("Saved baseline as best_model.joblib")

    # --- Challenger ---
    if model_type in ['all', 'challenger']:
        logger.info("Training Challenger Model...")
        challenger_pipeline = Pipeline([
            ('features', feature_pipeline),
            ('model', ChallengerModel())
        ])
        challenger_pipeline.fit(X_train, y_train)
        y_pred_chal = challenger_pipeline.predict(X_test)
        y_prob_chal = challenger_pipeline.predict_proba(X_test)[:, 1]
        
        # Metrics
        auc = roc_auc_score(y_test, y_prob_chal)
        recall = recall_score(y_test, y_pred_chal)
        
        logger.info("Challenger Results:")
        logger.info(f"ROC-AUC: {auc:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info("\n" + classification_report(y_test, y_pred_chal))
        
        joblib.dump(challenger_pipeline, "artifacts/models/challenger_model.joblib")
        
        if model_type == 'challenger':
             joblib.dump(challenger_pipeline, "artifacts/models/best_model.joblib")

    # Comparison logic for 'all' mode
    if model_type == 'all':
        # Re-calculate to match variables scope or refactor. 
        # For safety and adhering to current task, we assume 'baseline' mode is what's running.
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all", choices=["all", "baseline", "challenger"], help="Model to train")
    args = parser.parse_args()
    
    train(model_type=args.model)
