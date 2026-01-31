import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check
import logging

logger = logging.getLogger(__name__)

class DataValidator:
    def __init__(self):
        self.schema = DataFrameSchema({
            "customerID": Column(str, required=True),
            "gender": Column(str, checks=Check.isin(["Male", "Female"]), nullable=False),
            "SeniorCitizen": Column(int, checks=Check.isin([0, 1]), nullable=False),
            "Partner": Column(str, checks=Check.isin(["Yes", "No"]), nullable=False),
            "Dependents": Column(str, checks=Check.isin(["Yes", "No"]), nullable=False),
            "tenure": Column(int, checks=Check.ge(0), nullable=False),
            "PhoneService": Column(str, checks=Check.isin(["Yes", "No"]), nullable=False),
            "MultipleLines": Column(str, nullable=False), # "No phone service" is also a value
            "InternetService": Column(str, checks=Check.isin(["DSL", "Fiber optic", "No"]), nullable=False),
            "OnlineSecurity": Column(str, nullable=False),
            "OnlineBackup": Column(str, nullable=False),
            "DeviceProtection": Column(str, nullable=False),
            "TechSupport": Column(str, nullable=False),
            "StreamingTV": Column(str, nullable=False),
            "StreamingMovies": Column(str, nullable=False),
            "Contract": Column(str, checks=Check.isin(["Month-to-month", "One year", "Two year"]), nullable=False),
            "PaperlessBilling": Column(str, checks=Check.isin(["Yes", "No"]), nullable=False),
            "PaymentMethod": Column(str, nullable=False),
            "MonthlyCharges": Column(float, checks=Check.ge(0), nullable=False),
            # TotalCharges is float after cleaning, but might be object initially. Validation happens AFTER cleaning?
            # Or before? Usually validator checks "Processed" data schema or "Raw" data schema.
            # Let's assume we validate CLEANED data.
            "TotalCharges": Column(float, checks=Check.ge(0), nullable=True), 
            "Churn": Column(str, checks=Check.isin(["Yes", "No"]), nullable=False),
        })

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validates the dataframe against the schema.
        Returns the dataframe if valid, raises SchemaError otherwise.
        """
        try:
            validated_df = self.schema.validate(df)
            logger.info("Data Validation Passed.")
            return validated_df
        except pa.errors.SchemaError as e:
            logger.error(f"Data Validation Failed: {e}")
            raise e
