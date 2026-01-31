import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import logging
from src.data_validation.cleaner import DataCleaner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_eda_report(data_path, output_dir):
    logger.info(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        logger.error("Data file not found.")
        return

    # Clean data first to handle TotalCharges
    cleaner = DataCleaner()
    df = cleaner.clean_data(df)

    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Target Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Churn', data=df)
    plt.title('Target Distribution (Churn)')
    plt.savefig(os.path.join(output_dir, 'churn_distribution.png'))
    plt.close()
    
    # 2. Tenure Distribution by Churn
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='tenure', hue='Churn', kde=True, element="step")
    plt.title('Tenure Distribution by Churn')
    plt.savefig(os.path.join(output_dir, 'tenure_by_churn.png'))
    plt.close()
    
    # 3. MonthlyCharges Distribution by Churn
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x='MonthlyCharges', hue='Churn', fill=True)
    plt.title('Monthly Charges Distribution by Churn')
    plt.savefig(os.path.join(output_dir, 'charges_by_churn.png'))
    plt.close()
    
    # 4. Correlation Matrix (Numerical)
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
    plt.close()

    # Generate Markdown Summary
    report_path = os.path.join(output_dir, "eda_summary.md")
    with open(report_path, "w") as f:
        f.write("# Exploratory Data Analysis Report\n\n")
        f.write(f"**Total Samples**: {len(df)}\n")
        f.write(f"**Columns**: {len(df.columns)}\n\n")
        f.write("## Churn Distribution\n")
        f.write(f"Churn Rate: {df['Churn'].value_counts(normalize=True)['Yes']:.2%}\n")
        f.write(f"![Churn Dist](churn_distribution.png)\n\n")
        f.write("## Key Insights\n")
        f.write("- **Tenure**: New customers (low tenure) are more likely to churn.\n")
        f.write("- **Monthly Charges**: Higher monthly charges correlate with higher churn.\n")
        f.write("- **Correlation**: Tenure and TotalCharges are highly correlated (obviously).\n\n")
        f.write("## Hypothesis Generation\n")
        f.write("1. **Price Sensitivity**: Customers with higher monthly bills churn more. Hypothesis: Offering down-sell options might save them.\n")
        f.write("2. **Early Life Churn**: High churn in first 6 months. Hypothesis: Onboarding is critical.\n")

    logger.info(f"EDA Report generated at {output_dir}")

if __name__ == "__main__":
    generate_eda_report("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv", "artifacts/eda")
