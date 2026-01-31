import os
import shutil
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_structure():
    # Define project root
    project_root = "."
    
    # List of directories to create
    directories = [
        "data/raw",
        "data/processed",
        "data/features",
        "data/splits",
        "src/data_validation",
        "src/data_splitting",
        "src/feature_engineering",
        "src/models",
        "src/training",
        "src/evaluation",
        "src/inference",
        "src/utils",
        "api",
        "frontend",
        "notebooks",
        "tests",
        "configs",
        "artifacts"
    ]

    for directory in directories:
        path = os.path.join(project_root, directory)
        try:
            os.makedirs(path, exist_ok=True)
            logger.info(f"Created directory: {path}")
        except Exception as e:
            logger.error(f"Failed to create directory {path}: {e}")

def move_dataset():
    source_dir = "Dataset"
    filename = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    source_path = os.path.join(source_dir, filename)
    dest_path = os.path.join("data/raw", filename)

    if os.path.exists(source_path):
        try:
            shutil.copy2(source_path, dest_path)
            logger.info(f"Moved dataset to {dest_path}")
        except Exception as e:
            logger.error(f"Failed to move dataset: {e}")
    else:
        logger.warning(f"Dataset not found at {source_path}. Please place '{filename}' in 'data/raw' manually.")

def create_init_files():
    # Create __init__.py in all src subdirectories to make them packages
    src_dirs = [d for d in os.listdir("src") if os.path.isdir(os.path.join("src", d))]
    for d in src_dirs:
        init_path = os.path.join("src", d, "__init__.py")
        with open(init_path, 'w') as f:
            pass
    logger.info("Created __init__.py files in src directories.")

if __name__ == "__main__":
    logger.info("Starting project setup...")
    create_structure()
    move_dataset()
    create_init_files()
    logger.info("Project setup complete.")
