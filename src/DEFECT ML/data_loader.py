# ============================================================
# data_loader.py — Dataset loading, exploration, and Kaggle setup
# ============================================================

import os
import json
import pandas as pd

from config import DATASET_FILE, TARGET_COLUMN, KAGGLE_DATASET


def setup_kaggle_credentials() -> None:
    """
    Write Kaggle credentials from Colab Secrets to ~/.config/kaggle/kaggle.json.
    Only needed when running in Google Colab.
    """
    try:
        from google.colab import userdata

        kaggle_username = userdata.get("KAGGLE_USERNAME")
        kaggle_key = userdata.get("kagglekey")

        os.makedirs("/root/.config/kaggle", exist_ok=True)
        with open("/root/.config/kaggle/kaggle.json", "w") as f:
            json.dump({"username": kaggle_username, "key": kaggle_key}, f)
        os.chmod("/root/.config/kaggle/kaggle.json", 0o600)
        print(f"✓ kaggle.json written for: {kaggle_username}")
    except ImportError:
        print("Not running in Colab — skipping Kaggle credential setup.")


def download_dataset() -> None:
    """Download and unzip the dataset from Kaggle (requires kaggle CLI)."""
    os.system("pip install kaggle==1.6.17 -q")
    os.system(f"kaggle datasets download -d {KAGGLE_DATASET} --unzip -q")
    print("✓ Dataset downloaded.")


def load_data(filepath: str = DATASET_FILE) -> pd.DataFrame:
    """Load the CSV dataset and return a DataFrame."""
    df = pd.read_csv(filepath)
    print(f"✓ Loaded '{filepath}': {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def explore_data(df: pd.DataFrame) -> None:
    """Print a concise overview of the dataset."""
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Shape          : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\nColumns        : {list(df.columns)}")

    print(f"\nTarget distribution ({TARGET_COLUMN}):")
    print(df[TARGET_COLUMN].value_counts())
    print("\nClass balance (%):")
    print(df[TARGET_COLUMN].value_counts(normalize=True).mul(100).round(2))

    print(f"\nMissing values : {df.isnull().sum().sum()} total")
    print(f"\nData types:\n{df.dtypes.value_counts()}")
    print("\nFirst 5 rows:")
    print(df.head().to_string())
