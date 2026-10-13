
from pathlib import Path

import pandas as pd

# Locate the project directory and dataset
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "Crop_dataset.csv"

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    print("=" * 55)
    print("SMART AGRICULTURAL CROP PLANNING")
    print("=" * 55)

    print(f"\nDataset shape: {df.shape}")
    print("\nColumns:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nCrop distribution:")
    print(df["label"].value_counts().sort_index())

    print("\nStatistical summary:")
    print(df.describe().round(2).to_string())

if __name__ == "__main__":
    main()