
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "raw" / "crop_production.csv"

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

# Read the dataset
df = pd.read_csv(DATA_FILE, low_memory=False)

print("\n========== DATASET OVERVIEW ==========")
print(f"File: {DATA_FILE.name}")
print(f"Rows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")
print("\nColumn names:")
print(df.columns.tolist())

print("\n========== FIRST FIVE ROWS ==========")
print(df.head().to_string())

print("\n========== DATA TYPES ==========")
print(df.dtypes.to_string())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum().to_string())

print("\n========== NUMERIC SUMMARY ==========")
print(df.describe(include="number").to_string())

print("\n========== INSPECTION COMPLETE ==========")