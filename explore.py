import pandas as pd

df = pd.read_csv("data/Crop_dataset.csv")

print("Shape:", df.shape)
print("\nColumns and types:")
print(df.dtypes)
print("\nFirst rows:")
print(df.head())
print("\nMissing values:")
print(df.isna().sum())
print("\nUnique values per column:")
print(df.nunique())