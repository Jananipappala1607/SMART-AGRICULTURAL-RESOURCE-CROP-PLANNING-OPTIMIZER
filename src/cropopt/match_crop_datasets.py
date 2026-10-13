
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

PLANNING_FILE = ROOT / "data" / "planning_crops.csv"
HISTORY_FILE = ROOT / "data" / "processed" / "andhra_pradesh_annual_crop_trends.csv"
OUTPUT_DIR = ROOT / "data" / "processed"

planning = pd.read_csv(PLANNING_FILE)
history = pd.read_csv(HISTORY_FILE)

# Normalize names for matching without changing original crop names
planning["crop_key"] = (
    planning["crop"].astype(str).str.strip().str.casefold()
)
history["crop_key"] = (
    history["Crop"].astype(str).str.strip().str.casefold()
)

historical_crops = set(history["crop_key"].unique())

planning["has_historical_match"] = planning["crop_key"].isin(
    historical_crops
)

matches = planning[
    ["crop", "has_historical_match"]
].copy()

# Summarize historical data for matching crops
summary = (
    history[history["crop_key"].isin(set(planning["crop_key"]))]
    .groupby(["Crop", "crop_key"], as_index=False)
    .agg(
        years_available=("Crop_Year", "nunique"),
        total_area=("total_area", "sum"),
        total_production=("total_production", "sum"),
    )
)

summary["historical_weighted_yield"] = (
    summary["total_production"] / summary["total_area"]
)

matches_file = OUTPUT_DIR / "crop_dataset_match_report.csv"
summary_file = OUTPUT_DIR / "matched_crop_historical_summary.csv"

matches.to_csv(matches_file, index=False)
summary.to_csv(summary_file, index=False)

print("CROP DATASET MATCHING REPORT")
print("============================")
print("Planning crops:", len(planning))
print("Planning crops with historical matches:",
      int(matches["has_historical_match"].sum()))

print("\nPlanning crop matches:")
print(matches.to_string(index=False))

print("\nHistorical summaries for matching crops:")
if summary.empty:
    print("No exact crop-name matches found.")
else:
    print(summary.to_string(index=False))

print("\nSaved reports:")
print(matches_file)
print(summary_file)