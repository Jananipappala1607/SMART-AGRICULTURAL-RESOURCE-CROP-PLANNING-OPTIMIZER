
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "Crop_dataset.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "planning_crops.csv"

# Load the existing crop recommendation dataset.
df = pd.read_csv(INPUT_FILE)

# Keep one row per crop and calculate average soil/weather conditions.
planning = (
    df.groupby("label", as_index=False)
    .agg(
        avg_nitrogen=("N", "mean"),
        avg_phosphorus=("P", "mean"),
        avg_potassium=("K", "mean"),
        avg_temperature=("temperature", "mean"),
        avg_humidity=("humidity", "mean"),
        avg_ph=("ph", "mean"),
        avg_rainfall=("rainfall", "mean"),
    )
    .rename(columns={"label": "crop"})
)

# Optimization inputs are intentionally left blank until sourced.
planning["yield_tonnes_per_hectare"] = pd.NA
planning["water_m3_per_hectare"] = pd.NA
planning["cost_rupees_per_hectare"] = pd.NA
planning["price_rupees_per_tonne"] = pd.NA

planning.to_csv(OUTPUT_FILE, index=False)

print(f"Planning dataset saved to: {OUTPUT_FILE}")
print(f"Number of crops: {len(planning)}")
print(planning.head())
print("\nResource columns are blank until real values are sourced.")