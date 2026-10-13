
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "data" / "processed" / "andhra_pradesh_annual_crop_trends.csv"
OUTPUT = ROOT / "results" / "historical_crop_stability_summary.csv"

df = pd.read_csv(INPUT)
df["Crop"] = df["Crop"].astype(str).str.strip()

# Keep crops with at least 10 annual observations.
summary = (
    df.groupby("Crop")
    .agg(
        years_available=("Crop_Year", "nunique"),
        first_year=("Crop_Year", "min"),
        last_year=("Crop_Year", "max"),
        mean_production_per_area=("yield_per_area_unit", "mean"),
        std_production_per_area=("yield_per_area_unit", "std"),
    )
    .reset_index()
)

summary = summary[summary["years_available"] >= 10].copy()

summary["coefficient_of_variation_pct"] = (
    summary["std_production_per_area"]
    / summary["mean_production_per_area"].abs()
    * 100
)

summary["interpretation"] = summary["coefficient_of_variation_pct"].apply(
    lambda value: (
        "Lower observed variability"
        if value < 15
        else "Moderate observed variability"
        if value < 30
        else "Higher observed variability"
    )
)

summary = summary.sort_values("coefficient_of_variation_pct")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(OUTPUT, index=False)

print("HISTORICAL CROP STABILITY SUMMARY")
print("=================================")
print(summary.head(15).round(3).to_string(index=False))
print("\nSaved to:")
print(OUTPUT)
print("\nNote: production-per-area units must be verified.")
print("Lower historical variability does not guarantee higher profit.")