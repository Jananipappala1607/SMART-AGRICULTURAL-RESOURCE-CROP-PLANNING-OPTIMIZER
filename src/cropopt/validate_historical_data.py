
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "data" / "processed" / "matched_crop_historical_summary.csv"
OUTPUT = ROOT / "results" / "historical_data_quality_report.csv"

df = pd.read_csv(INPUT)

def assess(row):
    issues = []

    years = row["years_available"]
    crop = str(row["Crop"]).strip().casefold()
    production = row["total_production"]
    area = row["total_area"]

    if years < 5:
        issues.append("Very limited historical coverage")
    elif years < 10:
        issues.append("Limited historical coverage")

    if area <= 0 or production <= 0:
        issues.append("Check area or production values")

    if crop == "coconut":
        issues.append("Verify production units before comparison")

    if crop in ["grapes", "mango", "orange", "papaya"]:
        issues.append("Short time series; interpret cautiously")

    return "; ".join(issues) if issues else "Suitable for initial historical analysis; verify units"

df["data_quality_note"] = df.apply(assess, axis=1)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print("HISTORICAL DATA QUALITY REPORT")
print("==============================")
print(df[["Crop", "years_available", "historical_weighted_yield",
          "data_quality_note"]].to_string(index=False))
print("\nSaved report to:")
print(OUTPUT)