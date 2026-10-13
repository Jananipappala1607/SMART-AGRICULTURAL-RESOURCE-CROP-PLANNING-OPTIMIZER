
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "crop_production.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load historical crop-production records.
df = pd.read_csv(INPUT_FILE, low_memory=False)

# Keep Andhra Pradesh records only.
df["State_Name"] = df["State_Name"].astype(str).str.strip()
ap = df[df["State_Name"].str.casefold() == "andhra pradesh"].copy()

if ap.empty:
    print("Andhra Pradesh was not found.")
    print("Available states include:")
    print(sorted(df["State_Name"].unique())[:20])
    raise SystemExit(1)

print(f"Rows for Andhra Pradesh: {len(ap):,}")

# Convert measurement columns to numeric values.
ap["Area"] = pd.to_numeric(ap["Area"], errors="coerce")
ap["Production"] = pd.to_numeric(ap["Production"], errors="coerce")

# Report records that cannot be used for yield calculation.
missing_production = ap["Production"].isna().sum()
invalid_area = (ap["Area"].isna() | (ap["Area"] <= 0)).sum()

print(f"Missing production values: {missing_production:,}")
print(f"Missing or non-positive area values: {invalid_area:,}")

# Remove records where yield cannot be calculated.
clean = ap.dropna(subset=["Area", "Production"]).copy()
clean = clean[(clean["Area"] > 0) & (clean["Production"] >= 0)]

# Yield is production divided by cultivated area.
# Verify the source units before interpreting the numerical result.
clean["yield_per_area_unit"] = clean["Production"] / clean["Area"]

# Save cleaned district/season/year-level records.
detail_file = OUTPUT_DIR / "andhra_pradesh_crop_yield_detail.csv"
clean.to_csv(detail_file, index=False)

# Aggregate across records using total production / total area.
# This is an area-weighted yield estimate, not a simple average.
summary = (
    clean.groupby("Crop", as_index=False)
    .agg(
        total_area=("Area", "sum"),
        total_production=("Production", "sum"),
        record_count=("Crop", "size"),
        first_year=("Crop_Year", "min"),
        last_year=("Crop_Year", "max"),
    )
)

summary["weighted_yield"] = (
    summary["total_production"] / summary["total_area"]
)

summary = summary.sort_values("Crop")

summary_file = OUTPUT_DIR / "andhra_pradesh_crop_yield_summary.csv"
summary.to_csv(summary_file, index=False)

# Also save a recent-period summary, where the dataset permits it.
recent = clean[clean["Crop_Year"].between(2011, 2015)].copy()

recent_summary = (
    recent.groupby("Crop", as_index=False)
    .agg(
        total_area=("Area", "sum"),
        total_production=("Production", "sum"),
        record_count=("Crop", "size"),
    )
)

if not recent_summary.empty:
    recent_summary["weighted_yield"] = (
        recent_summary["total_production"] /
        recent_summary["total_area"]
    )

recent_file = OUTPUT_DIR / "andhra_pradesh_crop_yield_2011_2015.csv"
recent_summary.to_csv(recent_file, index=False)

print("\nTOP 15 CROPS BY TOTAL PRODUCTION")
print(
    summary.sort_values("total_production", ascending=False)
    .head(15)
    .to_string(index=False)
)

print("\nOUTPUT FILES")
print(detail_file)
print(summary_file)
print(recent_file)

print(
    "\nNote: yield values are production divided by area. "
    "Verify source units and crop-specific comparability before "
    "using these values in the optimizer."
)