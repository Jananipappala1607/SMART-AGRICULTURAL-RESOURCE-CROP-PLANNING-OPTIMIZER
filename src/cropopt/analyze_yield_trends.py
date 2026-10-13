
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "raw" / "crop_production.csv"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT, low_memory=False)
df["State_Name"] = df["State_Name"].astype(str).str.strip()
df["Area"] = pd.to_numeric(df["Area"], errors="coerce")
df["Production"] = pd.to_numeric(df["Production"], errors="coerce")

ap = df[
    df["State_Name"].str.casefold() == "andhra pradesh"
].copy()

# Remove missing values and invalid areas
ap = ap.dropna(subset=["Crop_Year", "Crop", "Area", "Production"])
ap = ap[(ap["Area"] > 0) & (ap["Production"] >= 0)]

# Aggregate across districts for each crop and year
annual = (
    ap.groupby(["Crop_Year", "Crop"], as_index=False)
    .agg(
        total_area=("Area", "sum"),
        total_production=("Production", "sum"),
        district_records=("District_Name", "size"),
    )
)

annual["yield_per_area_unit"] = (
    annual["total_production"] / annual["total_area"]
)

annual_path = OUT / "andhra_pradesh_annual_crop_trends.csv"
annual.to_csv(annual_path, index=False)

# Analyze crops with records in at least 10 different years
coverage = (
    annual.groupby("Crop")["Crop_Year"]
    .nunique()
    .rename("years_available")
)

eligible = coverage[coverage >= 10].index
stable = annual[annual["Crop"].isin(eligible)].copy()

# Calculate yield variability for each crop
variability = (
    stable.groupby("Crop")["yield_per_area_unit"]
    .agg(
        mean_yield="mean",
        yield_std="std",
        years_available="count",
    )
    .reset_index()
)

variability["yield_cv_percent"] = (
    variability["yield_std"] / variability["mean_yield"].replace(0, float("nan"))
) * 100

variability = variability.sort_values("yield_cv_percent")

variability_path = OUT / "andhra_pradesh_yield_variability.csv"
variability.to_csv(variability_path, index=False)

print("ANDHRA PRADESH HISTORICAL YIELD ANALYSIS")
print("========================================")
print("Valid annual crop records:", len(annual))
print("Distinct crops:", annual["Crop"].nunique())
print("Crops with records in at least 10 years:", len(eligible))

print("\nCrops with the lowest observed yield variability:")
print(variability.head(10).to_string(index=False))

print("\nCrops with the highest observed yield variability:")
print(variability.tail(10).sort_values(
    "yield_cv_percent", ascending=False
).to_string(index=False))

print("\nSaved files:")
print(annual_path)
print(variability_path)

print(
    "\nNote: These are historical production-to-area ratios. "
    "Verify crop-specific units before comparing different crops."
)