
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "processed" / "andhra_pradesh_annual_crop_trends.csv"
OUTPUT = ROOT / "results"
OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)
df["Crop"] = df["Crop"].str.strip()

selected = df[df["Crop"].str.casefold().isin(["rice", "maize"])]

plt.figure(figsize=(10, 5))

for crop in ["Rice", "Maize"]:
    part = selected[selected["Crop"].str.casefold() == crop.casefold()]
    part = part.sort_values("Crop_Year")
    plt.plot(
        part["Crop_Year"],
        part["yield_per_area_unit"],
        marker="o",
        label=crop,
    )

plt.title("Historical Crop Yield Trends in Andhra Pradesh")
plt.xlabel("Year")
plt.ylabel("Production per Area Unit")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

chart_path = OUTPUT / "historical_rice_maize_yield_trends.png"
plt.savefig(chart_path, dpi=300)
plt.close()

print("Chart saved to:")
print(chart_path)
print("Note: verify source units before interpreting yield as tonnes per hectare.")