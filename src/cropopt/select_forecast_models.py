
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "results" / "rolling_forecast_evaluation.csv"
OUTPUT = ROOT / "results" / "selected_forecast_models.csv"

df = pd.read_csv(INPUT)

df["selected_model"] = df.apply(
    lambda row: (
        "Random Forest"
        if row["random_forest_MAE"] < row["baseline_MAE"]
        else "Previous-year baseline"
    ),
    axis=1,
)

df["MAE_improvement_pct"] = (
    (df["baseline_MAE"] - df["random_forest_MAE"])
    / df["baseline_MAE"]
    * 100
)

df["selection_note"] = df.apply(
    lambda row: (
        "Random Forest reduced rolling MAE"
        if row["selected_model"] == "Random Forest"
        else "Baseline had lower rolling MAE"
    ),
    axis=1,
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print("FORECAST MODEL SELECTION REPORT")
print("================================")
print(
    df[
        [
            "crop",
            "test_years",
            "baseline_MAE",
            "random_forest_MAE",
            "selected_model",
            "MAE_improvement_pct",
            "selection_note",
        ]
    ].round(4).to_string(index=False)
)

print("\nSaved to:")
print(OUTPUT)
print("\nSelection is based on rolling-origin MAE, not on a guarantee")
print("of future accuracy.")