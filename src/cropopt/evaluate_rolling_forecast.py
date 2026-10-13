
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "processed" / "andhra_pradesh_annual_crop_trends.csv"
OUTPUT = ROOT / "results" / "rolling_forecast_evaluation.csv"

df = pd.read_csv(INPUT)
df["Crop"] = df["Crop"].astype(str).str.strip().str.casefold()
df = df[df["Crop"].isin(["rice", "maize"])].copy()

all_results = []

for crop in ["rice", "maize"]:
    part = df[df["Crop"] == crop].sort_values("Crop_Year").copy()
    part = part.dropna(subset=["yield_per_area_unit"])

    years = part["Crop_Year"].to_numpy()
    values = part["yield_per_area_unit"].to_numpy()

    # Begin testing after at least 8 previous annual observations.
    predictions = []

    for i in range(8, len(values)):
        train_x = values[:i - 1].reshape(-1, 1)
        train_y = values[1:i]
        test_x = np.array([[values[i - 1]]])

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=3,
            random_state=42
        )
        model.fit(train_x, train_y)

        predictions.append({
            "crop": crop.title(),
            "year": int(years[i]),
            "actual": float(values[i]),
            "baseline_prediction": float(values[i - 1]),
            "random_forest_prediction": float(model.predict(test_x)[0]),
        })

    all_results.extend(predictions)

pred = pd.DataFrame(all_results)

summary = pred.groupby("crop").apply(
    lambda g: pd.Series({
        "test_years": len(g),
        "test_start_year": g["year"].min(),
        "test_end_year": g["year"].max(),
        "baseline_MAE": mean_absolute_error(
            g["actual"], g["baseline_prediction"]
        ),
        "random_forest_MAE": mean_absolute_error(
            g["actual"], g["random_forest_prediction"]
        ),
        "baseline_RMSE": np.sqrt(mean_squared_error(
            g["actual"], g["baseline_prediction"]
        )),
        "random_forest_RMSE": np.sqrt(mean_squared_error(
            g["actual"], g["random_forest_prediction"]
        )),
    }),
    include_groups=False
).reset_index()

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(OUTPUT, index=False)

print("ROLLING-ORIGIN FORECAST EVALUATION")
print("==================================")
print(summary.round(4).to_string(index=False))
print("\nSaved to:")
print(OUTPUT)
print("\nLower MAE and RMSE are better.")
print("The evaluation uses historical data only; units still need verification.")