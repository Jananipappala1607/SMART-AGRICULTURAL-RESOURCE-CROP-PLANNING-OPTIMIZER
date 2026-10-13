
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "processed" / "andhra_pradesh_annual_crop_trends.csv"
OUTPUT = ROOT / "results" / "yield_forecast_evaluation.csv"

df = pd.read_csv(INPUT)
df["Crop"] = df["Crop"].astype(str).str.strip().str.casefold()
df = df[df["Crop"].isin(["rice", "maize"])].copy()
df = df.sort_values(["Crop", "Crop_Year"])

results = []

for crop in ["rice", "maize"]:
    part = df[df["Crop"] == crop].sort_values("Crop_Year").copy()

    # Use the previous year's value to predict the current year's value.
    part["previous_year_yield"] = part["yield_per_area_unit"].shift(1)
    part = part.dropna(subset=["previous_year_yield", "yield_per_area_unit"])

    # Preserve time order: earlier records for training, later records for testing.
    split = int(len(part) * 0.70)
    train = part.iloc[:split]
    test = part.iloc[split:]

    X_train = train[["previous_year_yield"]]
    y_train = train["yield_per_area_unit"]
    X_test = test[["previous_year_yield"]]
    y_test = test["yield_per_area_unit"]

    models = {
        "Previous-year baseline": DummyRegressor(strategy="mean"),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=3,
            random_state=42
        ),
    }

    for name, model in models.items():
        if name == "Previous-year baseline":
            # A fair naive baseline: predict the previous year's value.
            predictions = X_test["previous_year_yield"].to_numpy()
        else:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

        results.append({
            "crop": crop.title(),
            "model": name,
            "training_records": len(train),
            "testing_records": len(test),
            "test_start_year": int(test["Crop_Year"].min()),
            "test_end_year": int(test["Crop_Year"].max()),
            "MAE": mean_absolute_error(y_test, predictions),
            "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        })

result_df = pd.DataFrame(results)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
result_df.to_csv(OUTPUT, index=False)

print("TIME-ORDERED YIELD FORECAST EVALUATION")
print("======================================")
print(result_df.round(4).to_string(index=False))
print("\nSaved to:")
print(OUTPUT)
print("\nInterpretation: lower MAE and RMSE indicate smaller prediction errors.")
print("This is an exploratory small-data experiment, not proof of generalization.")
print("Verify the source units before interpreting the error units.")