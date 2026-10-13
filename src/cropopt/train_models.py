
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "Crop_dataset.csv"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

FEATURES = [
    "N", "P", "K", "temperature",
    "humidity", "ph", "rainfall"
]
TARGET = "label"


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required_columns = FEATURES + [TARGET]
    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    if df[required_columns].isnull().any().any():
        raise ValueError("Dataset contains missing values.")

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    models = {
        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }

    results = []
    fitted_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        macro_f1 = f1_score(
            y_test, predictions, average="macro"
        )

        results.append({
            "model": name,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
        })

        fitted_models[name] = model

        print(f"\n{'=' * 55}")
        print(name)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro-F1: {macro_f1:.4f}")
        print("\nClassification report:")
        print(classification_report(y_test, predictions))

    results_df = pd.DataFrame(results).sort_values(
        by="macro_f1", ascending=False
    )

    results_df.to_csv(
        RESULTS_DIR / "baseline_model_comparison.csv",
        index=False,
    )

    best_name = results_df.iloc[0]["model"]
    best_model = fitted_models[best_name]

    joblib.dump(
        best_model,
        MODEL_DIR / "best_crop_model.joblib",
    )

    metadata = {
        "model_name": best_name,
        "features": FEATURES,
        "target": TARGET,
        "random_state": 42,
        "test_size": 0.20,
        "note": (
            "Baseline benchmark only; evaluate on additional "
            "data before claiming real-world generalization."
        ),
    }

    with open(
        MODEL_DIR / "model_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=4)

    print("\n\nBASELINE COMPARISON")
    print(results_df.to_string(index=False))
    print(f"\nBest baseline model: {best_name}")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"Model saved to: {MODEL_DIR}")


if __name__ == "__main__":
    main()