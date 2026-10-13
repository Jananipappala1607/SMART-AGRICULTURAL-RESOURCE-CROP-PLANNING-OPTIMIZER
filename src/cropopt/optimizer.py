
from pathlib import Path
import pandas as pd
from scipy.optimize import linprog

from farm_config import FARM_CONFIG

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "planning_crops.csv"
OUTPUT_FILE = PROJECT_ROOT / "results" / "optimized_crop_plan.csv"

# ILLUSTRATIVE TEST VALUES ONLY.
# Replace these with sourced agricultural data before research evaluation.
TEST_PARAMETERS = {
    "rice":      {"water": 6000, "cost": 45000, "profit": 25000},
    "maize":     {"water": 3500, "cost": 30000, "profit": 18000},
    "chickpea":  {"water": 2500, "cost": 22000, "profit": 16000},
    "kidneybeans":{"water": 3000, "cost": 25000, "profit": 17000},
    "pigeonpeas":{"water": 2800, "cost": 23000, "profit": 15500},
    "mothbeans": {"water": 1800, "cost": 15000, "profit": 9000},
    "mungbean":  {"water": 2000, "cost": 18000, "profit": 12000},
    "blackgram": {"water": 2200, "cost": 19000, "profit": 13000},
    "lentil":    {"water": 2200, "cost": 20000, "profit": 14000},
    "pomegranate":{"water": 4000, "cost": 35000, "profit": 22000},
    "banana":    {"water": 7000, "cost": 60000, "profit": 30000},
    "mango":     {"water": 4500, "cost": 40000, "profit": 24000},
    "grapes":    {"water": 5000, "cost": 50000, "profit": 28000},
    "watermelon":{"water": 4000, "cost": 30000, "profit": 20000},
    "muskmelon":{"water": 3500, "cost": 28000, "profit": 19000},
    "orange":    {"water": 4500, "cost": 38000, "profit": 23000},
    "papaya":    {"water": 5000, "cost": 42000, "profit": 26000},
    "coconut":   {"water": 6000, "cost": 45000, "profit": 22000},
    "cotton":    {"water": 5000, "cost": 40000, "profit": 21000},
    "jute":      {"water": 4000, "cost": 32000, "profit": 17000},
    "coffee":    {"water": 5000, "cost": 45000, "profit": 22000},
    "apple":     {"water": 5000, "cost": 55000, "profit": 25000},
}

def main():
    crops = pd.read_csv(INPUT_FILE)

    # Only use crops with explicitly supplied test parameters.
    crops = crops[crops["crop"].isin(TEST_PARAMETERS)].copy()

    if crops.empty:
        raise ValueError("No matching crops found in planning_crops.csv.")

    crops["water"] = crops["crop"].map(
        lambda crop: TEST_PARAMETERS[crop]["water"]
    )
    crops["cost"] = crops["crop"].map(
        lambda crop: TEST_PARAMETERS[crop]["cost"]
    )
    crops["profit"] = crops["crop"].map(
        lambda crop: TEST_PARAMETERS[crop]["profit"]
    )

    # Decision variable: hectares allocated to each crop.
    # Maximize total illustrative profit by minimizing negative profit.
    objective = -crops["profit"].to_numpy(dtype=float)

    # Total water and total budget must remain within limits.
    constraints = [
        crops["water"].to_numpy(dtype=float),
        crops["cost"].to_numpy(dtype=float),
    ]

    limits = [
        FARM_CONFIG["available_water_m3"],
        FARM_CONFIG["budget_rupees"],
    ]

    result = linprog(
        c=objective,
        A_ub=constraints,
        b_ub=limits,
        A_eq=[ [1.0] * len(crops) ],
        b_eq=[FARM_CONFIG["land_area_hectares"]],
        bounds=[(0, None)] * len(crops),
        method="highs",
    )

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    crops["allocated_hectares"] = result.x
    crops = crops[crops["allocated_hectares"] > 0.0001].copy()

    crops["allocated_water_m3"] = (
        crops["allocated_hectares"] * crops["water"]
    )
    crops["allocated_cost_rupees"] = (
        crops["allocated_hectares"] * crops["cost"]
    )
    crops["estimated_profit_rupees"] = (
        crops["allocated_hectares"] * crops["profit"]
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    crops.to_csv(OUTPUT_FILE, index=False)

    print("\nOPTIMIZATION SUCCESSFUL (ILLUSTRATIVE TEST)")
    print("-------------------------------------------")
    print(crops[[
        "crop",
        "allocated_hectares",
        "allocated_water_m3",
        "allocated_cost_rupees",
        "estimated_profit_rupees",
    ]].to_string(index=False))

    print("\nTOTALS")
    print(f"Land allocated: {result.x.sum():.2f} hectares")
    print(f"Water used: {sum(result.x * constraints[0]):,.0f} m³")
    print(f"Budget used: ₹{sum(result.x * constraints[1]):,.0f}")
    print(f"Illustrative profit: ₹{-result.fun:,.0f}")
    print(f"\nSaved plan to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()