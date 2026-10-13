
from pathlib import Path
import pandas as pd

from farm_config import FARM_CONFIG

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "planning_crops.csv"
PLAN_FILE = PROJECT_ROOT / "results" / "optimized_crop_plan.csv"
OUTPUT_FILE = PROJECT_ROOT / "results" / "plan_comparison.csv"

# Same illustrative parameters used by optimizer.py.
# These are test assumptions, not verified agricultural measurements.
from optimizer import TEST_PARAMETERS


def main():
    all_crops = pd.read_csv(INPUT_FILE)
    eligible = all_crops[
        all_crops["crop"].isin(TEST_PARAMETERS)
    ]["crop"].tolist()

    if not eligible:
        raise ValueError("No eligible crops found.")

    # Baseline: divide the available land equally among eligible crops.
    baseline_area = FARM_CONFIG["land_area_hectares"] / len(eligible)

    baseline_water = sum(
        baseline_area * TEST_PARAMETERS[crop]["water"]
        for crop in eligible
    )
    baseline_cost = sum(
        baseline_area * TEST_PARAMETERS[crop]["cost"]
        for crop in eligible
    )
    baseline_profit = sum(
        baseline_area * TEST_PARAMETERS[crop]["profit"]
        for crop in eligible
    )

    optimized = pd.read_csv(PLAN_FILE)

    optimized_water = optimized["allocated_water_m3"].sum()
    optimized_cost = optimized["allocated_cost_rupees"].sum()
    optimized_profit = optimized["estimated_profit_rupees"].sum()
    optimized_area = optimized["allocated_hectares"].sum()

    comparison = pd.DataFrame([
        {
            "plan": "Equal allocation baseline",
            "land_hectares": FARM_CONFIG["land_area_hectares"],
            "water_m3": baseline_water,
            "cost_rupees": baseline_cost,
            "illustrative_profit_rupees": baseline_profit,
            "within_water_limit": baseline_water <= FARM_CONFIG["available_water_m3"],
            "within_budget_limit": baseline_cost <= FARM_CONFIG["budget_rupees"],
        },
        {
            "plan": "Optimized allocation",
            "land_hectares": optimized_area,
            "water_m3": optimized_water,
            "cost_rupees": optimized_cost,
            "illustrative_profit_rupees": optimized_profit,
            "within_water_limit": optimized_water <= FARM_CONFIG["available_water_m3"],
            "within_budget_limit": optimized_cost <= FARM_CONFIG["budget_rupees"],
        },
    ])

    comparison.to_csv(OUTPUT_FILE, index=False)

    print("\nPLAN COMPARISON (ILLUSTRATIVE VALUES)")
    print("------------------------------------")
    print(comparison.to_string(index=False))
    print(f"\nSaved comparison to: {OUTPUT_FILE}")

    if baseline_water > FARM_CONFIG["available_water_m3"]:
        print("\nNote: the equal-allocation baseline exceeds the water limit.")
    if baseline_cost > FARM_CONFIG["budget_rupees"]:
        print("Note: the equal-allocation baseline exceeds the budget limit.")


if __name__ == "__main__":
    main()