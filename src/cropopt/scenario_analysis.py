
from pathlib import Path
import pandas as pd
from scipy.optimize import linprog

from farm_config import FARM_CONFIG
from optimizer import TEST_PARAMETERS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_FILE = PROJECT_ROOT / "results" / "scenario_analysis.csv"


def main():
    crops = list(TEST_PARAMETERS.keys())

    water = [TEST_PARAMETERS[c]["water"] for c in crops]
    cost = [TEST_PARAMETERS[c]["cost"] for c in crops]
    profit = [TEST_PARAMETERS[c]["profit"] for c in crops]

    results = []

    # Test three water limits and three budget limits.
    water_limits = [30000, 40000, 50000]
    budget_limits = [300000, 400000, 500000]

    for water_limit in water_limits:
        for budget_limit in budget_limits:
            result = linprog(
                c=[-p for p in profit],
                A_ub=[water, cost],
                b_ub=[water_limit, budget_limit],
                A_eq=[[1.0] * len(crops)],
                b_eq=[FARM_CONFIG["land_area_hectares"]],
                bounds=[(0, None)] * len(crops),
                method="highs",
            )

            if result.success:
                allocation = {
                    crops[i]: result.x[i]
                    for i in range(len(crops))
                }

                results.append({
                    "water_limit_m3": water_limit,
                    "budget_limit_rupees": budget_limit,
                    "status": "Feasible",
                    "profit_rupees": -result.fun,
                    "water_used_m3": sum(
                        allocation[c] * TEST_PARAMETERS[c]["water"]
                        for c in crops
                    ),
                    "budget_used_rupees": sum(
                        allocation[c] * TEST_PARAMETERS[c]["cost"]
                        for c in crops
                    ),
                    "selected_crops": ", ".join(
                        c for c in crops if allocation[c] > 0.0001
                    ),
                })
            else:
                results.append({
                    "water_limit_m3": water_limit,
                    "budget_limit_rupees": budget_limit,
                    "status": "Infeasible",
                    "profit_rupees": None,
                    "water_used_m3": None,
                    "budget_used_rupees": None,
                    "selected_crops": "",
                })

    df = pd.DataFrame(results)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nRESOURCE SCENARIO ANALYSIS")
    print("==========================")
    print(df.to_string(index=False))
    print(f"\nSaved results to: {OUTPUT_FILE}")
    print("\nNote: all crop parameters are illustrative test values.")


if __name__ == "__main__":
    main()