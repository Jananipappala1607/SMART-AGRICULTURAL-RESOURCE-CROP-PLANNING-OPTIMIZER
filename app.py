
from pathlib import Path

import pandas as pd
import streamlit as st
from scipy.optimize import linprog

st.set_page_config(
    page_title="Smart Crop Planning Optimizer",
    page_icon="🌱",
    layout="wide",
)

# These are illustrative test values, not verified agricultural data.
CROPS = {
    "rice": (6000, 45000, 25000),
    "maize": (3500, 30000, 18000),
    "chickpea": (2500, 22000, 16000),
    "kidneybeans": (3000, 25000, 17000),
    "pigeonpeas": (2800, 23000, 15500),
    "mothbeans": (1800, 15000, 9000),
    "mungbean": (2000, 18000, 12000),
    "blackgram": (2200, 19000, 13000),
    "lentil": (2200, 20000, 14000),
    "pomegranate": (4000, 35000, 22000),
    "banana": (7000, 60000, 30000),
    "mango": (4500, 40000, 24000),
    "grapes": (5000, 50000, 28000),
    "watermelon": (4000, 30000, 20000),
    "muskmelon": (3500, 28000, 19000),
    "orange": (4500, 38000, 23000),
    "papaya": (5000, 42000, 26000),
    "coconut": (6000, 45000, 22000),
    "cotton": (5000, 40000, 21000),
    "jute": (4000, 32000, 17000),
    "coffee": (5000, 45000, 22000),
    "apple": (5000, 55000, 25000),
}

st.title("🌱 Smart Agricultural Resource & Crop Planning Optimizer")
st.caption("Hybrid framework for crop planning under land, water, and budget constraints")

st.warning(
    "Demonstration mode: crop water requirements, costs, and profits are "
    "illustrative test inputs. Results are not real-world farm profit predictions."
)

st.subheader("1. Enter farm resources")

c1, c2, c3 = st.columns(3)
with c1:
    land = st.number_input("Available land (hectares)", 1.0, 1000.0, 10.0, 1.0)
with c2:
    water = st.number_input("Available water (m³)", 1000, 10000000, 50000, 1000)
with c3:
    budget = st.number_input("Available budget (₹)", 10000, 1000000000, 500000, 10000)

if st.button("Optimize Crop Allocation", type="primary", use_container_width=True):
    names = list(CROPS)
    water_per_ha = [CROPS[c][0] for c in names]
    cost_per_ha = [CROPS[c][1] for c in names]
    profit_per_ha = [CROPS[c][2] for c in names]

    result = linprog(
        c=[-p for p in profit_per_ha],
        A_ub=[water_per_ha, cost_per_ha],
        b_ub=[water, budget],
        A_eq=[[1.0] * len(names)],
        b_eq=[land],
        bounds=[(0, None)] * len(names),
        method="highs",
    )

    if not result.success:
        st.error(
            "No feasible allocation was found for these inputs. "
            "Try increasing the available water or budget."
        )
    else:
        rows = []
        for i, name in enumerate(names):
            hectares = result.x[i]
            if hectares > 0.0001:
                rows.append({
                    "Crop": name.title(),
                    "Allocated hectares": hectares,
                    "Water used (m³)": hectares * water_per_ha[i],
                    "Cost (₹)": hectares * cost_per_ha[i],
                    "Illustrative profit (₹)": hectares * profit_per_ha[i],
                })

        plan = pd.DataFrame(rows)
        total_water = sum(result.x[i] * water_per_ha[i] for i in range(len(names)))
        total_cost = sum(result.x[i] * cost_per_ha[i] for i in range(len(names)))
        total_profit = -result.fun

        st.subheader("2. Recommended allocation")
        m1, m2, m3 = st.columns(3)
        m1.metric("Land allocated", f"{sum(result.x):,.2f} ha")
        m2.metric("Water used", f"{total_water:,.0f} m³")
        m3.metric("Budget used", f"₹{total_cost:,.0f}")

        st.metric("Illustrative total profit", f"₹{total_profit:,.0f}")
        st.dataframe(plan.round(2), use_container_width=True, hide_index=True)

        st.subheader("3. Resource utilization")
        st.progress(min(total_water / water, 1.0), text=f"Water used: {total_water / water:.1%}")
        st.progress(min(total_cost / budget, 1.0), text=f"Budget used: {total_cost / budget:.1%}")

        st.download_button(
            "Download crop allocation (CSV)",
            data=plan.to_csv(index=False).encode("utf-8"),
            file_name="optimized_crop_plan.csv",
            mime="text/csv",
        )

        st.success("Optimization completed and resource constraints satisfied.")
        st.caption(
            "The optimizer maximizes the supplied illustrative profit values. "
            "Validate all agricultural inputs before practical use."
        )