
# Farm resource limits for initial testing.
# These are illustrative values, not real farm data.

FARM_CONFIG = {
    "land_area_hectares": 10,
    "available_water_m3": 50000,
    "budget_rupees": 500000,
}

if __name__ == "__main__":
    print("Farm Resource Configuration")
    print("----------------------------")
    print(f"Available land: {FARM_CONFIG['land_area_hectares']} hectares")
    print(f"Available water: {FARM_CONFIG['available_water_m3']} m³")
    print(f"Available budget: ₹{FARM_CONFIG['budget_rupees']:,}")