"""
Quick demo you can run or show in today's meeting.
Shows three different intervention plans side-by-side.
"""

from intervention_simulater import create_dummy_bengaluru_grid, InterventionSimulator
import pandas as pd

def main():
    print("=" * 60)
    print("UrbanHeat AI – Intervention Simulator Demo")
    print("=" * 60)

    grid = create_dummy_bengaluru_grid(n_cells=30, seed=7)
    sim = InterventionSimulator(grid)

    plans = {
        "Green-heavy": {
            "green_cover": 0.8,
            "cool_roofs": 0.2,
            "albedo_boost": 0.1,
        },
        "Cool-roof focus": {
            "green_cover": 0.2,
            "cool_roofs": 0.9,
            "albedo_boost": 0.3,
        },
        "Balanced + water": {
            "green_cover": 0.5,
            "cool_roofs": 0.4,
            "water_bodies": 0.25,
            "albedo_boost": 0.2,
        },
    }

    rows = []
    for name, plan in plans.items():
        res = sim.estimate_impact(plan)
        rows.append({
            "Plan": name,
            "Mean cooling": round(res["mean_cooling"], 4),
            "Pop-weighted cooling": round(res["pop_weighted_cooling"], 4),
            "Total cost (₹ Cr)": round(res["total_cost_inr"] / 1e7, 2),
            "Cooling / lakh ₹": round(res["cooling_per_lakh_inr"], 4),
        })

    df = pd.DataFrame(rows)
    print("\nComparison of three intervention strategies:\n")
    print(df.to_string(index=False))
    print("\n(Higher cooling + higher cooling-per-lakh is better)")
    print("\nThis table is exactly the kind of output the optimizer and dashboard will consume.")

if __name__ == "__main__":
    main()
