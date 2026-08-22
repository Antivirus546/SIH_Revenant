"""
UrbanHeat AI – Intervention Simulator Demo (Real Ward-1 data)
Shows three different intervention strategies side-by-side.
"""

from intervention_simulater import load_ward1_grid, InterventionSimulator
import pandas as pd

def main():
    print("=" * 65)
    print("UrbanHeat AI – Intervention Simulator Demo (Real Ward-1)")
    print("=" * 65)

    # Load real data
    grid = load_ward1_grid("ward1_processed.csv")
    print(f"\nLoaded real grid → {len(grid.cells)} cells\n")

    sim = InterventionSimulator(grid)

    plans = {
        "Cool-roof heavy": {
            "cool_roofs": 0.85,
            "green_cover": 0.15,
            "albedo_boost": 0.20,
        },
        "Green + Cool balanced": {
            "cool_roofs": 0.55,
            "green_cover": 0.60,
            "albedo_boost": 0.25,
        },
        "Max cooling (aggressive)": {
            "cool_roofs": 0.75,
            "green_cover": 0.50,
            "albedo_boost": 0.40,
            "water_bodies": 0.15,
        },
    }

    rows = []
    for name, plan in plans.items():
        res = sim.estimate_impact(plan)
        rows.append({
            "Plan": name,
            "Mean cooling": round(res["mean_cooling"], 2),
            "Pop-weighted cooling": round(res["pop_weighted_cooling"], 2),
            "Total cost (₹ Cr)": round(res["total_cost_inr"] / 1e7, 2),
            "Cooling / lakh ₹": round(res["cooling_per_lakh_inr"], 3),
            "Max cell cooling": round(res["max_cooling"], 1),
        })

    df = pd.DataFrame(rows)
    print("Comparison of three intervention strategies:\n")
    print(df.to_string(index=False))
    print("\n(Higher cooling + higher Cooling/lakh ₹ is better)")
    print("This table is exactly the kind of output the optimizer and dashboard will consume.")

if __name__ == "__main__":
    main()