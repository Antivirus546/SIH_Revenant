"""
UrbanHeat AI – End-to-End Demo (Real Ward-1 data)
Team Revenant | SIH 2026 | IHSIH031

Demonstrates the complete simulator + optimizer subsystem OFFLINE:

    1. Load the processed Ward-1 grid
    2. Heuristic simulation of example plans
    3. ML simulation of the same plans (auto-fallback if XGBoost absent)
    4. Heuristic vs ML comparison
    5. NSGA-II multi-objective optimization (cooling vs cost)
    6. Greedy budget-allocation baseline (neutral comparison)
    7. Readable Pareto table + comparison metrics
    8. Verification checklist

Run:  python demo_usage.py

NOTE ON SCIENTIFIC STATUS
-------------------------
All cooling numbers are SCENARIO ESTIMATES under ASSUMED coefficients
(see intervention_simulater.py). They are not measured physical
outcomes. The demo is decision-support, not validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from intervention_simulater import (
    InterventionSimulator,
    create_ml_simulator,
    load_ward1_grid,
)
from optimizer import (
    INTERVENTION_NAMES,
    NSGA2Optimizer,
    greedy_baseline,
)

from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT_DIR / "Data" / "ward1_processed.csv"

EXAMPLE_PLANS = {
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


def plan_table(sim: InterventionSimulator, plans: dict) -> pd.DataFrame:
    rows = []
    for name, plan in plans.items():
        res = sim.estimate_impact(plan)
        rows.append({
            "Plan": name,
            "Mean cooling (°C)": round(res["mean_cooling"], 3),
            "Pop-weighted (°C)": round(res["pop_weighted_cooling"], 3),
            "Max cell (°C)": round(res["max_cooling"], 3),
            "Cost (₹ Cr)": round(res["total_cost_inr"] / 1e7, 2),
            "Cooling / lakh ₹": round(res["cooling_per_lakh_inr"], 4),
        })
    return pd.DataFrame(rows)


def main() -> None:
    print("=" * 72)
    print("UrbanHeat AI – Simulator + Optimizer Demo (Ward 1, Bengaluru)")
    print("=" * 72)

    checks: list[tuple[str, bool]] = []

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    grid = load_ward1_grid(CSV_PATH)
    base = np.array([c.baseline_heat_stress for c in grid.cells.values()])
    print("\n[1] Loaded grid: {} cells".format(len(grid.cells)))
    print("    Baseline LST °C  min/mean/max: {:.2f} / {:.2f} / {:.2f}".format(
        base.min(), base.mean(), base.max()))
    checks.append(("baseline temps plausible °C (0–60)",
                   bool(np.all(base > 0) and np.all(base < 60))))

    # ------------------------------------------------------------------
    # 2. Heuristic simulation
    # ------------------------------------------------------------------
    print("\n[2] HEURISTIC simulation (linear scenario model)")
    sim_h = InterventionSimulator(grid)
    df_h = plan_table(sim_h, EXAMPLE_PLANS)
    print(df_h.to_string(index=False))

    res_h = sim_h.estimate_impact(EXAMPLE_PLANS["Green + Cool balanced"])
    required_keys = [
        "plan", "total_cooling", "mean_cooling", "pop_weighted_cooling",
        "total_cost_inr", "cooling_per_lakh_inr", "delta_per_cell",
        "modified_grid",
    ]
    checks.append(("public interface keys preserved",
                   all(k in res_h for k in required_keys)))
    checks.append(("heuristic costs positive", res_h["total_cost_inr"] > 0))
    checks.append(("heuristic deltas finite",
                   bool(np.all(np.isfinite(res_h["delta_per_cell"])))))

    # ------------------------------------------------------------------
    # 3. ML simulation (with graceful fallback)
    # ------------------------------------------------------------------
    print("\n[3] ML simulation (locally retrained model adapter)")
    sim_ml, ml_info = create_ml_simulator(grid, csv_path=CSV_PATH)
    if ml_info["mode"] == "ml":
        src = ml_info["model_info"]["source"]
        print("    ML mode ACTIVE (model source: {})".format(src))
        df_ml = plan_table(sim_ml, EXAMPLE_PLANS)
        print(df_ml.to_string(index=False))
        r = sim_ml.evaluate_plan_metrics(EXAMPLE_PLANS["Cool-roof heavy"])
        checks.append(("ML deltas finite",
                       bool(np.all(np.isfinite(r["delta_per_cell"])))))
        checks.append(("ML predictions plausible °C",
                       bool(np.all(r["predictions"] > 0)
                            and np.all(r["predictions"] < 60))))
    else:
        print("    ML mode UNAVAILABLE -> fell back to heuristic.")
        print("    Reason: {}".format(ml_info["model_info"].get("message", "unknown")))
        checks.append(("graceful fallback when XGBoost missing",
                       ml_info["mode"] == "heuristic"))

    # ------------------------------------------------------------------
    # 4. Heuristic vs ML comparison
    # ------------------------------------------------------------------
    print("\n[4] Heuristic vs ML (same plans)")
    if ml_info["mode"] == "ml":
        cmp_rows = []
        for name, plan in EXAMPLE_PLANS.items():
            h = sim_h.evaluate_plan_metrics(plan)
            m = sim_ml.evaluate_plan_metrics(plan)
            cmp_rows.append({
                "Plan": name,
                "Heuristic pop-cool (°C)": round(h["pop_weighted_cooling"], 3),
                "ML pop-cool (°C)": round(m["pop_weighted_cooling"], 3),
                "Heuristic cost (₹ Cr)": round(h["total_cost_inr"] / 1e7, 2),
                "ML cost (₹ Cr)": round(m["total_cost_inr"] / 1e7, 2),
            })
        print(pd.DataFrame(cmp_rows).to_string(index=False))
        print("    (Differences are expected: the two engines encode different")
        print("     ASSUMED response models. Neither is physically validated.)")
    else:
        print("    Skipped — ML mode unavailable on this machine.")

    # Monotonicity spot-check (both engines)
    for label, s in (("heuristic", sim_h),
                     ("ml", sim_ml if ml_info["mode"] == "ml" else None)):
        if s is None:
            continue
        means = [s.evaluate_plan_metrics({"green_cover": lv})["mean_cooling"]
                 for lv in (0.25, 0.5, 1.0)]
        ok = means[0] < means[1] < means[2]
        checks.append((label + ": mean cooling rises with intensity "
                       "({:.3f}->{:.3f}->{:.3f})".format(*means), ok))

    # ------------------------------------------------------------------
    # 5–6. Optimization: NSGA-II + greedy baseline
    # ------------------------------------------------------------------
    hot = sim_h.select_hottest_cells(0.3)
    print("\n[5] NSGA-II optimization — target: hottest 30% ({} cells), "
          "objectives: max pop-weighted cooling / min cost".format(len(hot)))

    opt = NSGA2Optimizer(
        sim_h, target_cells=hot,
        pop_size=48, n_generations=60, seed=42,
    )
    result = opt.optimize()
    sols = result.pareto_solutions
    print("    {} Pareto solutions from {} evaluations\n".format(
        len(sols), result.n_evaluations))

    pareto_rows = [{
        "Cost (₹ Cr)": round(s["cost"] / 1e7, 2),
        "Pop-cool (°C)": round(s["cooling"], 3),
        "Mean (°C)": round(s["mean_cooling"], 3),
        "Cool/lakh ₹": round(s["cooling_per_lakh_inr"], 4),
        "CR/GC/AB/WB": "/".join("{:.2f}".format(s["plan"][n])
                                for n in INTERVENTION_NAMES),
    } for s in sols]
    print(pd.DataFrame(pareto_rows).to_string(index=False))

    checks.append(("NSGA-II produced a non-empty valid Pareto set", len(sols) > 0))
    checks.append(("all intensities within [0, 1]",
                   all(all(0.0 <= v <= 1.0 for v in s["intensities"])
                       for s in sols)))

    print("\n[6] Greedy baseline (same target cells)")
    g_plan, g_sum = greedy_baseline(sim_h, target_cells=hot)
    print("    Plan : {{{}}}".format(
        ", ".join("{}: {:.2f}".format(k, v) for k, v in g_plan.items())))
    print("    Pop-weighted cooling : {:.3f} °C".format(
        g_sum["pop_weighted_cooling"]))
    print("    Total cost           : ₹{:.2f} Cr".format(
        g_sum["total_cost_inr"] / 1e7))

    # Neutral comparison at nearest cost — no assumption about the winner
    if sols:
        nearest = min(sols, key=lambda s: abs(s["cost"] - g_sum["total_cost_inr"]))
        diff = g_sum["pop_weighted_cooling"] - nearest["cooling"]
        winner = "greedy" if diff > 0 else "NSGA-II"
        print("\n    Comparison at ≈₹{:.2f} Cr: {} reaches more pop-weighted "
              "cooling (Δ = {:.3f} °C). Results decide, not assumptions.".format(
                  nearest["cost"] / 1e7, winner, abs(diff)))

    # Budget-constrained demonstration
    budget = 5.0e7
    opt_b = NSGA2Optimizer(sim_h, target_cells=hot, pop_size=24, n_generations=25,
                           budget_inr=budget, seed=7)
    res_b = opt_b.optimize()
    within = all(s["cost"] <= budget * (1 + 1e-9) for s in res_b.pareto_solutions)
    checks.append(("budget constraint respected (≤ ₹{:.0f} Cr)".format(budget / 1e7),
                   within))

    # ------------------------------------------------------------------
    # 8. Verification summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("VERIFICATION SUMMARY")
    print("=" * 72)
    all_ok = True
    for name, ok in checks:
        status = "PASS" if ok else "FAIL"
        print("[{}] {}".format(status, name))
        all_ok &= bool(ok)
    print("\nRESULT:", "PASS" if all_ok else "ISSUES FOUND")
    print("\nReminder: all cooling figures are ASSUMED-coefficient scenario")
    print("estimates, not measured physical outcomes.")


if __name__ == "__main__":
    main()