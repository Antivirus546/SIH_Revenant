Intervention Simulator – UrbanHeat AI (Team Revenant)

Owner: Ajay + Karthikeya
Phase: 1 (Foundation & De-Risking) – Aug 20–22

What this module does





Holds a 100 m grid (features + population).



Applies cooling interventions by changing the relevant features.



Estimates the resulting change in heat stress (uses a real ML model when available, otherwise a simple heuristic).



Calculates approximate implementation cost.



Returns metrics that the NSGA-II optimizer and Streamlit dashboard need.

Files





intervention_simulator.py – main module (ready to import)

Quick start

python intervention_simulator.py

You should see a successful run with dummy data.

How to use from other code

from intervention_simulator import (
    create_dummy_bengaluru_grid,
    InterventionSimulator,
    DEFAULT_INTERVENTIONS,
)

# 1. Load grid (replace with real data later)
grid = create_dummy_bengaluru_grid(n_cells=50)

# 2. Create simulator
sim = InterventionSimulator(grid)

# 3. Define a plan (intensity 0–1)
plan = {
    "green_cover": 0.6,
    "cool_roofs": 0.4,
    "water_bodies": 0.2,
}

# 4. Get impact
result = sim.estimate_impact(plan)

print(result["total_cooling"])
print(result["total_cost_inr"])
print(result["pop_weighted_cooling"])

When the real model arrives (Darshith)

import joblib
model = joblib.load("heat_stress_model.joblib")

def predict_fn(X):
    return model.predict(X)

sim = InterventionSimulator(
    grid=real_grid,
    model_predict_fn=predict_fn,
    feature_order=["ndvi", "ndbi", "albedo", "imperviousness", ...]  # exact order model expects
)

Next steps for Phase 1 (Ajay / Karthikeya)





[x] Review and tweak the feature_deltas and cost_per_m2 values (make them more realistic).



[ ] Decide whether interventions are applied to all cells, only hotspots, or user-selected wards.



[x] Add a simple function that returns a list of candidate plans for the optimizer to evaluate.



[ ] Keep the interface stable so Srija/Bindu can already mock the dashboard against these return keys.

Interface contract (do not break these keys)

estimate_impact() always returns a dict containing at least:





plan



total_cooling



mean_cooling



pop_weighted_cooling



total_cost_inr



cooling_per_lakh_inr



delta_per_cell



modified_grid

