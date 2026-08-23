# Intervention Simulator + Optimizer — UrbanHeat AI (Team Revenant)

SIH 2026 | IHSIH031 | Branch: `simulator-optimizer`

A decision-support subsystem that simulates urban cooling interventions
on the processed Ward-1 (Bengaluru) 100 m grid and optimizes intervention
plans with NSGA-II. **It is an MVP built on ASSUMED scenario
coefficients — not a validated physical urban microclimate model.**

---

## Current Architecture

```
ward1_processed.csv  (1530 cells; Data-pipeline output)
        │
        ▼
model_adapter.py ── locally retrained XGBoost reproducing the
        │            ml-modeling specification (°C target)
        ▼
intervention_simulater.py ── heuristic engine (default) or ML engine
        │
        ▼
optimizer.py ── dependency-free NSGA-II + greedy baseline
        │
        ▼
demo_usage.py ── end-to-end offline demo / CLI output
```

Files in this folder:

| File | Role |
|---|---|
| `intervention_simulater.py` | Grid loading, heuristic + ML simulation engines, self-tests |
| `model_adapter.py` | Local ML adapter: retrain/cache/predict per ml-modeling contract |
| `optimizer.py` | NSGA-II (no pymoo) + greedy baseline, self-tests |
| `demo_usage.py` | Full offline pipeline demo |
| `requirements.txt` | Python dependencies for this folder |

---

## °C Unit Correction (important semantic change)

Earlier versions treated raw Landsat ST_B10 digital numbers (~47,000) as
temperatures. All temperature math now uses **degrees Celsius**, applying
the same conversion as `ml-modeling/model_pipeline.py`:

```
celsius = DN × 0.00341802 + 149.0 − 273.15
```

The raw DN is still stored per cell (`features["target_temp_dn"]`) for
reference only and is never used in cooling calculations. All user-facing
outputs (`baseline_heat_stress`, `predictions`, cooling deltas) are °C.
Typical baselines on this dataset: ~36–39 °C (March–May midday LST).

---

## Simulation Modes

### Heuristic mode (default — backward compatible)

Linear scenario model per cell:

```
ΔT_cell = Σ_interventions  intensity × (applicable_area / cell_area) × max_delta_c
```

The legacy per-m² `cooling_factor` constants operated on meaningless
DN-scale values and have been replaced by °C-scale `max_delta_c`
coefficients (see Tunable coefficients).

### ML mode

Cooling comes from the locally retrained model's response to
intervention-driven feature modifications:

```
ΔT_cell = f(features_baseline) − f(features_modified)
```

Feature modifications (ASSUMED coefficients):

| Intervention | Feature change |
|---|---|
| cool_roofs | `albedo += 0.30 × intensity × (roof_area / cell_area)` |
| albedo_boost | `albedo += 0.20 × intensity × (road_area / cell_area)` |
| green_cover | `ndvi += 0.45 × intensity × (open_area / cell_area)`; `ndbi -= 0.05 × intensity × (open_area / cell_area)` |
| water_bodies | **heuristic term only** — NDWI is not one of the model's features, so water bodies cannot be represented through model features. This limitation is deliberate and documented. |

Modified features are clamped to physical ranges:
`ndvi ∈ [-0.20, 0.90]`, `ndbi ∈ [-0.50, 0.50]`, `albedo ∈ [0.03, 0.60]`.

If XGBoost/scikit-learn is unavailable, `create_ml_simulator()` falls
back to heuristic mode with a warning instead of crashing.

---

## Model Adapter

**`model_adapter.py` provides a LOCALLY RETRAINED MODEL REPRODUCING THE
EXISTING ml-modeling SPECIFICATION.** It does NOT load an "original
trained model" because `ml-modeling/model_pipeline.py` never persists
one (no joblib/pickle dump exists there).

Reproduced specification (identical to ml-modeling):

- Features, exact order: `bldg_area_sqm, road_length_m, ndvi, ndbi, albedo`
- Estimator: `XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)`
- Target: ST_B10 DN → °C conversion above
- Split: `train_test_split(test_size=0.2, random_state=42)`; fit on the training fold

The trained model is cached to `heat_stress_model.pkl` next to the module.
The cache records its feature order/hyperparameters and auto-invalidates
if the specification changes. Delete the file (or pass
`refresh_cache=True`) to force a retrain.

---

## Optimizer

`optimizer.py` implements standard **NSGA-II without pymoo** (numpy only):
fast non-dominated sorting, crowding distance, binary tournament,
simulated binary crossover (SBX), polynomial mutation, constraint
domination for budgets, Pareto extraction from all evaluated points.

- Decision variables: `[cool_roofs, green_cover, albedo_boost, water_bodies]`, each ∈ [0, 1]
- Objectives (minimised internally): `−pop-weighted cooling (°C)`, `total cost (₹)`
  - optional third objective: `std of per-cell cooling` (inequality) via
    `objectives=("cooling", "cost", "inequality")`
- Optional budget constraint: `budget_inr=` (Deb constraint-domination)
- Cell targeting: all cells, caller-selected list, or hottest top-k%
  via `simulator.select_hottest_cells(fraction)`

`greedy_baseline()` hill-climbs on marginal cooling-per-rupee. It is a
comparison reference ONLY — no assumption is made about whether NSGA-II
or greedy performs better; results decide.

---

## Public Interface Contract (backward compatible)

`estimate_impact(plan, target_cells=None)` always returns a dict with at least:

| Key | Meaning |
|---|---|
| `plan` | the evaluated plan dict |
| `total_cooling` | sum of per-cell ΔT (°C) |
| `mean_cooling` | mean ΔT over ALL cells (°C) |
| `pop_weighted_cooling` | population-weighted mean ΔT (°C) |
| `total_cost_inr` | total implementation cost (₹) |
| `cooling_per_lakh_inr` | efficiency metric |
| `delta_per_cell` | numpy array of per-cell ΔT (°C) |
| `modified_grid` | new UrbanGrid with `temp_reduction` (+ modified features in ML mode) |

Additional keys (additive): `max_cooling`, `predictions`,
`baseline_predictions`, `mode`, `baseline_mean_temp_c`, `predictions_ml`.

Preserved public API: `Intervention`, `GridCell`, `UrbanGrid`,
`InterventionSimulator(grid, interventions=None, model_predict_fn=None,
feature_order=None, mode="heuristic")`, `apply_interventions()`,
`estimate_impact()`, `load_ward1_grid(csv_path, ward_name)`,
`DEFAULT_INTERVENTIONS`. New additions: `evaluate_plan_metrics()`,
`select_hottest_cells()`, `baseline_temps_c`, `create_ml_simulator()`.

Note: `Intervention.cooling_factor` is deprecated (kept for construction
compatibility); engines use `max_delta_c`.

---

## How to Run

From this folder:

```bash
# Simulator self-tests (heuristic + ML if available)
python intervention_simulater.py

# Optimizer self-tests (small NSGA-II run + greedy comparison)
python optimizer.py

# Full end-to-end demo (simulation → optimization → verification summary)
python demo_usage.py

# Model adapter self-test (trains fresh, prints plausibility checks)
python model_adapter.py
```

Programmatic use:

```python
from intervention_simulater import load_ward1_grid, create_ml_simulator
from optimizer import NSGA2Optimizer, greedy_baseline

grid = load_ward1_grid("ward1_processed.csv")
sim, info = create_ml_simulator(grid)          # falls back to heuristic
print(info["mode"])                            # "ml" or "heuristic"

hot = sim.select_hottest_cells(0.3)            # hottest 30% of cells
result = NSGA2Optimizer(sim, target_cells=hot,
                        pop_size=60, n_generations=120,
                        budget_inr=5e7).optimize()
for s in result.pareto_solutions[:5]:
    print(s["plan"], s["cooling"], s["cost"])

g_plan, g_summary = greedy_baseline(sim, target_cells=hot)
```

Dependencies: see `requirements.txt`. Core needs are numpy + pandas;
xgboost + scikit-learn are OPTIONAL (ML mode falls back to heuristic
without them).

---

## OBSERVED vs ASSUMED (read this before quoting numbers)

**OBSERVED**
- Satellite-derived input features (NDVI, NDBI, albedo, building area, road length)
- Observed Landsat LST target (converted to °C)
- The existing ml-modeling specification (features, hyperparameters, split)

**ASSUMED (tunable scenario coefficients — NOT measured)**
- Intervention feature deltas (0.30 albedo for roofs, 0.45 NDVI for greening, …)
- Heuristic max-cooling coefficients (2.5 / 2.0 / 1.5 / 3.0 °C)
- Cost figures (₹180/350/120/900 per m²)
- Population proxy (0.08 people per m² of building area)
- Road width (8 m), open-space approximation, cell area (10,000 m²)
- Intervention applicability assumptions (which area each intervention treats)

**Limitations**
- One ward, one season snapshot; no temporal dynamics.
- The ML model captures statistical associations from a single image;
  random-split evaluation in ml-modeling likely inflates accuracy due to
  spatial autocorrelation (known upstream issue, not addressed here).
- Water-body cooling cannot flow through the ML model (NDWI absent).
- Population is a crude proxy; equity metrics inherit that crudeness.
- No physical validation of any coefficient has been performed.

This subsystem is decision-support for exploring scenarios and trade-offs,
not a validated physical urban microclimate simulator.