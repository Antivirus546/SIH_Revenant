"""
UrbanHeat AI – Intervention Simulator (non-ML core)
Team Revenant | SIH 2026 | IHSIH031

Purpose
-------
Apply cooling interventions (green cover, cool roofs, albedo, water bodies)
to a 100 m analytical grid, estimate the resulting change in heat stress
using a trained ML model, and compute approximate costs.

This module is deliberately model-agnostic:
- It only needs a callable that accepts a feature matrix and returns predictions.
- Later you will plug in Darshith's trained XGBoost / Random Forest model.

Author: Ajay (+ Karthikeya)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# 1. Intervention definitions
# ----------------------------------------------------------------------

@dataclass
class Intervention:
    """Single intervention type with its effect on features and unit cost."""
    name: str
    # How much each unit of intervention changes the feature values
    # Example: +0.15 NDVI per 10% increase in green cover
    feature_deltas: Dict[str, float]
    # Approximate cost in INR per unit area (m²) or per unit intensity
    cost_per_m2: float
    # Human-readable description
    description: str = ""


# Pre-defined interventions for Bengaluru MVP

""" ## These numbers are placeholders – refine with literature / local costs later.
DEFAULT_INTERVENTIONS: Dict[str, Intervention] = {
    "green_cover": Intervention(
        name="green_cover",
        feature_deltas={
            "ndvi": 0.12,          # moderate greening
            "albedo": 0.02,        # slight increase
            "imperviousness": -0.08,
            "ndbi": -0.05,
        },
        cost_per_m2=350.0,         # tree planting + maintenance (rough)
        description="Increase vegetative cover (street trees, parks, vertical greening)",
    ),
    "cool_roofs": Intervention(
        name="cool_roofs",
        feature_deltas={
            "albedo": 0.25,        # strong reflective effect
            "ndbi": -0.03,
        },
        cost_per_m2=180.0,         # coating / cool paint
        description="Apply high-albedo cool roof coatings on buildings",
    ),
    "albedo_boost": Intervention(
        name="albedo_boost",
        feature_deltas={
            "albedo": 0.15,        # pavements, open surfaces
        },
        cost_per_m2=120.0,
        description="Increase surface albedo of roads and open spaces",
    ),
    "water_bodies": Intervention(
        name="water_bodies",
        feature_deltas={
            "ndvi": 0.04,
            "albedo": -0.05,       # water is darker
            "imperviousness": -0.10,
        },
        cost_per_m2=900.0,         # higher cost (excavation, lining, etc.)
        description="Create or expand small water bodies / wetlands",
    ),
}
 """
# ----------------------------------------------------------------------
# 1. Intervention definitions (Updated with Researched Market Rates)
# ----------------------------------------------------------------------

DEFAULT_INTERVENTIONS: Dict[str, Intervention] = {
    "green_cover": Intervention(
        name="green_cover",
        feature_deltas={
            # Miyawaki urban forestry creates dense canopy rapidly, drastically boosting NDVI
            "ndvi": 0.20,          
            "albedo": 0.02,        # Leaves offer a slight albedo increase over dark asphalt
            "imperviousness": -0.15, # Breaks up concrete, increasing water absorption
            "ndbi": -0.10,         # Reduces the built-up index signature
        },
        # Cost Basis: Indian market rate for Miyawaki site prep, soil, and saplings (₹300 - ₹500)
        cost_per_m2=400.0,         
        description="High-density Miyawaki urban afforestation",
    ),
    "cool_roofs": Intervention(
        name="cool_roofs",
        feature_deltas={
            # White elastomeric paint reflects massive amounts of shortwave radiation
            "albedo": 0.35,        
            "ndbi": -0.05,
        },
        # Cost Basis: High-SRI cool roof paint application (~₹45/sq.ft. converted to m2)
        cost_per_m2=480.0,         
        description="Elastomeric high-SRI cool roof coatings on buildings",
    ),
    "albedo_boost": Intervention(
        name="albedo_boost",
        feature_deltas={
            # General pavement whitening / cool pavements
            "albedo": 0.15,        
        },
        # Cost Basis: Lower than elastomeric roof paint, basic reflective surfacing
        cost_per_m2=250.0,         
        description="Increase surface albedo of roads and open pavements",
    ),
    "water_bodies": Intervention(
        name="water_bodies",
        feature_deltas={
            "ndvi": 0.05,          # Riparian edges add slight vegetation
            "albedo": -0.05,       # Water absorbs light, technically lowering albedo
            "imperviousness": -0.20, # Replaces concrete with permeable retention
        },
        # Cost Basis: Median municipal lake/pond excavation & impermeable lining 
        cost_per_m2=2500.0,        
        description="Create or restore urban water retention ponds",
    ),
}
# ----------------------------------------------------------------------
# 2. Grid representation
# ----------------------------------------------------------------------

@dataclass
class GridCell:
    """One 100 m × 100 m cell."""
    cell_id: str
    ward: str
    # Core features that the ML model will use
    features: Dict[str, float]
    population: float = 0.0
    area_m2: float = 10_000.0          # 100 m × 100 m
    # Optional: original heat-stress prediction (baseline)
    baseline_heat_stress: Optional[float] = None


class UrbanGrid:
    """
    Holds the entire analytical grid for the selected wards.
    For Phase 1 we use synthetic / dummy data.
    Later Anirudh will replace this with real GEE-derived GeoDataFrame.
    """

    def __init__(self, cells: List[GridCell]):
        self.cells = {c.cell_id: c for c in cells}
        self.feature_names = self._infer_feature_names()

    def _infer_feature_names(self) -> List[str]:
        if not self.cells:
            return []
        first = next(iter(self.cells.values()))
        return list(first.features.keys())

    def to_dataframe(self) -> pd.DataFrame:
        """Convert grid to a tidy DataFrame (one row per cell)."""
        rows = []
        for cell in self.cells.values():
            row = {
                "cell_id": cell.cell_id,
                "ward": cell.ward,
                "population": cell.population,
                "area_m2": cell.area_m2,
                "baseline_heat_stress": cell.baseline_heat_stress,
            }
            row.update(cell.features)
            rows.append(row)
        return pd.DataFrame(rows)

    def get_feature_matrix(self, feature_order: Optional[List[str]] = None) -> np.ndarray:
        """Return X matrix in the order expected by the ML model."""
        if feature_order is None:
            feature_order = self.feature_names
        df = self.to_dataframe()
        return df[feature_order].values.astype(float)

    def copy(self) -> "UrbanGrid":
        """Deep-ish copy so interventions do not mutate the original."""
        new_cells = []
        for c in self.cells.values():
            new_cells.append(
                GridCell(
                    cell_id=c.cell_id,
                    ward=c.ward,
                    features=c.features.copy(),
                    population=c.population,
                    area_m2=c.area_m2,
                    baseline_heat_stress=c.baseline_heat_stress,
                )
            )
        return UrbanGrid(new_cells)


# ----------------------------------------------------------------------
# 3. Simulator core
# ----------------------------------------------------------------------

class InterventionSimulator:
    """
    Applies interventions and estimates impact.
    """

    def __init__(
        self,
        grid: UrbanGrid,
        interventions: Optional[Dict[str, Intervention]] = None,
        model_predict_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        feature_order: Optional[List[str]] = None,
    ):
        self.baseline_grid = grid
        self.interventions = interventions or DEFAULT_INTERVENTIONS
        self.model_predict_fn = model_predict_fn
        self.feature_order = feature_order or grid.feature_names

        # Cache baseline predictions if a model is already available
        self.baseline_predictions: Optional[np.ndarray] = None
        if self.model_predict_fn is not None:
            X = self.baseline_grid.get_feature_matrix(self.feature_order)
            self.baseline_predictions = self.model_predict_fn(X)

    def apply_interventions(
        self,
        plan: Dict[str, float],
        target_cells: Optional[List[str]] = None,
    ) -> UrbanGrid:
        """
        Apply a set of interventions to selected cells.

        Parameters
        ----------
        plan : dict
            {intervention_name: intensity}
            intensity is usually in [0, 1] where 1 = full recommended dose.
            Example: {"green_cover": 0.6, "cool_roofs": 0.4}
        target_cells : list of cell_id, optional
            If None, apply to every cell. Otherwise only to the listed cells.

        Returns
        -------
        A new UrbanGrid with modified features.
        """
        new_grid = self.baseline_grid.copy()

        cells_to_touch = (
            target_cells if target_cells is not None else list(new_grid.cells.keys())
        )

        for cell_id in cells_to_touch:
            cell = new_grid.cells[cell_id]
            for int_name, intensity in plan.items():
                if intensity <= 0:
                    continue
                if int_name not in self.interventions:
                    raise ValueError(f"Unknown intervention: {int_name}")

                interv = self.interventions[int_name]
                for feat, delta in interv.feature_deltas.items():
                    if feat in cell.features:
                        # Apply proportional change; clamp to realistic ranges later
                        cell.features[feat] += delta * intensity

                # Optional: soft clamping to keep features in physical range
                self._clamp_features(cell)

        return new_grid

    def _clamp_features(self, cell: GridCell) -> None:
        """Keep features inside plausible physical bounds."""
        bounds = {
            "ndvi": (0.0, 0.95),
            "ndbi": (-0.5, 0.6),
            "albedo": (0.05, 0.85),
            "imperviousness": (0.0, 1.0),
        }
        for feat, (lo, hi) in bounds.items():
            if feat in cell.features:
                cell.features[feat] = float(np.clip(cell.features[feat], lo, hi))

    def estimate_impact(
        self,
        plan: Dict[str, float],
        target_cells: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Full pipeline: apply interventions → predict → compute metrics.

        Returns a dictionary with everything the optimizer / dashboard needs.
        """
        modified_grid = self.apply_interventions(plan, target_cells)

        if self.model_predict_fn is None:
            # Phase-1 fallback: simple heuristic so the rest of the pipeline works
            preds = self._heuristic_heat_stress(modified_grid)
            baseline = self._heuristic_heat_stress(self.baseline_grid)
        else:
            X_mod = modified_grid.get_feature_matrix(self.feature_order)
            preds = self.model_predict_fn(X_mod)
            baseline = self.baseline_predictions

        # Metrics
        delta = baseline - preds                     # positive = cooling
        total_cooling = float(np.sum(delta))
        mean_cooling = float(np.mean(delta))
        max_cooling = float(np.max(delta))

        # Population-weighted cooling
        pop = np.array([c.population for c in modified_grid.cells.values()])
        pop_weighted_cooling = float(np.sum(delta * pop) / (np.sum(pop) + 1e-6))

        # Cost
        total_cost = self._compute_cost(plan, target_cells, modified_grid)

        return {
            "plan": plan,
            "modified_grid": modified_grid,
            "predictions": preds,
            "baseline_predictions": baseline,
            "delta_per_cell": delta,
            "total_cooling": total_cooling,
            "mean_cooling": mean_cooling,
            "max_cooling": max_cooling,
            "pop_weighted_cooling": pop_weighted_cooling,
            "total_cost_inr": total_cost,
            "cooling_per_lakh_inr": (total_cooling / (total_cost / 1e5 + 1e-6)),
        }

    def _compute_cost(
        self,
        plan: Dict[str, float],
        target_cells: Optional[List[str]],
        grid: UrbanGrid,
    ) -> float:
        cells = (
            [grid.cells[cid] for cid in target_cells]
            if target_cells is not None
            else list(grid.cells.values())
        )
        total = 0.0
        for cell in cells:
            for int_name, intensity in plan.items():
                if intensity <= 0:
                    continue
                interv = self.interventions[int_name]
                # cost = intensity * area * unit cost
                total += intensity * cell.area_m2 * interv.cost_per_m2
        return total

    def _heuristic_heat_stress(self, grid: UrbanGrid) -> np.ndarray:
        """
        Very rough proxy used only until the real model arrives.
        Higher NDVI & albedo → lower heat stress.
        """
        scores = []
        for cell in grid.cells.values():
            ndvi = cell.features.get("ndvi", 0.3)
            albedo = cell.features.get("albedo", 0.2)
            imp = cell.features.get("imperviousness", 0.6)
            # Simple linear combination (arbitrary weights)
            stress = 0.45 * imp - 0.35 * ndvi - 0.25 * albedo + 0.4
            scores.append(max(0.0, min(1.0, stress)))
        return np.array(scores)


# ----------------------------------------------------------------------
# 4. Helper: create a small dummy grid for immediate testing
# ----------------------------------------------------------------------

def create_dummy_bengaluru_grid(n_cells: int = 25, seed: int = 42) -> UrbanGrid:
    """
    Synthetic 5×5 grid that looks roughly like a few Bengaluru wards.
    Replace this later with real data from Anirudh.
    """
    rng = np.random.default_rng(seed)
    cells = []
    wards = ["Ward_1", "Ward_2", "Ward_3", "Ward_4", "Ward_5"]

    for i in range(n_cells):
        ward = wards[i % 5]
        cell_id = f"{ward}_C{i:02d}"

        # Realistic-ish random features
        features = {
            "ndvi": float(rng.uniform(0.15, 0.55)),
            "ndbi": float(rng.uniform(-0.1, 0.35)),
            "albedo": float(rng.uniform(0.12, 0.28)),
            "imperviousness": float(rng.uniform(0.35, 0.85)),
            "built_up_density": float(rng.uniform(0.2, 0.9)),
            "population_density": float(rng.uniform(50, 400)),  # people per cell
        }

        cells.append(
            GridCell(
                cell_id=cell_id,
                ward=ward,
                features=features,
                population=features["population_density"] * 0.8,  # rough
                baseline_heat_stress=None,
            )
        )
    return UrbanGrid(cells)


# ----------------------------------------------------------------------
# 5. Quick self-test (run this file directly)
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("=== UrbanHeat AI – Intervention Simulator (Phase-1 scaffold) ===\n")

    # 1. Create dummy grid
    grid = create_dummy_bengaluru_grid(n_cells=20)
    print(f"Created dummy grid with {len(grid.cells)} cells")
    print("Feature names:", grid.feature_names)
    print()

    # 2. Initialise simulator (no real model yet → uses heuristic)
    sim = InterventionSimulator(grid)

    # 3. Define a simple plan
    example_plan = {
        "green_cover": 0.7,
        "cool_roofs": 0.5,
        "albedo_boost": 0.3,
    }
    print("Example intervention plan:", example_plan)
    print()

    # 4. Run impact estimation
    result = sim.estimate_impact(example_plan)

    print("Results")
    print("-" * 40)
    print(f"Total cooling (sum of Δ):     {result['total_cooling']:.3f}")
    print(f"Mean cooling per cell:        {result['mean_cooling']:.3f}")
    print(f"Max cooling in any cell:      {result['max_cooling']:.3f}")
    print(f"Population-weighted cooling:  {result['pop_weighted_cooling']:.3f}")
    print(f"Total estimated cost (INR):   {result['total_cost_inr']:,.0f}")
    print(f"Cooling per lakh INR:         {result['cooling_per_lakh_inr']:.4f}")
    print()

    # 5. Show a few modified cells
    print("Sample of modified features (first 3 cells):")
    df = result["modified_grid"].to_dataframe()
    print(df[["cell_id", "ndvi", "albedo", "imperviousness"]].head(3).to_string(index=False))
    print("\nScaffold is ready. Next: plug in real model + real grid.")

# ----------------------------------------------------------------------
# 6. Optimizer Helpers
# ----------------------------------------------------------------------

def generate_random_intervention_plans(num_plans=10):
    """
    Generates random intervention plans for the NSGA-II optimizer to evaluate.
    Intensity values represent the percentage of a 100m grid cell to alter (0.0 to 1.0).
    """
    import random
    plans = []
    
    for _ in range(num_plans):
        plan = {
            # Bounded to 40%: Cannot realistically plant trees over an entire city block
            "green_cover": round(random.uniform(0, 0.4), 2),   
            
            # Bounded to 60%: Assumes a maximum of 60% of the cell is paintable roof area
            "cool_roofs": round(random.uniform(0, 0.6), 2),    
            
            # Bounded to 30%: Reflective pavement coverage limits
            "albedo_boost": round(random.uniform(0, 0.3), 2),
            
            # Bounded to 15%: Extremely difficult to find large open space for new lakes
            "water_bodies": round(random.uniform(0, 0.15), 2)  
        }
        plans.append(plan)
        
    return plans