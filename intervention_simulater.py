"""
UrbanHeat AI – Intervention Simulator (Real Ward-1 data)
Team Revenant | SIH 2026 | IHSIH031

Now driven by real processed grid:
  grid_id, target_temp, bldg_area_sqm, road_length_m
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any
import numpy as np
import pandas as pd
from pathlib import Path


# ----------------------------------------------------------------------
# 1. Intervention definitions (calibrated for real building/road data)
# ----------------------------------------------------------------------

@dataclass
class Intervention:
    name: str
    # How much temperature reduction (in target_temp units) per unit intensity
    # applied to the relevant area
    cooling_factor: float          # Δtemp per m² of intervention at intensity=1
    cost_per_m2: float
    applies_to: str                # "buildings" | "roads" | "open" | "all"
    description: str = ""


DEFAULT_INTERVENTIONS: Dict[str, Intervention] = {
    "cool_roofs": Intervention(
        name="cool_roofs",
        cooling_factor=0.018,      # strong effect – reflective coating on roofs
        cost_per_m2=180.0,
        applies_to="buildings",
        description="High-albedo cool roof coatings on buildings",
    ),
    "green_cover": Intervention(
        name="green_cover",
        cooling_factor=0.012,      # street trees, parks, vertical greening
        cost_per_m2=350.0,
        applies_to="open",
        description="Increase vegetative cover (trees, parks, green walls)",
    ),
    "albedo_boost": Intervention(
        name="albedo_boost",
        cooling_factor=0.009,      # cool pavements / reflective surfaces
        cost_per_m2=120.0,
        applies_to="roads",
        description="Increase surface albedo of roads and open hard surfaces",
    ),
    "water_bodies": Intervention(
        name="water_bodies",
        cooling_factor=0.025,      # high cooling but expensive
        cost_per_m2=900.0,
        applies_to="open",
        description="Create / expand small water bodies or wetlands",
    ),
}


# ----------------------------------------------------------------------
# 2. Grid representation
# ----------------------------------------------------------------------

@dataclass
class GridCell:
    cell_id: str
    ward: str
    features: Dict[str, float]
    population: float = 0.0
    area_m2: float = 10_000.0          # 100 m × 100 m
    baseline_heat_stress: Optional[float] = None


class UrbanGrid:
    def __init__(self, cells: List[GridCell]):
        self.cells = {c.cell_id: c for c in cells}
        self.feature_names = self._infer_feature_names()

    def _infer_feature_names(self) -> List[str]:
        if not self.cells:
            return []
        return list(next(iter(self.cells.values())).features.keys())

    def to_dataframe(self) -> pd.DataFrame:
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
        if feature_order is None:
            feature_order = self.feature_names
        df = self.to_dataframe()
        return df[feature_order].values.astype(float)

    def copy(self) -> "UrbanGrid":
        new_cells = [
            GridCell(
                cell_id=c.cell_id,
                ward=c.ward,
                features=c.features.copy(),
                population=c.population,
                area_m2=c.area_m2,
                baseline_heat_stress=c.baseline_heat_stress,
            )
            for c in self.cells.values()
        ]
        return UrbanGrid(new_cells)


# ----------------------------------------------------------------------
# 3. Simulator core
# ----------------------------------------------------------------------

class InterventionSimulator:
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
        Apply interventions and return a new grid with updated features
        + estimated temperature reduction stored in features["temp_reduction"].
        """
        new_grid = self.baseline_grid.copy()
        cells_to_touch = (
            target_cells if target_cells is not None
            else list(new_grid.cells.keys())
        )

        for cell_id in cells_to_touch:
            cell = new_grid.cells[cell_id]
            total_cooling = 0.0

            bldg = cell.features.get("bldg_area_sqm", 0.0)
            road = cell.features.get("road_length_m", 0.0)
            # Approximate open space (very rough – roads are linear)
            open_space = max(0.0, cell.area_m2 - bldg - (road * 8.0))  # ~8 m road width

            for int_name, intensity in plan.items():
                if intensity <= 0:
                    continue
                if int_name not in self.interventions:
                    raise ValueError(f"Unknown intervention: {int_name}")

                interv = self.interventions[int_name]

                if interv.applies_to == "buildings":
                    area = bldg
                elif interv.applies_to == "roads":
                    area = road * 8.0          # convert length → approximate area
                elif interv.applies_to == "open":
                    area = open_space
                else:  # "all"
                    area = cell.area_m2

                # Cooling = intensity × area × cooling_factor
                cooling = intensity * area * interv.cooling_factor
                total_cooling += cooling

            # Store the estimated temperature reduction
            cell.features["temp_reduction"] = total_cooling
            # Also keep a soft-clamped version of original features
            cell.features["bldg_area_sqm"] = bldg
            cell.features["road_length_m"] = road
            cell.features["open_space_sqm"] = open_space

        return new_grid

    def estimate_impact(
        self,
        plan: Dict[str, float],
        target_cells: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        modified_grid = self.apply_interventions(plan, target_cells)

        # Baseline temperatures
        baseline = np.array([
            c.baseline_heat_stress if c.baseline_heat_stress is not None else 47000.0
            for c in self.baseline_grid.cells.values()
        ])

        # Predicted = baseline − cooling
        cooling = np.array([
            c.features.get("temp_reduction", 0.0)
            for c in modified_grid.cells.values()
        ])
        preds = baseline - cooling

        delta = cooling                                   # positive = cooling
        total_cooling = float(np.sum(delta))
        mean_cooling = float(np.mean(delta))
        max_cooling = float(np.max(delta))

        # Population-weighted (if population available, else uniform)
        pop = np.array([c.population for c in modified_grid.cells.values()])
        if pop.sum() > 0:
            pop_weighted_cooling = float(np.sum(delta * pop) / pop.sum())
        else:
            pop_weighted_cooling = mean_cooling

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
            "cooling_per_lakh_inr": total_cooling / (total_cost / 1e5 + 1e-9),
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
            bldg = cell.features.get("bldg_area_sqm", 0.0)
            road = cell.features.get("road_length_m", 0.0)
            open_space = cell.features.get("open_space_sqm", max(0.0, cell.area_m2 - bldg))

            for int_name, intensity in plan.items():
                if intensity <= 0:
                    continue
                interv = self.interventions[int_name]

                if interv.applies_to == "buildings":
                    area = bldg
                elif interv.applies_to == "roads":
                    area = road * 8.0
                elif interv.applies_to == "open":
                    area = open_space
                else:
                    area = cell.area_m2

                total += intensity * area * interv.cost_per_m2
        return total


# ----------------------------------------------------------------------
# 4. Real data loader
# ----------------------------------------------------------------------

def load_ward1_grid(
    csv_path: str | Path = "ward1_processed.csv",
    ward_name: str = "Ward_1",
) -> UrbanGrid:
    """
    Load the real processed Ward-1 grid.
    """
    df = pd.read_csv(csv_path)

    cells = []
    for _, row in df.iterrows():
        cell_id = f"{ward_name}_C{int(row['grid_id']):04d}"

        bldg = float(row["bldg_area_sqm"])
        road = float(row["road_length_m"])
        temp = float(row["target_temp"])

        # Simple population proxy (you can replace later)
        # Higher building area → higher population density
        pop = max(0.0, bldg * 0.08)   # rough people per m² of building

        features = {
            "bldg_area_sqm": bldg,
            "road_length_m": road,
            "open_space_sqm": max(0.0, 10_000.0 - bldg - road * 8.0),
            "target_temp": temp,
        }

        cells.append(
            GridCell(
                cell_id=cell_id,
                ward=ward_name,
                features=features,
                population=pop,
                area_m2=10_000.0,
                baseline_heat_stress=temp,
            )
        )

    return UrbanGrid(cells)


# ----------------------------------------------------------------------
# 5. Quick self-test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("=== UrbanHeat AI – Intervention Simulator (Real Ward-1) ===\n")

    # Load real data
    grid = load_ward1_grid("ward1_processed.csv")
    print(f"Loaded real grid → {len(grid.cells)} cells")
    print("Feature names:", grid.feature_names)
    print()

    sim = InterventionSimulator(grid)

    example_plan = {
        "cool_roofs": 0.7,
        "green_cover": 0.4,
        "albedo_boost": 0.3,
    }
    print("Example plan:", example_plan)
    print()

    result = sim.estimate_impact(example_plan)

    print("Results")
    print("-" * 45)
    print(f"Total cooling (sum Δtemp):     {result['total_cooling']:,.1f}")
    print(f"Mean cooling per cell:         {result['mean_cooling']:.2f}")
    print(f"Max cooling in any cell:       {result['max_cooling']:.2f}")
    print(f"Population-weighted cooling:   {result['pop_weighted_cooling']:.2f}")
    print(f"Total estimated cost (INR):    ₹{result['total_cost_inr']:,.0f}")
    print(f"Cooling per lakh INR:          {result['cooling_per_lakh_inr']:.4f}")
    print()

    # Show a few cells
    df = result["modified_grid"].to_dataframe()
    print("Sample cells (first 5):")
    cols = ["cell_id", "bldg_area_sqm", "road_length_m", "temp_reduction", "baseline_heat_stress"]
    print(df[cols].head().to_string(index=False))