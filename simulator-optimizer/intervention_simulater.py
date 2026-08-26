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

        self.mode = "ml" if self.model_predict_fn is not None else "heuristic"

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
        for both the heuristic model (temp_reduction) and the ML model (NDVI, Albedo, etc.).
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
            open_space = max(0.0, cell.area_m2 - bldg - (road * 8.0))

            for int_name, intensity in plan.items():
                if intensity <= 0:
                    continue
                if int_name not in self.interventions:
                    raise ValueError(f"Unknown intervention: {int_name}")

                interv = self.interventions[int_name]

                if interv.applies_to == "buildings":
                    area = bldg
                elif interv.applies_to == "roads":
                    area = road * 8.0
                elif interv.applies_to == "open":
                    area = open_space
                else:  # "all"
                    area = cell.area_m2

                # 1. Heuristic Math (Fallback)
                MAX_COOLING_PER_CELL_C = 4.0   # realistic ceiling for a single intervention type
                fraction_treated = min(intensity, 1.0)
                total_cooling += fraction_treated * interv.cooling_factor * MAX_COOLING_PER_CELL_C

                # 2. ML Feature Modifications (Updates the 7-feature schema)
                if int_name == "cool_roofs":
                    cell.features["albedo"] = min(1.0, cell.features.get("albedo", 0.0) + (0.2 * intensity))
                elif int_name == "albedo_boost":
                    cell.features["albedo"] = min(1.0, cell.features.get("albedo", 0.0) + (0.15 * intensity))
                elif int_name == "green_cover":
                    cell.features["ndvi"] = min(1.0, cell.features.get("ndvi", 0.0) + (0.3 * intensity))
                    cell.features["ndbi"] = max(-1.0, cell.features.get("ndbi", 0.0) - (0.2 * intensity))
                elif int_name == "water_bodies":
                    cell.features["ndwi"] = min(1.0, cell.features.get("ndwi", 0.0) + (0.4 * intensity))

            cell.features["temp_reduction"] = total_cooling
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

        baseline = np.array([
            c.baseline_heat_stress if c.baseline_heat_stress is not None else 35.0
            for c in self.baseline_grid.cells.values()
        ])

        # If ML is active, run the modified features through Darshith's model!
        if self.model_predict_fn is not None:
            X_mod = modified_grid.get_feature_matrix(self.feature_order)
            preds = self.model_predict_fn(X_mod)
            delta = self.baseline_predictions - preds
        # Otherwise, fall back on the linear heuristic math
        else:
            cooling = np.array([
                c.features.get("temp_reduction", 0.0)
                for c in modified_grid.cells.values()
            ])
            delta = cooling
            preds = baseline - cooling

        total_cooling = float(np.sum(delta))
        mean_cooling = float(np.mean(delta))
        max_cooling = float(np.max(delta))

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

    # --- NEW HELPER METHODS FOR OPTIMIZER AND DEMO ---
    
    def evaluate_plan_metrics(
        self,
        plan: Dict[str, float],
        target_cells: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Wrapper required by the optimizer engine."""
        return self.estimate_impact(plan, target_cells)

    def select_hottest_cells(self, fraction: float = 0.25) -> List[str]:
        """Returns the IDs of the hottest fraction of cells."""
        cells = list(self.baseline_grid.cells.values())
        cells.sort(key=lambda c: c.baseline_heat_stress or 0.0, reverse=True)
        count = max(1, int(len(cells) * fraction))
        return [c.cell_id for c in cells[:count]]


# ----------------------------------------------------------------------
# 4. Real data loader & ML factory
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
        
        # Convert raw Landsat ST_B10 DN values to degrees Celsius
        raw_temp = float(row["target_temp"])
        temp = (raw_temp * 0.00341802) + 149.0 - 273.15

        # Simple population proxy (you can replace later)
        # Higher building area → higher population density
        pop = max(0.0, float(row.get("population", bldg * 0.08)))   # real GHSL count, falls back to proxy if column missing
        features = {
            "bldg_area_sqm": bldg,
            "road_length_m": road,
            "open_space_sqm": max(0.0, 10_000.0 - bldg - road * 8.0),
            "target_temp": temp,
        }

        # Dynamically load the ML features if they exist in the CSV
        for col in ['ndvi', 'ndbi', 'albedo', 'ndwi', 'elevation']:
            if col in row:
                features[col] = float(row[col])

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


def create_ml_simulator(
    grid: Optional[UrbanGrid] = None,
    csv_path: str | Path = "ward1_processed.csv",
    ward_name: str = "Ward_1",
) -> tuple[InterventionSimulator, dict]:
    """Factory helper to create an InterventionSimulator powered by the ML adapter."""
    from model_adapter import build_predict_fn, FEATURE_ORDER
    
    if grid is None:
        grid = load_ward1_grid(csv_path, ward_name)
        
    predict_fn, info = build_predict_fn(csv_path=str(csv_path))
    
    # demo_usage.py expects 'mode' at the top level and adapter info under 'model_info'
    ml_info = {"model_info": info}
    
    if predict_fn is None:
        print(f"Warning: ML model not loaded. {info.get('message')}")
        ml_info["mode"] = "heuristic"
    else:
        ml_info["mode"] = "ml"
        
    simulator = InterventionSimulator(
        grid=grid,
        model_predict_fn=predict_fn,
        feature_order=FEATURE_ORDER
    )
    
    return simulator, ml_info


# ----------------------------------------------------------------------
# 5. Quick self-test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("=== UrbanHeat AI – Intervention Simulator (Real Ward-1) ===\n")


    # Load real data from the Data folder
    from pathlib import Path
    ROOT_DIR = Path(__file__).resolve().parent.parent
    CSV_PATH = ROOT_DIR / "Data" / "ward1_processed.csv"
    
    grid = load_ward1_grid(CSV_PATH)

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