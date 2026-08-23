"""
UrbanHeat AI – Intervention Simulator (Real Ward-1 data)
Team Revenant | SIH 2026 | IHSIH031

Driven by the real processed grid (ward1_processed.csv):
    grid_id, target_temp, ndvi, ndbi, albedo, bldg_area_sqm, road_length_m

=======================================================================
UNIT CORRECTION (semantic change vs. earlier versions)
=======================================================================
Earlier versions of this simulator treated the raw Landsat ST_B10
digital numbers (~47,000) as if they were temperatures. All internal
temperature calculations now use DEGREES CELSIUS, applying the same
conversion as ml-modeling/model_pipeline.py:

    celsius = DN * 0.00341802 + 149.0 - 273.15

The raw DN is retained per cell (features["target_temp_dn"]) for
reference only. It is NEVER used in cooling calculations. All
user-facing temperature outputs (baseline_heat_stress, predictions,
cooling deltas) are in °C.

=======================================================================
SIMULATION MODES
=======================================================================
mode="heuristic"  (default — backward compatible)
    Cooling is a linear scenario model:
        ΔT_cell = Σ_interventions intensity × (applicable_area / cell_area)
                       × max_delta_c
    where max_delta_c is an ASSUMED maximum cooling (°C) when the whole
    applicable area is treated at intensity = 1. The old per-m²
    "cooling_factor" constants operated on meaningless DN-scale values;
    they have been replaced by physically-plausible °C-scale
    coefficients (see ASSUMPTIONS below).

mode="ml"
    Cooling comes from the locally retrained ML model's response to
    intervention-driven FEATURE modifications:
        ΔT_cell = f(features_baseline) − f(features_modified)
    plus a heuristic term for water_bodies (NDWI is not one of the
    model's features, so water bodies CANNOT be represented through
    model features — this limitation is deliberate and documented).
    Use create_ml_simulator() to build an ML-mode simulator; it falls
    back to heuristic mode automatically if XGBoost is unavailable.

=======================================================================
ASSUMPTIONS (NOT experimentally measured — tunable scenario coefficients)
=======================================================================
OBSERVED : satellite-derived features, observed LST target.
ASSUMED  : everything below.

Geometry / context assumptions
    - Cell area .................... 10,000 m² (100 m × 100 m)
    - Road width ................... 8 m (road_length_m × 8 = road area)
    - Open space ................... cell_area − buildings − road area
    - Population proxy ............. 0.08 people per m² of building area

Heuristic-mode coefficients (max Δ°C at full applicable-area coverage)
    - cool_roofs ....... 2.5 °C
    - green_cover ...... 2.0 °C
    - albedo_boost ..... 1.5 °C
    - water_bodies ..... 3.0 °C

ML-mode feature-delta coefficients (applied to cell-mean features)
    - cool_roofs:   albedo += 0.30 × intensity × (roof_area / cell_area)
    - albedo_boost: albedo += 0.20 × intensity × (road_area / cell_area)
    - green_cover:  ndvi   += 0.45 × intensity × (open_area / cell_area)
                    ndbi   -= 0.05 × intensity × (open_area / cell_area)
    - water_bodies: heuristic only (see above)

Physical clamps on modified features
    - ndvi ∈ [-0.20, 0.90], ndbi ∈ [-0.50, 0.50], albedo ∈ [0.03, 0.60]

Costs (₹ per m² of treated area — assumed planning figures)
    - cool_roofs ₹180, green_cover ₹350, albedo_boost ₹120,
      water_bodies ₹900

This module is a decision-support MVP, not a validated physical urban
microclimate simulator.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from pathlib import Path

# ----------------------------------------------------------------------
# Unit conversion (identical to ml-modeling/model_pipeline.py)
# ----------------------------------------------------------------------

DN_TO_CELSIUS_SCALE = 0.00341802
DN_TO_CELSIUS_OFFSET = 149.0 - 273.15  # Kelvin offset minus absolute zero


def dn_to_celsius(dn) -> np.ndarray:
    """Convert raw Landsat ST_B10 DN values to degrees Celsius."""
    return np.asarray(dn, dtype=float) * DN_TO_CELSIUS_SCALE + DN_TO_CELSIUS_OFFSET


# Fallback baseline for cells missing baseline_heat_stress: the old code
# used 47000.0 raw DN; we keep that reference value but convert it to °C.
FALLBACK_BASELINE_DN = 47000.0
FALLBACK_BASELINE_C = float(dn_to_celsius(FALLBACK_BASELINE_DN))

# ----------------------------------------------------------------------
# Geometry / assumption constants (see module docstring)
# ----------------------------------------------------------------------

ROAD_WIDTH_M = 8.0
DEFAULT_CELL_AREA_M2 = 10_000.0
POPULATION_PROXY_PER_SQM_BUILDING = 0.08

# ML-mode feature-delta coefficients (ASSUMED — tunable)
ML_COOL_ROOFS_ALBEDO_DELTA = 0.30
ML_ALBEDO_BOOST_ALBEDO_DELTA = 0.20
ML_GREEN_COVER_NDVI_DELTA = 0.45
ML_GREEN_COVER_NDBI_DELTA = -0.05
ML_WATER_BODIES_MAX_DELTA_C = 3.0

# Physical clamps for modified features
PHYSICAL_BOUNDS: Dict[str, tuple] = {
    "ndvi": (-0.20, 0.90),
    "ndbi": (-0.50, 0.50),
    "albedo": (0.03, 0.60),
}

# Feature order expected by the ML model (mirrors model_adapter /
# ml-modeling). Used when mode="ml" and no explicit feature_order given.
DEFAULT_ML_FEATURE_ORDER = [
    "bldg_area_sqm",
    "road_length_m",
    "ndvi",
    "ndbi",
    "albedo",
]


# ----------------------------------------------------------------------
# 1. Intervention definitions
# ----------------------------------------------------------------------

@dataclass
class Intervention:
    """A cooling intervention definition.

    Attributes
    ----------
    name :
        Identifier used inside plans, e.g. {"cool_roofs": 0.7}.
    max_delta_c :
        ASSUMED maximum cooling in °C when the entire applicable area
        of a cell is treated at intensity = 1 (heuristic engine; also
        used for the water_bodies heuristic term in ML mode).
    cost_per_m2 :
        ASSUMED implementation cost in ₹ per m² of treated area.
    applies_to :
        Which part of the cell the intervention treats:
        "buildings" | "roads" | "open" | "all".
    cooling_factor :
        DEPRECATED legacy field from the DN-scale era. Kept only so
        older code constructing Intervention objects keeps working; the
        engines no longer use it.
    description :
        Human-readable description.
    """

    name: str
    max_delta_c: float
    cost_per_m2: float
    applies_to: str                # "buildings" | "roads" | "open" | "all"
    cooling_factor: float = 0.0    # deprecated legacy field (unused)
    description: str = ""


DEFAULT_INTERVENTIONS: Dict[str, Intervention] = {
    "cool_roofs": Intervention(
        name="cool_roofs",
        max_delta_c=2.5,           # ASSUMED — reflective coating effect
        cost_per_m2=180.0,         # ASSUMED planning cost
        applies_to="buildings",
        description="High-albedo cool roof coatings on buildings",
    ),
    "green_cover": Intervention(
        name="green_cover",
        max_delta_c=2.0,           # ASSUMED — street trees / parks effect
        cost_per_m2=350.0,         # ASSUMED planning cost
        applies_to="open",
        description="Increase vegetative cover (trees, parks, green walls)",
    ),
    "albedo_boost": Intervention(
        name="albedo_boost",
        max_delta_c=1.5,           # ASSUMED — cool pavement effect
        cost_per_m2=120.0,         # ASSUMED planning cost
        applies_to="roads",
        description="Increase surface albedo of roads and open hard surfaces",
    ),
    "water_bodies": Intervention(
        name="water_bodies",
        max_delta_c=3.0,           # ASSUMED — high cooling but expensive
        cost_per_m2=900.0,         # ASSUMED planning cost
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
    area_m2: float = DEFAULT_CELL_AREA_M2   # 100 m × 100 m
    baseline_heat_stress: Optional[float] = None   # °C after the unit fix


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
        return df[feature_order].to_numpy(dtype=float)

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
    """Simulate cooling-intervention plans over an urban grid.

    Parameters
    ----------
    grid :
        UrbanGrid built by load_ward1_grid (temperatures in °C).
    interventions :
        Override the DEFAULT_INTERVENTIONS registry if desired.
    model_predict_fn :
        Callable X(n, d) -> temperatures in °C. Required for mode="ml".
    feature_order :
        Column order the model expects. Defaults to the ML contract
        order when mode="ml".
    mode :
        "heuristic" (default, backward compatible) or "ml".
    """

    def __init__(
        self,
        grid: UrbanGrid,
        interventions: Optional[Dict[str, Intervention]] = None,
        model_predict_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        feature_order: Optional[List[str]] = None,
        mode: str = "heuristic",
    ):
        if mode not in ("heuristic", "ml"):
            raise ValueError(f"Unknown mode: {mode!r} (use 'heuristic' or 'ml')")
        if mode == "ml" and model_predict_fn is None:
            raise ValueError("mode='ml' requires model_predict_fn — use "
                             "create_ml_simulator() or pass a predict callable.")

        self.baseline_grid = grid
        self.interventions = interventions or DEFAULT_INTERVENTIONS
        self.model_predict_fn = model_predict_fn
        self.mode = mode

        if feature_order is None:
            feature_order = (
                list(DEFAULT_ML_FEATURE_ORDER) if mode == "ml" else grid.feature_names
            )
        self.feature_order = list(feature_order)

        # ---- Precompute vectorised cell arrays -------------------------
        cells = list(grid.cells.values())
        self._cell_ids: List[str] = [c.cell_id for c in cells]
        self._index: Dict[str, int] = {cid: i for i, cid in enumerate(self._cell_ids)}
        n = len(cells)

        self._bldg = np.array([c.features.get("bldg_area_sqm", 0.0) for c in cells], dtype=float)
        self._road_len = np.array([c.features.get("road_length_m", 0.0) for c in cells], dtype=float)
        self._cell_area = np.array([c.area_m2 for c in cells], dtype=float)
        self._road_area = self._road_len * ROAD_WIDTH_M
        self._open = np.clip(self._cell_area - self._bldg - self._road_area, 0.0, None)
        self._pop = np.array([c.population for c in cells], dtype=float)
        self._base_temp = np.array(
            [c.baseline_heat_stress if c.baseline_heat_stress is not None else FALLBACK_BASELINE_C
             for c in cells],
            dtype=float,
        )

        # Named index features (needed by the ML engine)
        try:
            self._i_bldg = self.feature_order.index("bldg_area_sqm")
            self._i_road = self.feature_order.index("road_length_m")
            self._i_ndvi = self.feature_order.index("ndvi")
            self._i_ndbi = self.feature_order.index("ndbi")
            self._i_albedo = self.feature_order.index("albedo")
        except ValueError as exc:
            raise ValueError(
                f"feature_order must contain the ML model features "
                f"{DEFAULT_ML_FEATURE_ORDER}; got {self.feature_order}"
            ) from exc

        self._X_base = np.column_stack([
            self._bldg, self._road_len,
            np.array([c.features.get("ndvi", 0.0) for c in cells], dtype=float),
            np.array([c.features.get("ndbi", 0.0) for c in cells], dtype=float),
            np.array([c.features.get("albedo", 0.0) for c in cells], dtype=float),
        ])

        self.baseline_predictions: Optional[np.ndarray] = None
        if self.mode == "ml":
            self.baseline_predictions = np.asarray(
                self.model_predict_fn(self._X_base), dtype=float
            )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def cell_ids(self) -> List[str]:
        return list(self._cell_ids)

    @property
    def baseline_temps_c(self) -> np.ndarray:
        """Baseline temperatures in °C (observed LST, converted from DN)."""
        return self._base_temp.copy()

    def select_hottest_cells(self, fraction: float = 0.3) -> List[str]:
        """Return the cell ids of the hottest `fraction` of cells (by °C)."""
        if not 0 < fraction <= 1:
            raise ValueError("fraction must be in (0, 1]")
        k = max(1, int(round(fraction * len(self._cell_ids))))
        hottest = np.argsort(self._base_temp)[-k:]
        return [self._cell_ids[i] for i in hottest]

    def _mask_for(self, target_cells: Optional[List[str]]) -> np.ndarray:
        n = len(self._cell_ids)
        if target_cells is None:
            return np.ones(n, dtype=bool)
        mask = np.zeros(n, dtype=bool)
        for cid in target_cells:
            if cid not in self._index:
                raise KeyError(f"Unknown cell id: {cid}")
            mask[self._index[cid]] = True
        return mask

    # ------------------------------------------------------------------
    # Fast plan evaluation (vectorised — used by estimate_impact AND the
    # optimizer; identical math to the historical per-cell loop)
    # ------------------------------------------------------------------

    def evaluate_plan_metrics(
        self,
        plan: Dict[str, float],
        target_cells: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Evaluate a plan and return metrics WITHOUT building a modified grid.

        This is the fast path used by the optimizer. See estimate_impact()
        for the full result including modified_grid.
        """
        mask = self._mask_for(target_cells)
        return self._evaluate_fast(plan, mask)

    def _evaluate_fast(self, plan: Dict[str, float], mask: np.ndarray) -> Dict[str, Any]:
        n_total = len(self._cell_ids)
        n_sel = int(mask.sum())

        # ---- validate plan -------------------------------------------
        for int_name, intensity in plan.items():
            if int_name not in self.interventions:
                raise ValueError(f"Unknown intervention: {int_name}")

        # ---- cooling + cost ------------------------------------------
        delta_sel = np.zeros(n_sel, dtype=float)
        cost = 0.0
        preds_mod_sel: Optional[np.ndarray] = None
        mod_feats: Optional[Dict[str, np.ndarray]] = None

        if self.mode == "heuristic":
            area_sel = self._cell_area[mask]
            frac_b = self._bldg[mask] / area_sel
            frac_r = self._road_area[mask] / area_sel
            frac_o = self._open[mask] / area_sel

            for int_name, intensity in plan.items():
                intensity = float(intensity)
                if intensity <= 0:
                    continue
                interv = self.interventions[int_name]
                if interv.applies_to == "buildings":
                    frac, ar = frac_b, self._bldg[mask]
                elif interv.applies_to == "roads":
                    frac, ar = frac_r, self._road_area[mask]
                elif interv.applies_to == "open":
                    frac, ar = frac_o, self._open[mask]
                else:  # "all"
                    frac, ar = np.ones(n_sel), area_sel
                # Linear scenario model: intensity × coverage × max Δ°C
                delta_sel += intensity * frac * interv.max_delta_c
                cost += intensity * float(np.sum(ar)) * interv.cost_per_m2

        else:  # mode == "ml"
            area_sel = self._cell_area[mask]
            roof_frac = self._bldg[mask] / area_sel
            road_frac = self._road_area[mask] / area_sel
            open_frac = self._open[mask] / area_sel

            i_cr = float(plan.get("cool_roofs", 0.0))
            i_gc = float(plan.get("green_cover", 0.0))
            i_ab = float(plan.get("albedo_boost", 0.0))
            i_wb = float(plan.get("water_bodies", 0.0))

            lo_a, hi_a = PHYSICAL_BOUNDS["albedo"]
            lo_v, hi_v = PHYSICAL_BOUNDS["ndvi"]
            lo_b, hi_b = PHYSICAL_BOUNDS["ndbi"]

            albedo_mod = np.clip(
                self._X_base[mask][:, self._i_albedo]
                + ML_COOL_ROOFS_ALBEDO_DELTA * i_cr * roof_frac
                + ML_ALBEDO_BOOST_ALBEDO_DELTA * i_ab * road_frac,
                lo_a, hi_a,
            )
            ndvi_mod = np.clip(
                self._X_base[mask][:, self._i_ndvi]
                + ML_GREEN_COVER_NDVI_DELTA * i_gc * open_frac,
                lo_v, hi_v,
            )
            ndbi_mod = np.clip(
                self._X_base[mask][:, self._i_ndbi]
                + ML_GREEN_COVER_NDBI_DELTA * i_gc * open_frac,
                lo_b, hi_b,
            )

            X_mod = self._X_base[mask].copy()
            X_mod[:, self._i_ndvi] = ndvi_mod
            X_mod[:, self._i_ndbi] = ndbi_mod
            X_mod[:, self._i_albedo] = albedo_mod

            preds_mod_sel = np.asarray(self.model_predict_fn(X_mod), dtype=float)
            # ΔT = baseline prediction − modified prediction (positive = cooler)
            delta_sel += self.baseline_predictions[mask] - preds_mod_sel
            # water_bodies: NDWI is not a model feature → heuristic term only
            delta_sel += i_wb * open_frac * ML_WATER_BODIES_MAX_DELTA_C

            mod_feats = {"ndvi": ndvi_mod, "ndbi": ndbi_mod, "albedo": albedo_mod}

            # Cost uses the same areas as the heuristic engine
            for int_name, intensity in plan.items():
                intensity = float(intensity)
                if intensity <= 0:
                    continue
                interv = self.interventions[int_name]
                if interv.applies_to == "buildings":
                    ar = self._bldg[mask]
                elif interv.applies_to == "roads":
                    ar = self._road_area[mask]
                elif interv.applies_to == "open":
                    ar = self._open[mask]
                else:
                    ar = area_sel
                cost += intensity * float(np.sum(ar)) * interv.cost_per_m2

        # ---- embed into full-length arrays ----------------------------
        delta_full = np.zeros(n_total, dtype=float)
        delta_full[mask] = delta_sel

        baseline_full = self.baseline_predictions.copy() \
            if self.baseline_predictions is not None else self._base_temp.copy()
        preds_full = baseline_full - delta_full

        total_cooling = float(np.sum(delta_full))
        mean_cooling = float(np.mean(delta_full))
        max_cooling = float(np.max(delta_full))
        pop_sum = float(np.sum(self._pop))
        pop_weighted = (
            float(np.sum(delta_full * self._pop) / pop_sum) if pop_sum > 0 else mean_cooling
        )

        return {
            "delta_per_cell": delta_full,
            "delta_masked": delta_sel,
            "mask": mask,
            "predictions": preds_full,
            "baseline_predictions": baseline_full,
            "predictions_modified": preds_mod_sel,
            "modified_features": mod_feats,
            "total_cooling": total_cooling,
            "mean_cooling": mean_cooling,
            "max_cooling": max_cooling,
            "pop_weighted_cooling": pop_weighted,
            "total_cost_inr": float(cost),
            "cooling_per_lakh_inr": total_cooling / (cost / 1e5 + 1e-9),
        }

    # ------------------------------------------------------------------
    # Grid-building path (public API preserved)
    # ------------------------------------------------------------------

    def apply_interventions(
        self,
        plan: Dict[str, float],
        target_cells: Optional[List[str]] = None,
    ) -> UrbanGrid:
        """Apply interventions and return a NEW grid.

        Touched cells get features["temp_reduction"] (°C, positive =
        cooling). In ML mode they also get the modified physical
        features ("ndvi_modified", "ndbi_modified", "albedo_modified").
        """
        fast = self.evaluate_plan_metrics(plan, target_cells)
        return self._build_modified_grid(plan, fast)

    def _build_modified_grid(self, plan: Dict[str, float], fast: Dict[str, Any]) -> UrbanGrid:
        new_grid = self.baseline_grid.copy()
        mask = fast["mask"]
        delta_sel = fast["delta_masked"]
        mod_feats = fast["modified_features"]
        sel_idx = np.where(mask)[0]

        for j, i in enumerate(sel_idx):
            cell = new_grid.cells[self._cell_ids[i]]
            cell.features["temp_reduction"] = float(delta_sel[j])
            if mod_feats is not None:
                cell.features["ndvi_modified"] = float(mod_feats["ndvi"][j])
                cell.features["ndbi_modified"] = float(mod_feats["ndbi"][j])
                cell.features["albedo_modified"] = float(mod_feats["albedo"][j])
        return new_grid

    def estimate_impact(
        self,
        plan: Dict[str, float],
        target_cells: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Full impact report for a plan (public interface — keys preserved).

        Returns a dict containing AT LEAST (interface contract):
            plan, total_cooling, mean_cooling, pop_weighted_cooling,
            total_cost_inr, cooling_per_lakh_inr, delta_per_cell,
            modified_grid
        plus: max_cooling, predictions, baseline_predictions (legacy
        extras), and the newer keys mode, baseline_mean_temp_c,
        predictions_ml.

        UNITS: all temperatures/deltas are in °C (unit-corrected).
        """
        fast = self.evaluate_plan_metrics(plan, target_cells)
        modified_grid = self._build_modified_grid(plan, fast)

        preds_ml = fast["predictions_modified"]
        return {
            "plan": dict(plan),
            "mode": self.mode,
            "modified_grid": modified_grid,
            "predictions": fast["predictions"],
            "baseline_predictions": fast["baseline_predictions"],
            "predictions_ml": (
                preds_ml if preds_ml is not None else None
            ),
            "baseline_mean_temp_c": float(np.mean(fast["baseline_predictions"])),
            "delta_per_cell": fast["delta_per_cell"],
            "total_cooling": fast["total_cooling"],
            "mean_cooling": fast["mean_cooling"],
            "max_cooling": fast["max_cooling"],
            "pop_weighted_cooling": fast["pop_weighted_cooling"],
            "total_cost_inr": fast["total_cost_inr"],
            "cooling_per_lakh_inr": fast["cooling_per_lakh_inr"],
        }


# ----------------------------------------------------------------------
# 4. Real data loader
# ----------------------------------------------------------------------

def load_ward1_grid(
    csv_path: str | Path = "ward1_processed.csv",
    ward_name: str = "Ward_1",
) -> UrbanGrid:
    """Load the real processed Ward-1 grid.

    UNIT FIX: target_temp in the CSV is a raw Landsat ST_B10 DN
    (~47,000). It is converted to °C here using the ml-modeling
    conversion and stored as baseline_heat_stress / features["target_temp"].
    The original DN is preserved as features["target_temp_dn"] for
    reference ONLY — never use it in temperature math.
    """
    df = pd.read_csv(csv_path)

    required = ["grid_id", "target_temp", "bldg_area_sqm", "road_length_m"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    temp_c_values = dn_to_celsius(df["target_temp"].to_numpy())

    cells: List[GridCell] = []
    for pos, (_, row) in enumerate(df.iterrows()):
        cell_id = f"{ward_name}_C{int(row['grid_id']):04d}"

        bldg = float(row["bldg_area_sqm"])
        road = float(row["road_length_m"])
        temp_c = float(temp_c_values[pos])
        temp_dn = float(row["target_temp"])

        # ASSUMED population proxy: higher building area → more residents
        pop = max(0.0, bldg * POPULATION_PROXY_PER_SQM_BUILDING)

        features = {
            "bldg_area_sqm": bldg,
            "road_length_m": road,
            "open_space_sqm": max(0.0, DEFAULT_CELL_AREA_M2 - bldg - road * ROAD_WIDTH_M),
            "target_temp": temp_c,       # °C (unit-corrected)
            "target_temp_dn": temp_dn,   # raw DN — reference only
        }
        if "ndvi" in row:
            features["ndvi"] = float(row["ndvi"])
        if "ndbi" in row:
            features["ndbi"] = float(row["ndbi"])
        if "albedo" in row:
            features["albedo"] = float(row["albedo"])

        cells.append(
            GridCell(
                cell_id=cell_id,
                ward=ward_name,
                features=features,
                population=pop,
                area_m2=DEFAULT_CELL_AREA_M2,
                baseline_heat_stress=temp_c,
            )
        )

    return UrbanGrid(cells)


# ----------------------------------------------------------------------
# 5. ML-mode convenience factory (graceful fallback)
# ----------------------------------------------------------------------

def create_ml_simulator(
    grid: UrbanGrid,
    csv_path: str = "ward1_processed.csv",
    refresh_cache: bool = False,
) -> tuple:
    """Build an ML-mode simulator wired to the local model adapter.

    Falls back to a heuristic-mode simulator (with a warning) whenever
    XGBoost/scikit-learn is unavailable or training fails, so callers
    never crash just because the ML stack is missing.

    Returns
    -------
    (simulator, info)
        info["mode"] is "ml" or "heuristic"; info["reason"] explains a
        fallback; info["model_info"] carries adapter details in ML mode.
    """
    info: Dict[str, Any] = {"mode": "heuristic", "reason": None, "model_info": None}
    try:
        from model_adapter import build_predict_fn, FEATURE_ORDER

        predict_fn, model_info = build_predict_fn(
            csv_path=csv_path, refresh_cache=refresh_cache
        )
        if predict_fn is None:
            info["reason"] = model_info.get("message", "predictor unavailable")
            warnings.warn(
                f"ML mode unavailable ({info['reason']}) — falling back to "
                f"heuristic simulation.",
                RuntimeWarning,
                stacklevel=2,
            )
            return InterventionSimulator(grid, mode="heuristic"), info

        sim = InterventionSimulator(
            grid,
            model_predict_fn=predict_fn,
            feature_order=list(FEATURE_ORDER),
            mode="ml",
        )
        info["mode"] = "ml"
        info["model_info"] = model_info
        return sim, info

    except Exception as exc:  # pragma: no cover — defensive fallback
        info["reason"] = f"{type(exc).__name__}: {exc}"
        warnings.warn(
            f"ML mode failed ({info['reason']}) — falling back to heuristic.",
            RuntimeWarning,
            stacklevel=2,
        )
        return InterventionSimulator(grid, mode="heuristic"), info


# ----------------------------------------------------------------------
# 6. Lightweight self-tests
# ----------------------------------------------------------------------

def _run_self_tests() -> bool:
    print("=== UrbanHeat AI – Intervention Simulator self-tests ===\n")

    csv_path = "ward1_processed.csv"
    grid = load_ward1_grid(csv_path)
    print(f"Loaded grid: {len(grid.cells)} cells")

    base = np.array([c.baseline_heat_stress for c in grid.cells.values()])
    checks: List[tuple] = []

    checks.append((
        "baseline temps are plausible °C (0–60), not raw DN (~47,000)",
        bool(np.all(base > 0) and np.all(base < 60)),
    ))
    print(f"Baseline °C min/mean/max: {base.min():.2f} / {base.mean():.2f} / {base.max():.2f}")

    sim = InterventionSimulator(grid)
    plan = {"cool_roofs": 0.7, "green_cover": 0.4, "albedo_boost": 0.3}
    res = sim.estimate_impact(plan)

    required_keys = [
        "plan", "total_cooling", "mean_cooling", "pop_weighted_cooling",
        "total_cost_inr", "cooling_per_lakh_inr", "delta_per_cell",
        "modified_grid",
    ]
    checks.append(("all public interface keys present",
                   all(k in res for k in required_keys)))
    checks.append(("costs positive", res["total_cost_inr"] > 0))
    checks.append(("cooling deltas finite",
                   bool(np.all(np.isfinite(res["delta_per_cell"])))))

    # Monotonicity: higher intensity should generally cool more
    means = []
    for level in (0.25, 0.5, 1.0):
        r = sim.evaluate_plan_metrics({"cool_roofs": level})
        means.append(r["mean_cooling"])
    monotone = means[0] < means[1] < means[2]
    checks.append(("mean cooling increases with intensity "
                   f"({means[0]:.3f} → {means[1]:.3f} → {means[2]:.3f} °C)",
                   monotone))

    # Targeted subset behaves like a subset
    hot = sim.select_hottest_cells(0.1)
    r_hot = sim.evaluate_plan_metrics(plan, target_cells=hot)
    checks.append(("targeted hotspot run cools fewer cells than full run",
                   float(np.count_nonzero(r_hot["delta_per_cell"])) <= len(hot) + 1e-9))

    # ML mode (only if the ML stack is available)
    try:
        ml_sim, ml_info = create_ml_simulator(grid, csv_path=csv_path)
        if ml_info["mode"] == "ml":
            r_ml = ml_sim.estimate_impact(plan)
            checks.append(("ML mode runs and returns finite deltas",
                           bool(np.all(np.isfinite(r_ml["delta_per_cell"])))))
            checks.append(("ML mode reports plausible °C predictions",
                           bool(np.all(r_ml["predictions"] > 0)
                                and np.all(r_ml["predictions"] < 60))))
            print(f"ML mode: ACTIVE ({ml_info['model_info']['source']})")
        else:
            print(f"ML mode: SKIPPED — {ml_info['reason']}")
    except Exception as exc:  # pragma: no cover
        print(f"ML mode: SKIPPED — unexpected error: {exc}")

    print()
    all_ok = True
    for name, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        all_ok &= ok
    print("\nRESULT:", "PASS" if all_ok else "FAIL")
    return all_ok


if __name__ == "__main__":
    ok = _run_self_tests()
    raise SystemExit(0 if ok else 1)