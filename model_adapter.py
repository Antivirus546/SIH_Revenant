"""
model_adapter.py — Local ML model adapter for the UrbanHeat AI simulator.

Team Revenant | SIH 2026 | IHSIH031

WHAT THIS IS
------------
A local adapter that REPRODUCES the existing machine-learning model
specification defined in ``ml-modeling/model_pipeline.py`` WITHOUT
modifying that file.

WHAT THIS IS NOT
----------------
This is NOT a loader for an "original trained model". The existing
ml-modeling pipeline does not currently persist any trained artifact
(it never calls joblib.dump / pickle.dump), so there is nothing to
load. Instead, this adapter LOCALLY RETRAINS a model that reproduces
the existing specification exactly:

    features (exact order, from ml-modeling/model_pipeline.py):
        ["bldg_area_sqm", "road_length_m", "ndvi", "ndbi", "albedo"]

    estimator:
        XGBRegressor(n_estimators=100, learning_rate=0.1,
                     max_depth=5, random_state=42)

    target:
        Landsat-8 Collection 2 Level-2 ST_B10 surface temperature
        converted to degrees Celsius using the SAME conversion as
        ml-modeling/model_pipeline.py:

            celsius = DN * 0.00341802 + 149.0 - 273.15

    training split:
        train_test_split(test_size=0.2, random_state=42) — identical to
        the existing pipeline (the model is fit on the 80% training
        fold, exactly like the original script).

A trained adapter model is cached locally (pickle) so repeated runs do
not retrain. The cache records the feature order and hyperparameters it
was built with and is automatically invalidated if the specification
changes.

If XGBoost / scikit-learn is unavailable, ``build_predict_fn`` returns
 ``(None, info)`` and callers should fall back to the heuristic
simulator instead of crashing (see intervention_simulater.py).

SCIENTIFIC STATUS
-----------------
OBSERVED     : satellite-derived input features, observed LST target.
ASSUMED      : everything built on top of the model (see the ASSUMPTIONS
               section in intervention_simulater.py).
The model captures statistical associations from ONE ward / ONE season
snapshot. It is a decision-support MVP component, NOT a validated
physical urban microclimate model.
"""

from __future__ import annotations

import os
import pickle
import sys
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Model specification (mirrors ml-modeling/model_pipeline.py — DO NOT alter)
# ----------------------------------------------------------------------

FEATURE_ORDER = [
    "bldg_area_sqm",
    "road_length_m",
    "ndvi",
    "ndbi",
    "albedo",
]

XGB_PARAMS: Dict[str, Any] = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 5,
    "random_state": 42,
}

# Landsat-8 Collection 2 Level-2 ST_B10 scaling (identical to ml-modeling)
LST_SCALE = 0.00341802
LST_OFFSET_KELVIN = 149.0
KELVIN_TO_CELSIUS = 273.15

DEFAULT_CACHE_FILENAME = "heat_stress_model.pkl"
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE_PATH = os.path.join(_MODULE_DIR, DEFAULT_CACHE_FILENAME)


def dn_to_celsius(dn) -> np.ndarray:
    """Convert raw Landsat ST_B10 DN values to degrees Celsius.

    Same conversion as ml-modeling/model_pipeline.py:
        celsius = DN * 0.00341802 + 149.0 - 273.15
    """
    return np.asarray(dn, dtype=float) * LST_SCALE + LST_OFFSET_KELVIN - KELVIN_TO_CELSIUS


# ----------------------------------------------------------------------
# Training / caching
# ----------------------------------------------------------------------

def _new_xgb_model():
    """Instantiate the specified XGBRegressor, or None if xgboost is absent."""
    try:
        from xgboost import XGBRegressor
    except ImportError:
        return None
    return XGBRegressor(**XGB_PARAMS)


def train_model(csv_path: str = "ward1_processed.csv"):
    """Locally retrain a model reproducing the ml-modeling specification.

    Uses the identical feature order, °C target conversion, 80/20 split
    (random_state=42) and XGBoost hyperparameters as
    ml-modeling/model_pipeline.py.

    Raises
    ------
    RuntimeError   if xgboost or scikit-learn is unavailable.
    FileNotFoundError if the dataset is missing.
    ValueError     if the dataset lacks expected columns.
    """
    model = _new_xgb_model()
    if model is None:
        raise RuntimeError(
            "xgboost is not installed — cannot train the ML adapter. "
            "Callers should fall back to the heuristic simulator."
        )
    try:
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is not installed — cannot train the ML adapter."
        ) from exc

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [c for c in FEATURE_ORDER + ["target_temp"] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    X = df[FEATURE_ORDER].to_numpy(dtype=float)
    y = dn_to_celsius(df["target_temp"].to_numpy())

    # Identical split to ml-modeling/model_pipeline.py
    X_train, _X_test, y_train, _y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)
    return model


def save_model(model, path: str = DEFAULT_CACHE_PATH) -> None:
    """Persist the adapter model together with its specification fingerprint."""
    payload = {
        "kind": "urbanheat_model_adapter",
        "feature_order": list(FEATURE_ORDER),
        "xgb_params": dict(XGB_PARAMS),
        "lst_scale": LST_SCALE,
        "lst_offset_kelvin": LST_OFFSET_KELVIN,
        "kelvin_to_celsius": KELVIN_TO_CELSIUS,
        "model": model,
    }
    with open(path, "wb") as fh:
        pickle.dump(payload, fh)


def load_cached_model(path: str = DEFAULT_CACHE_PATH):
    """Return a cached adapter model, or None.

    The cache is rejected (returns None) if it does not exist, cannot be
    read, or was produced with a different feature order / hyperparameter
    specification.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
    except Exception:
        return None
    if not isinstance(payload, dict) or payload.get("kind") != "urbanheat_model_adapter":
        return None
    if list(payload.get("feature_order", [])) != list(FEATURE_ORDER):
        return None
    if payload.get("xgb_params") != XGB_PARAMS:
        return None
    return payload.get("model")


def get_model(
    csv_path: str = "ward1_processed.csv",
    cache_path: str = DEFAULT_CACHE_PATH,
    refresh_cache: bool = False,
):
    """Load the cached adapter model, or (re)train and cache it."""
    if not refresh_cache:
        cached = load_cached_model(cache_path)
        if cached is not None:
            return cached
    model = train_model(csv_path)
    try:
        save_model(model, cache_path)
    except OSError:
        pass  # caching is best-effort; training result is still returned
    return model


# ----------------------------------------------------------------------
# Predictor factory used by the simulator
# ----------------------------------------------------------------------

def build_predict_fn(
    csv_path: str = "ward1_processed.csv",
    cache_path: str = DEFAULT_CACHE_PATH,
    refresh_cache: bool = False,
) -> Tuple[Optional[Callable[[np.ndarray], np.ndarray]], Dict[str, Any]]:
    """Build a predict-in-°C callable from the locally retrained model.

    Returns
    -------
    (predict_fn, info)
        predict_fn : callable mapping X of shape (n, 5) to temperatures
                     in °C, or None if XGBoost / scikit-learn is
                     unavailable or training failed.
        info       : dict describing what happened ("source", "message",
                     "cache_path", ...). Callers MUST fall back to the
                     heuristic simulator when predict_fn is None.
    """
    try:
        model = get_model(
            csv_path=csv_path, cache_path=cache_path, refresh_cache=refresh_cache
        )
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        return None, {"source": "unavailable", "message": str(exc)}

    def predict_fn(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.asarray(model.predict(X), dtype=float)

    used_cache = load_cached_model(cache_path) is not None
    return predict_fn, {
        "source": "cache" if used_cache else "fresh-training",
        "cache_path": cache_path,
        "feature_order": list(FEATURE_ORDER),
        "message": (
            "Locally retrained model reproducing the existing "
            "ml-modeling specification (no original artifact exists)."
        ),
    }


# ----------------------------------------------------------------------
# Lightweight self-test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("=== model_adapter self-test ===")
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "ward1_processed.csv"

    predict_fn, info = build_predict_fn(csv_path=csv_path, refresh_cache=True)
    if predict_fn is None:
        print(f"ML adapter unavailable: {info.get('message')}")
        print("RESULT: FAIL (cannot verify ML mode on this machine)")
        sys.exit(1)

    print(f"Model source : {info['source']}")
    print(f"Features     : {info['feature_order']}")

    df = pd.read_csv(csv_path)
    X = df[FEATURE_ORDER].to_numpy(dtype=float)
    actual_c = dn_to_celsius(df["target_temp"].to_numpy())
    pred_c = predict_fn(X)

    checks = {
        "predictions finite": bool(np.all(np.isfinite(pred_c))),
        "predictions plausible (0–60 °C)": bool(
            np.all(pred_c > 0) and np.all(pred_c < 60)
        ),
        "actual LST plausible (0–60 °C)": bool(
            np.all(actual_c > 0) and np.all(actual_c < 60)
        ),
    }
    print(f"Predicted °C  min/mean/max: {pred_c.min():.2f} / {pred_c.mean():.2f} / {pred_c.max():.2f}")
    print(f"Actual °C     min/mean/max: {actual_c.min():.2f} / {actual_c.mean():.2f} / {actual_c.max():.2f}")
    rmse = float(np.sqrt(np.mean((pred_c - actual_c) ** 2)))
    print(f"In-sample RMSE vs observed LST: {rmse:.2f} °C")
    for name, ok in checks.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print("RESULT:", "PASS" if all(checks.values()) else "FAIL")