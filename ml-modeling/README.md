# ML Modeling — UrbanHeat AI

This module contains the machine learning pipeline used to predict urban heat stress from processed geospatial and environmental data.

## Overview

The ML pipeline uses **XGBoost** to model the relationship between urban and environmental features and surface temperature.

The pipeline includes:

- Data preprocessing
- Feature preparation
- Spatial train-test splitting
- XGBoost model training
- Hyperparameter optimization
- Model evaluation
- SHAP-based explainability
- Model serialization

## Features

The model uses the following seven features:

1. `bldg_area_sqm`
2. `road_length_m`
3. `ndvi`
4. `ndbi`
5. `albedo`
6. `ndwi`
7. `elevation`

### Feature Ranges

| Feature | Type | Range / Description |
|---|---|---|
| `bldg_area_sqm` | Float | Building area in square meters |
| `road_length_m` | Float | Road length in meters |
| `ndvi` | Float | -1.0 to 1.0 |
| `ndbi` | Float | -1.0 to 1.0 |
| `albedo` | Float | 0.0 to 1.0 |
| `ndwi` | Float | -1.0 to 1.0 |
| `elevation` | Float | Elevation value |

The features must be provided to the trained model in exactly this order.

## Model

The primary prediction model is **XGBoost**.

Hyperparameters are optimized using:

- `RandomizedSearchCV`
- `GroupKFold`
- Spatial grouping based on `grid_id`

This approach helps prevent spatial data leakage during model evaluation.

## Spatial Evaluation

The dataset is divided using `GroupShuffleSplit` with `grid_id` as the grouping variable.

This ensures that spatial observations are separated by grid during evaluation, resulting in a more reliable estimate of model performance.

## Explainability

The pipeline uses **SHAP (SHapley Additive exPlanations)** to analyze feature contributions.

The generated `shap_summary.png` provides a visual representation of feature importance and their impact on model predictions.

## Files

```text
ml-modeling/
├── model_pipeline.py
├── heat_model.pkl
├── shap_summary.png
├── ward1_processed.csv
└── README.md
```

### `model_pipeline.py`

Main script responsible for preprocessing, training, evaluation, hyperparameter optimization, and model export.

### `heat_model.pkl`

Serialized trained XGBoost model.

### `shap_summary.png`

SHAP summary plot showing feature importance and impact.

### `ward1_processed.csv`

Processed dataset used for model training and evaluation.

## Usage

Ensure the virtual environment is activated and `ward1_processed.csv` is available in the project directory.

Run:

```bash
python model_pipeline.py
```

The pipeline will train and evaluate the XGBoost model and generate the required model artifacts.

## Output

The main outputs are:

- **`heat_model.pkl`** — trained XGBoost model
- **`shap_summary.png`** — SHAP feature-importance visualization
