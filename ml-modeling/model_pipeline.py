import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import shap
import matplotlib.pyplot as plt
import os
import joblib

print("Starting Phase 2 ML Pipeline (Spatial Block Split with Tuning)...")

if not os.path.exists("ward1_processed.csv"):
    print("ERROR: Could not find 'ward1_processed.csv'.")
    exit()

df = pd.read_csv("ward1_processed.csv")

# TASK 1: Decode the Temperature Data
df['target_temp_c'] = (df['target_temp'] * 0.00341802) + 149.0 - 273.15

# TASK 2: Define Spatial Groups and Features
# NOTE TO ADAPTER TEAM: ndwi and elevation have been appended.
features = ['bldg_area_sqm', 'road_length_m', 'ndvi', 'ndbi', 'albedo', 'ndwi', 'elevation']
groups = df['grid_id'] 

X = df[features]
y = df['target_temp_c']

# Use a spatial block split (GroupShuffleSplit) instead of random split to prevent spatial leakage
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
groups_train = groups.iloc[train_idx]

# TASK 3: Hyperparameter Tuning with Spatial Cross-Validation
print("\nInitializing hyperparameter tuning...")

param_grid = {
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 300, 500],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
}
# Direction each feature is physically expected to push temperature:
# +1 = more of this → hotter, -1 = more of this → cooler, 0 = no constraint
# (bldg_area_sqm, road_length_m, ndvi, ndbi, albedo, ndwi, elevation)
monotone_constraints = (1, 0, -1, 1, -1, -1, 0)

base_model = XGBRegressor(random_state=42, monotone_constraints=monotone_constraints)

# Use GroupKFold for internal cross-validation during tuning
gkf = GroupKFold(n_splits=3)

random_search = RandomizedSearchCV(
    estimator=base_model, 
    param_distributions=param_grid, 
    n_iter=20,          
    scoring='neg_root_mean_squared_error', 
    cv=gkf,               
    verbose=1, 
    random_state=42,
    n_jobs=-1           
)

print("Training and tuning upgraded XGBoost engine...")
# Crucial: pass the spatial groups to fit()
random_search.fit(X_train, y_train, groups=groups_train)

# Extract the winning model
best_model = random_search.best_estimator_
print(f"\nBest parameters found: {random_search.best_params_}")

# Evaluate using the spatial test set
preds = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print(f"\n--- Model Results (Honest Spatial Split) ---")
print(f"RMSE: {rmse:.2f} °C")
print(f"R-squared: {r2:.2f}")

# TASK 4: SHAP Explainability
print("\nRunning SHAP explainability...")
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

plt.figure(figsize=(8, 6))
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("shap_summary.png", bbox_inches='tight')
print("SUCCESS! Updated SHAP plot saved locally as 'shap_summary.png'")

# TASK 5: Save the Optimized Model
joblib.dump(best_model, "heat_model.pkl")
print("Optimized model saved as heat_model.pkl")