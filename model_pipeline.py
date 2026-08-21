import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import shap
import matplotlib.pyplot as plt
import os

print("Starting Phase 2 ML Pipeline (Random Split)...")

if not os.path.exists("ward1_processed.csv"):
    print("ERROR: Could not find 'ward1_processed.csv'.")
    exit()

df = pd.read_csv("ward1_processed.csv")

# TASK 1: Decode the Temperature Data
df['target_temp_c'] = (df['target_temp'] * 0.00341802) + 149.0 - 273.15

# TASK 2: Standard Train/Test Split
features = ['bldg_area_sqm', 'road_length_m', 'ndvi', 'ndbi', 'albedo']

X = df[features]
y = df['target_temp_c']

# We are using a random shuffle to ensure the model sees all types of terrain
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TASK 3: Train the XGBoost Model
print("Training upgraded XGBoost engine...")
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print(f"\n--- Model Results (Standard Split) ---")
print(f"RMSE: {rmse:.2f} °C")
print(f"R-squared: {r2:.2f}")

# TASK 4: SHAP Explainability
print("\nRunning SHAP explainability...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

plt.figure(figsize=(8, 6))
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("shap_summary.png", bbox_inches='tight')
print("SUCCESS! Updated SHAP plot saved locally as 'shap_summary.png'")