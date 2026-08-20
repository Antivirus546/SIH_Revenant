import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import shap
import matplotlib.pyplot as plt
import os

print("Starting SIH_Revenant ML Pipeline...")

# Check if file exists
if not os.path.exists("ward1_processed.csv"):
    print("ERROR: Could not find 'ward1_processed.csv'. Make sure it is in the same folder as this script!")
    exit()

print("Loading ward1_processed.csv...")
df = pd.read_csv("ward1_processed.csv")

# TASK 1: Decode the Temperature Data (Landsat 8 Scaling)
# Formula: (Raw * 0.00341802) + 149.0 - 273.15
df['target_temp_c'] = (df['target_temp'] * 0.00341802) + 149.0 - 273.15

# TASK 2: Stop Cheating the Validation (Spatial Split)
# Instead of random splitting, we sort by the geographical grid_id 
# to keep neighboring cells together.
df = df.sort_values('grid_id')
split_index = int(len(df) * 0.8)

train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

# Define X and y using the new Celsius target
X_train = train_df[['bldg_area_sqm', 'road_length_m']]
y_train = train_df['target_temp_c']

X_test = test_df[['bldg_area_sqm', 'road_length_m']]
y_test = test_df['target_temp_c']

# TASK 3: Train the XGBoost Model
print("Training XGBoost model...")
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print(f"\n--- Model Results (Spatial Split) ---")
print(f"RMSE: {rmse:.2f} °C")
print(f"R-squared: {r2:.2f}")

# TASK 4: Wire up SHAP Explainability
print("\nRunning SHAP explainability...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Generate and save the Summary Plot
plt.figure(figsize=(8, 6))
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("shap_summary.png", bbox_inches='tight')
print("SUCCESS! SHAP summary plot saved locally as 'shap_summary.png'")