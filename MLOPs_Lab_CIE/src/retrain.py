import os
import json
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso

import mlflow


# -----------------------------
# 1. Setup MLflow
# -----------------------------
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("chemreact-reaction-yield-pct")


# -----------------------------
# 2. Load Data
# -----------------------------
train_df = pd.read_csv(os.path.join("data", "training_data.csv"))
new_df = pd.read_csv(os.path.join("data", "new_data.csv"))

combined_df = pd.concat([train_df, new_df], ignore_index=True)


# -----------------------------
# 3. Get Best Model Type (Task 1)
# -----------------------------
with open(os.path.join("results", "step1_s1.json"), "r") as f:
    step1 = json.load(f)

best_model_name = step1["best_model"]


# -----------------------------
# 4. Prepare SAME Test Set
# -----------------------------
X = train_df.drop("reaction_yield_pct", axis=1)
y = train_df["reaction_yield_pct"]

X_train_old, X_test, y_train_old, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Combined training data (NO split again)
X_combined = combined_df.drop("reaction_yield_pct", axis=1)
y_combined = combined_df["reaction_yield_pct"]


# -----------------------------
# 5. Initialize Models
# -----------------------------
def get_model():
    if best_model_name == "GradientBoosting":
        return GradientBoostingRegressor(random_state=42)
    elif best_model_name == "Lasso":
        return Lasso(random_state=42)
    else:
        raise Exception("Unsupported model")


# Champion (old data)
champion_model = get_model()
champion_model.fit(X_train_old, y_train_old)

# Retrained (combined data)
retrained_model = get_model()
retrained_model.fit(X_combined, y_combined)


# -----------------------------
# 6. Evaluate BOTH on SAME test set
# -----------------------------
champion_preds = champion_model.predict(X_test)
retrained_preds = retrained_model.predict(X_test)

champion_mae = mean_absolute_error(y_test, champion_preds)
retrained_mae = mean_absolute_error(y_test, retrained_preds)

improvement = champion_mae - retrained_mae


# -----------------------------
# 7. Promotion Decision
# -----------------------------
threshold = 1.0

if improvement >= threshold:
    action = "promoted"
else:
    action = "kept_champion"


# -----------------------------
# 8. MLflow Logging
# -----------------------------
with mlflow.start_run(run_name="retraining-pipeline"):

    mlflow.log_metric("champion_mae", champion_mae)
    mlflow.log_metric("retrained_mae", retrained_mae)
    mlflow.log_metric("improvement", improvement)

    mlflow.log_param("model_type", best_model_name)
    mlflow.log_param("threshold", threshold)
    mlflow.log_param("action", action)


# -----------------------------
# 9. Save JSON Output
# -----------------------------
output = {
    "original_data_rows": int(len(train_df)),
    "new_data_rows": int(len(new_df)),
    "combined_data_rows": int(len(combined_df)),
    "champion_mae": float(champion_mae),
    "retrained_mae": float(retrained_mae),
    "improvement": float(improvement),
    "min_improvement_threshold": threshold,
    "action": action,
    "comparison_metric": "mae"
}

output_path = os.path.join("results", "step4_s8.json")

with open(output_path, "w") as f:
    json.dump(output, f, indent=4)


print("Task 4 completed.")
print("Champion MAE:", champion_mae)
print("Retrained MAE:", retrained_mae)
print("Improvement:", improvement)
print("Action:", action)
print("Results saved to:", output_path)