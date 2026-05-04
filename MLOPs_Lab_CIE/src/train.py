import os
import json
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import mlflow


# -----------------------------
# 1. Setup MLflow
# -----------------------------
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("chemreact-reaction-yield-pct")


# -----------------------------
# 2. Load Data
# -----------------------------
data_path = os.path.join("data", "training_data.csv")
df = pd.read_csv(data_path)

X = df.drop("reaction_yield_pct", axis=1)
y = df["reaction_yield_pct"]


# -----------------------------
# 3. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -----------------------------
# 4. Define Models
# -----------------------------
models = {
    "Lasso": Lasso(alpha=0.1, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
}

results = []
trained_models = {}


# -----------------------------
# 5. Train + Log Each Model
# -----------------------------
for name, model in models.items():

    with mlflow.start_run(run_name=name):

        # Tag
        mlflow.set_tag("experiment_type", "baseline_comparison")

        # Log params
        mlflow.log_params(model.get_params())

        # Train
        model.fit(X_train, y_train)

        # Store trained model
        trained_models[name] = model

        # Predict
        preds = model.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        # Log metrics
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)

        # Save results
        results.append({
            "name": name,
            "mae": float(mae),
            "rmse": float(rmse)
        })


# -----------------------------
# 6. Select Best Model
# -----------------------------
best_model_info = min(results, key=lambda x: x["mae"])
best_model_name = best_model_info["name"]
best_model_object = trained_models[best_model_name]


# -----------------------------
# 7. Save Best Model
# -----------------------------
os.makedirs("models", exist_ok=True)

model_filename = f"{best_model_name.lower()}_model.pkl"
model_path = os.path.join("models", model_filename)

joblib.dump(best_model_object, model_path)


# -----------------------------
# 8. Save JSON Output
# -----------------------------
output = {
    "experiment_name": "chemreact-reaction-yield-pct",
    "models": results,
    "best_model": best_model_name,
    "best_metric_name": "mae",
    "best_metric_value": best_model_info["mae"]
}

os.makedirs("results", exist_ok=True)
output_path = os.path.join("results", "step1_s1.json")

with open(output_path, "w") as f:
    json.dump(output, f, indent=4)


# -----------------------------
# 9. Done
# -----------------------------
print("Task 1 completed.")
print("Best model saved at:", model_path)
print("Results saved at:", output_path)