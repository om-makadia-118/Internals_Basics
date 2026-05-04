import os
import json
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error

import mlflow


# -----------------------------
# 1. Setup MLflow
# -----------------------------
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("chemreact-reaction-yield-pct")


# -----------------------------
# 2. Load Task 1 Results
# -----------------------------
with open(os.path.join("results", "step1_s1.json"), "r") as f:
    step1_results = json.load(f)

best_model_name = step1_results["best_model"]


# -----------------------------
# 3. Load Data
# -----------------------------
df = pd.read_csv(os.path.join("data", "training_data.csv"))

X = df.drop("reaction_yield_pct", axis=1)
y = df["reaction_yield_pct"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -----------------------------
# 4. Define Param Grid Dynamically
# -----------------------------
all_params = []

if best_model_name == "GradientBoosting":
    param_grid = {
        "n_estimators": [50, 150],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [3, 5, 10]
    }

    for n in param_grid["n_estimators"]:
        for lr in param_grid["learning_rate"]:
            for md in param_grid["max_depth"]:
                all_params.append({
                    "n_estimators": n,
                    "learning_rate": lr,
                    "max_depth": md
                })

elif best_model_name == "Lasso":
    param_grid = {
        "alpha": [0.001, 0.01, 0.1, 1, 10]
    }

    for a in param_grid["alpha"]:
        all_params.append({
            "alpha": a
        })

else:
    raise Exception("Unsupported model type")


# Random search shuffle
random.seed(42)
random.shuffle(all_params)


# -----------------------------
# 5. Cross Validation
# -----------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)


# -----------------------------
# 6. MLflow Parent Run
# -----------------------------
best_cv_mae = float("inf")
best_params = None
trial_count = 0

with mlflow.start_run(run_name="tuning-chemreact"):

    for params in all_params:
        trial_count += 1

        with mlflow.start_run(nested=True):

            # Model selection
            if best_model_name == "GradientBoosting":
                model = GradientBoostingRegressor(
                    **params,
                    random_state=42
                )
            else:
                model = Lasso(
                    **params,
                    random_state=42
                )

            # CV MAE
            cv_scores = cross_val_score(
                model,
                X_train,
                y_train,
                scoring="neg_mean_absolute_error",
                cv=kf
            )

            cv_mae = -np.mean(cv_scores)

            # Log
            mlflow.log_params(params)
            mlflow.log_metric("cv_mae", cv_mae)

            # Best tracking
            if cv_mae < best_cv_mae:
                best_cv_mae = cv_mae
                best_params = params


    # -----------------------------
    # 7. Train Best Model
    # -----------------------------
    if best_model_name == "GradientBoosting":
        best_model = GradientBoostingRegressor(
            **best_params,
            random_state=42
        )
    else:
        best_model = Lasso(
            **best_params,
            random_state=42
        )

    best_model.fit(X_train, y_train)

    preds = best_model.predict(X_test)
    test_mae = mean_absolute_error(y_test, preds)

    # Log best
    mlflow.log_params(best_params)
    mlflow.log_metric("best_cv_mae", best_cv_mae)
    mlflow.log_metric("test_mae", test_mae)


# -----------------------------
# 8. Save JSON
# -----------------------------
output = {
    "search_type": "random",
    "n_folds": 5,
    "total_trials": trial_count,
    "best_params": best_params,
    "best_mae": float(test_mae),
    "best_cv_mae": float(best_cv_mae),
    "parent_run_name": "tuning-chemreact"
}

output_path = os.path.join("results", "step2_s2.json")

with open(output_path, "w") as f:
    json.dump(output, f, indent=4)


print("Task 2 completed.")
print("Best Model:", best_model_name)
print("Best Params:", best_params)
print("Results saved to:", output_path)