import os
import json
import joblib
import mlflow
import mlflow.sklearn

from mlflow.tracking import MlflowClient


# -----------------------------
# 1. Setup MLflow
# -----------------------------
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("chemreact-reaction-yield-pct")

MODEL_NAME = "chemreact-reaction-yield-pct-predictor"


# -----------------------------
# 2. Load Tuned Model
# -----------------------------
model_path = os.path.join("models", "tuned_model.pkl")

if not os.path.exists(model_path):
    raise Exception("Tuned model not found. Run tune.py first.")

model = joblib.load(model_path)


# -----------------------------
# 3. Load Task 2 Metrics
# -----------------------------
with open(os.path.join("results", "step2_s2.json"), "r") as f:
    step2_results = json.load(f)

best_mae = step2_results["best_mae"]


# -----------------------------
# 4. Start MLflow Run
# -----------------------------
with mlflow.start_run(run_name="model-registration") as run:

    run_id = run.info.run_id

    # Log model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name=MODEL_NAME
    )

    # Log metric for traceability
    mlflow.log_metric("mae", best_mae)


# -----------------------------
# 5. Get Model Version
# -----------------------------
client = MlflowClient()

latest_version = client.get_latest_versions(MODEL_NAME, stages=["None"])[0].version


# -----------------------------
# 6. Save Results JSON
# -----------------------------
output = {
    "registered_model_name": MODEL_NAME,
    "version": int(latest_version),
    "run_id": run_id,
    "source_metric": "mae",
    "source_metric_value": float(best_mae)
}

output_path = os.path.join("results", "step3_s6.json")

with open(output_path, "w") as f:
    json.dump(output, f, indent=4)


print("Task 3 completed.")
print("Model registered as:", MODEL_NAME)
print("Version:", latest_version)
print("Run ID:", run_id)
print("Results saved to:", output_path)