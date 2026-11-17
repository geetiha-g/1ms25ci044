
import mlflow
from mlflow import MlflowClient
import pandas as pd

# Connect to MLflow
TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME = "SentimentChampionChallenger"

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

# Get the Champion model version using its alias
model_version = client.get_model_version_by_alias(
    MODEL_NAME,
    "champion"
)

# Get the Hugging Face model ID from the original run
run = mlflow.get_run(model_version.run_id)
hf_model_id = run.data.params.get("hf_model_id", "Not found")

print("\nCHAMPION MODEL DETAILS")
print("Registered model:", MODEL_NAME)
print("Model version:", model_version.version)
print("Hugging Face model:", hf_model_id)

# Load the model directly from MLflow
model_uri = f"models:/{MODEL_NAME}@champion"
model = mlflow.pyfunc.load_model(model_uri)

# Load new data
data = pd.read_csv("data/new_data.csv")

# Generate predictions through MLflow
predictions = model.predict(data[["text"]])

# Prepare readable output
results = data.copy()

if isinstance(predictions, pd.DataFrame):
    results["prediction"] = predictions["prediction"].values
else:
    results["prediction"] = predictions

results["sentiment"] = results["prediction"].map({
    1: "POSITIVE",
    0: "NEGATIVE"
})

# Display results
print("\nNEW DATA PREDICTIONS")
print(results[["text", "sentiment"]].to_string(index=False))

# Save predictions
results.to_csv(
    "outputs/new_data_predictions.csv",
    index=False
)

print("\nPredictions saved to outputs/new_data_predictions.csv")
