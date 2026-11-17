
import mlflow
from mlflow import MlflowClient

# Connect to MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

EXPERIMENT_NAME = "hf-sentiment-champion-challenger"
REGISTERED_MODEL_NAME = "SentimentChampionChallenger"

# Find the experiment
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    raise Exception("Experiment not found!")

# Get all runs
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id]
)

# Find the two model runs
run_a = None
run_b = None

for run in runs:
    if run.info.run_name == "A_sentiment_distilbert-sst2":
        run_a = run

    elif run.info.run_name == "B_sentiment_roberta-large":
        run_b = run

if run_a is None or run_b is None:
    raise Exception("Could not find both model runs!")

# Register DistilBERT
model_a_uri = f"runs:/{run_a.info.run_id}/model"

version_a = mlflow.register_model(
    model_uri=model_a_uri,
    name=REGISTERED_MODEL_NAME
)

# Register RoBERTa
model_b_uri = f"runs:/{run_b.info.run_id}/model"

version_b = mlflow.register_model(
    model_uri=model_b_uri,
    name=REGISTERED_MODEL_NAME
)

# Assign aliases
client.set_registered_model_alias(
    REGISTERED_MODEL_NAME,
    "champion",
    version_a.version
)

client.set_registered_model_alias(
    REGISTERED_MODEL_NAME,
    "challenger",
    version_b.version
)

print("\nMODEL REGISTRATION COMPLETED!")
print("Champion: DistilBERT | Version:", version_a.version)
print("Challenger: RoBERTa | Version:", version_b.version)
print("Registered model:", REGISTERED_MODEL_NAME)
