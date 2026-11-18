
import os
import json
import time
import platform

import mlflow
import mlflow.pyfunc

import pandas as pd
import matplotlib.pyplot as plt

import torch
import transformers

from transformers import pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "hf-sentiment-champion-challenger"

MODEL_A = "distilbert-base-uncased-finetuned-sst-2-english"
MODEL_B = "siebert/sentiment-roberta-large-english"

TASK = "sentiment-classification"
SEED = 42
MAX_LENGTH = 256
BATCH_SIZE = 4

DATA_FILE = "data/eval_inputs.csv"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

torch.manual_seed(SEED)


# --------------------------------------------------
# 2. CUSTOM MLFLOW MODEL WRAPPER
# --------------------------------------------------

class SentimentModel(mlflow.pyfunc.PythonModel):

    def __init__(self, hf_model_id):
        self.hf_model_id = hf_model_id

    def load_context(self, context):
        self.model = pipeline(
            "sentiment-analysis",
            model=self.hf_model_id,
            device=-1
        )

    def predict(self, context, model_input, params=None):

        texts = model_input["text"].astype(str).tolist()

        results = self.model(
            texts,
            truncation=True,
            max_length=MAX_LENGTH
        )

        predictions = []

        for result in results:
            label = result["label"].upper()

            if "POSITIVE" in label or label == "LABEL_1":
                predictions.append(1)
            else:
                predictions.append(0)

        return pd.DataFrame({
            "text": texts,
            "prediction": predictions
        })


# --------------------------------------------------
# 3. LOAD EVALUATION DATA
# --------------------------------------------------

print("\nLoading evaluation dataset...")

df = pd.read_csv(DATA_FILE)

df["label"] = df["label"].astype(int)

texts = df["text"].astype(str).tolist()
y_true = df["label"].tolist()

print("Number of evaluation samples:", len(df))


# --------------------------------------------------
# 4. FUNCTION TO EVALUATE ONE MODEL
# --------------------------------------------------

def evaluate_model(model_id, role, run_name):

    print("\n" + "=" * 55)
    print("Evaluating:", model_id)
    print("=" * 55)

    # Load Hugging Face model
    print("Loading model...")

    classifier = pipeline(
        "sentiment-analysis",
        model=model_id,
        device=-1
    )

    # Measure prediction time
    print("Generating predictions...")

    start_time = time.perf_counter()

    results = classifier(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        batch_size=BATCH_SIZE
    )

    total_time = time.perf_counter() - start_time

    # Convert model labels into 0 and 1
    predictions = []

    for result in results:

        label = result["label"].upper()

        if "POSITIVE" in label or label == "LABEL_1":
            predictions.append(1)
        else:
            predictions.append(0)

    # Calculate metrics
    accuracy = accuracy_score(y_true, predictions)

    macro_f1 = f1_score(
        y_true,
        predictions,
        average="macro",
        zero_division=0
    )

    avg_latency = (total_time / len(texts)) * 1000

    metrics = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "avg_latency_ms": float(avg_latency),
        "total_inference_s": float(total_time)
    }

    print("\nResults:")
    print(metrics)

    # Save predictions
    prediction_df = pd.DataFrame({
        "text": texts,
        "reference": y_true,
        "prediction": predictions,
        "correct": [
            actual == predicted
            for actual, predicted in zip(y_true, predictions)
        ]
    })

    short_name = "distilbert" if role == "champion" else "roberta"

    prediction_file = os.path.join(
        OUTPUT_DIR,
        f"predictions_{short_name}.csv"
    )

    prediction_df.to_csv(prediction_file, index=False)

    # Save confusion matrix
    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Negative", "Positive"]
    )

    display.plot()
    plt.title(f"Confusion Matrix - {short_name}")

    plot_file = os.path.join(
        OUTPUT_DIR,
        f"confusion_matrix_{short_name}.png"
    )

    plt.savefig(plot_file, bbox_inches="tight")
    plt.close()

    # Save individual metrics
    metrics_file = os.path.join(
        OUTPUT_DIR,
        f"metrics_{short_name}.json"
    )

    with open(metrics_file, "w") as file:
        json.dump(metrics, file, indent=4)

    # Log model and results to MLflow
    with mlflow.start_run(run_name=run_name) as run:

        mlflow.log_params({
            "hf_model_id": model_id,
            "task": TASK,
            "role": role,
            "max_length": MAX_LENGTH,
            "batch_size": BATCH_SIZE,
            "eval_samples": len(df),
            "seed": SEED,
            "device": "CPU"
        })

        mlflow.set_tags({
            "student_usn": "1MS25CI044",
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "mlflow_version": mlflow.__version__,
            "library_versions": (
                f"transformers={transformers.__version__}; "
                f"torch={torch.__version__}; "
                f"mlflow={mlflow.__version__}"
            ),
            "dataset_name": "custom_sentiment_dataset"
        })

        mlflow.log_metrics(metrics)

        mlflow.log_artifact(prediction_file)
        mlflow.log_artifact(plot_file)
        mlflow.log_artifact(metrics_file)
        mlflow.log_artifact(DATA_FILE)

        # Log the model so it can later be loaded through MLflow
        mlflow.pyfunc.log_model(
            name="model",
            python_model=SentimentModel(model_id),
            pip_requirements=[
                "mlflow",
                "transformers",
                "torch",
                "pandas"
            ]
        )

        run_id = run.info.run_id

    print("\nMLflow run completed:", run_name)

    del classifier

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "run_id": run_id,
        "model_id": model_id,
        "role": role,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "avg_latency_ms": avg_latency,
        "total_inference_s": total_time,
        "predictions": predictions,
        "prediction_df": prediction_df
    }


# --------------------------------------------------
# 5. EVALUATE CHAMPION AND CHALLENGER
# --------------------------------------------------

print("\nSTARTING CHAMPION-CHALLENGER EVALUATION")

model_a_results = evaluate_model(
    MODEL_A,
    "champion",
    "A_sentiment_distilbert-sst2"
)

model_b_results = evaluate_model(
    MODEL_B,
    "challenger",
    "B_sentiment_roberta-large"
)


# --------------------------------------------------
# 6. APPLY THE PREDEFINED DECISION RULE
# --------------------------------------------------

print("\nApplying Champion-Challenger decision rule...")

MARGIN = 0.01
LATENCY_LIMIT = 5.0

delta = (
    model_b_results["macro_f1"]
    - model_a_results["macro_f1"]
)

latency_ratio = (
    model_b_results["avg_latency_ms"]
    / max(model_a_results["avg_latency_ms"], 0.000001)
)

if delta >= MARGIN and latency_ratio <= LATENCY_LIMIT:
    winner = "challenger"
    print("RoBERTa becomes the Champion.")

elif abs(delta) < MARGIN:
    if model_b_results["avg_latency_ms"] < model_a_results["avg_latency_ms"]:
        winner = "challenger"
    else:
        winner = "champion"

else:
    winner = "champion"

print("Final winner:", winner)
print("Macro-F1 difference:", delta)


# --------------------------------------------------
# 7. SAVE COMPARISON RESULTS
# --------------------------------------------------

summary = pd.DataFrame([
    {
        "model": "DistilBERT",
        "accuracy": model_a_results["accuracy"],
        "macro_f1": model_a_results["macro_f1"],
        "avg_latency_ms": model_a_results["avg_latency_ms"],
        "total_inference_s": model_a_results["total_inference_s"]
    },
    {
        "model": "RoBERTa",
        "accuracy": model_b_results["accuracy"],
        "macro_f1": model_b_results["macro_f1"],
        "avg_latency_ms": model_b_results["avg_latency_ms"],
        "total_inference_s": model_b_results["total_inference_s"]
    }
])

# Save all four comparison metrics.
summary_file = os.path.join(
    OUTPUT_DIR,
    "metrics_summary.json"
)

with open(summary_file, "w") as file:
    json.dump(summary.to_dict(orient="records"), file, indent=4)


# Create a four-panel comparison figure so each metric has
# its own scale and the values remain easy to compare.
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

metrics_to_plot = [
    ("accuracy", "Accuracy", "Score"),
    ("macro_f1", "Macro-F1", "Score"),
    ("avg_latency_ms", "Average Latency", "Milliseconds"),
    ("total_inference_s", "Total Inference Time", "Seconds")
]

for ax, (metric, title, ylabel) in zip(axes.flatten(), metrics_to_plot):
    summary.plot(
        kind="bar",
        x="model",
        y=metric,
        ax=ax,
        legend=False
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)

    if metric in ("accuracy", "macro_f1"):
        ax.set_ylim(0, 1.05)

fig.suptitle("Champion vs Challenger Comparison")
fig.tight_layout()

comparison_plot = os.path.join(
    OUTPUT_DIR,
    "model_comparison.png"
)

fig.savefig(comparison_plot, bbox_inches="tight")
plt.close(fig)


# --------------------------------------------------
# 8. CHECK EVALUATION DISAGREEMENTS
# --------------------------------------------------

# Keep evaluation-set disagreements separate from the five curated
# supplemental examples already stored in disagreement_examples.csv.
evaluation_disagreements = []

for i in range(len(texts)):

    if (
        model_a_results["predictions"][i]
        != model_b_results["predictions"][i]
    ):

        evaluation_disagreements.append({
            "text": texts[i],
            "reference": y_true[i],
            "distilbert_prediction": model_a_results["predictions"][i],
            "roberta_prediction": model_b_results["predictions"][i]
        })

evaluation_disagreement_file = os.path.join(
    OUTPUT_DIR,
    "evaluation_disagreements.csv"
)

pd.DataFrame(evaluation_disagreements).to_csv(
    evaluation_disagreement_file,
    index=False
)

# This is the separate, manually reviewed supplemental set of five examples.
supplemental_disagreement_file = os.path.join(
    OUTPUT_DIR,
    "disagreement_examples.csv"
)


# --------------------------------------------------
# 9. CREATE SUMMARY MLFLOW RUN
# --------------------------------------------------

with mlflow.start_run(run_name="compare_sentiment"):

    mlflow.log_params({
        "primary_metric": "macro_f1",
        "margin": MARGIN,
        "latency_limit": LATENCY_LIMIT,
        "eval_samples": len(df)
    })

    mlflow.set_tag("winner", winner)

    mlflow.log_metric("delta_primary_metric", delta)

    mlflow.log_artifact(summary_file)
    mlflow.log_artifact(comparison_plot)
    mlflow.log_artifact(evaluation_disagreement_file)

    # Log the five curated supplemental examples when the file is present.
    if os.path.exists(supplemental_disagreement_file):
        mlflow.log_artifact(supplemental_disagreement_file)

    if os.path.exists("README.md"):
        mlflow.log_artifact("README.md")


# --------------------------------------------------
# 10. FINAL OUTPUT
# --------------------------------------------------

print("\n" + "=" * 55)
print("CHAMPION-CHALLENGER EVALUATION COMPLETED")
print("=" * 55)

print(summary.to_string(index=False))

print("\nWinner:", winner)
print("\nAll outputs saved in:", OUTPUT_DIR)
print("Check your MLflow dashboard for the experiment.")
