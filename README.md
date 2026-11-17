# Champion-Challenger Model Comparison

## 1. Task and Models

Task: Sentiment Classification

Model A - Initial Champion:
distilbert-base-uncased-finetuned-sst-2-english

Model B - Challenger:
siebert/sentiment-roberta-large-english

Dataset: Custom sentiment evaluation dataset.

Labels:
1 = Positive
0 = Negative

## 2. How to Run

1. Install requirements.
2. Start the MLflow tracking server.
3. Run evaluate.py.
4. Run register_champion.py.
5. Run predict_from_mlflow.py.

## 3. Champion Decision Rule

Primary metric: Macro F1-score (higher is better).

The Challenger must improve the macro F1-score
by at least 0.01 to replace the Champion.

If the improvement is less than 0.01,
the faster model remains Champion.

The Challenger will also be rejected if its
average inference time exceeds 5 times that
of the current Champion.

## 4. Results

Results will be added after evaluation.

## 5. Decision

The final Champion will be selected using
the predefined decision rule.

The results, performance differences,
sample size and inference speed will be discussed
after evaluation.

## 6. Limitations and Declaration

The evaluation uses a limited dataset, so the
results may not represent all possible inputs.

AI assistance was used during project development.
All implementation and design choices will be
reviewed and understood by the student.