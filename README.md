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

## Results

Both models achieved an accuracy of 1.00 and a macro-F1 score of 1.00 on the 20-example evaluation dataset.

| Metric | DistilBERT (Champion) | RoBERTa (Challenger) |
|---|---:|---:|
| Accuracy | 1.00 | 1.00 |
| Macro F1 | 1.00 | 1.00 |
| Average Latency (ms) | 23.15 | 190.40 |

Since both models achieved identical accuracy and macro-F1 scores, DistilBERT was selected as the Champion because it had significantly lower average prediction latency.

The evaluation dataset contains only 20 examples, so these results should not be considered a guarantee of performance on larger or unseen datasets.

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
## Discussion

The experiment compared DistilBERT and RoBERTa for sentiment analysis using the same evaluation dataset of 20 text examples. Both models achieved an accuracy of 1.00 and a macro-F1 score of 1.00, indicating that they correctly classified all examples in this evaluation.

Since the macro-F1 scores were identical, the predefined decision rule favoured the model with lower average prediction latency. DistilBERT recorded an average latency of 23.15 ms, while RoBERTa recorded 190.40 ms. Therefore, DistilBERT was selected as the Champion, and RoBERTa remained the Challenger.

This experiment demonstrates that model selection should consider both predictive performance and computational efficiency rather than accuracy alone. MLflow helped track experiments, compare metrics, store model versions, and assign Champion and Challenger aliases.

However, the evaluation dataset contained only 20 examples. This small sample may not represent real-world sentiment variations, complex language, sarcasm, or different writing styles. Further evaluation using a larger and more diverse dataset would be necessary before drawing conclusions about general performance.
## Model Disagreement Analysis

Both models produced identical predictions across all 20 evaluation examples. Therefore, no genuine disagreements were found, and `disagreement_examples.csv` is empty.

To provide additional analysis, `comparison_observations.csv` contains five representative examples where both models agreed, along with comments explaining their predictions.

These observations are not disagreements and are reported separately to preserve the accuracy and transparency of the evaluation.