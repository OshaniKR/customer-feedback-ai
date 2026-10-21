
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "sentiment_pipeline.joblib"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test_sample.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

model = joblib.load(MODEL_PATH)
test_df = pd.read_csv(TEST_PATH)

X_test = test_df["feedback"]
y_test = test_df["sentiment"]
y_pred = model.predict(X_test)

print("CLASSIFICATION REPORT")
print(classification_report(y_test, y_pred))

print("CONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred, labels=["negative", "positive"]))

errors_df = pd.DataFrame({
    "feedback": X_test,
    "actual_sentiment": y_test,
    "predicted_sentiment": y_pred,
})

false_positives = errors_df[
    (errors_df["actual_sentiment"] == "negative")
    & (errors_df["predicted_sentiment"] == "positive")
].copy()

false_negatives = errors_df[
    (errors_df["actual_sentiment"] == "positive")
    & (errors_df["predicted_sentiment"] == "negative")
].copy()

false_positives.to_csv(
    RESULTS_DIR / "false_positives.csv", index=False
)
false_negatives.to_csv(
    RESULTS_DIR / "false_negatives.csv", index=False
)

print(f"\nFalse positives: {len(false_positives)}")
print(f"False negatives: {len(false_negatives)}")

print("\nExamples of false positives:")
for _, row in false_positives.head(5).iterrows():
    print("\nReview:", row["feedback"])
    print("Actual:", row["actual_sentiment"])
    print("Predicted:", row["predicted_sentiment"])

print("\nExamples of false negatives:")
for _, row in false_negatives.head(5).iterrows():
    print("\nReview:", row["feedback"])
    print("Actual:", row["actual_sentiment"])
    print("Predicted:", row["predicted_sentiment"])

print("\nDetailed error files saved in the results folder.")
