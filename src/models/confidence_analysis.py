
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score

# Project root directory
ROOT_DIR = Path(__file__).resolve().parents[2]

# File paths
MODEL_PATH = ROOT_DIR / "models" / "sentiment_pipeline.joblib"
TEST_PATH = ROOT_DIR / "data" / "processed" / "test_sample.csv"
OUTPUT_PATH = ROOT_DIR / "results" / "confidence_analysis.csv"

# Load the trained model and test data
model = joblib.load(MODEL_PATH)
test_df = pd.read_csv(TEST_PATH)

X_test = test_df["feedback"].astype(str)
y_test = test_df["sentiment"]

# Predict sentiment and probabilities
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Confidence is the highest predicted class probability
confidence = probabilities.max(axis=1)

# Check whether each prediction is correct
correct = predictions == y_test

# Create a results DataFrame
results = pd.DataFrame({
    "review": X_test,
    "actual_label": y_test,
    "predicted_label": predictions,
    "confidence": confidence,
    "correct": correct,
})

# Overall confidence analysis
print("\n=== Confidence Analysis ===")
print(f"Total reviews: {len(results)}")
print(f"Correct predictions: {correct.sum()}")
print(f"Incorrect predictions: {(~correct).sum()}")
print(f"Overall accuracy: {accuracy_score(y_test, predictions):.2%}")
print(f"Average confidence (all): {confidence.mean():.2%}")
print(f"Average confidence (correct): {confidence[correct].mean():.2%}")
print(f"Average confidence (incorrect): {confidence[~correct].mean():.2%}")

# Group reviews by confidence level
bins = [0.0, 0.60, 0.70, 0.80, 0.90, 1.01]
labels = ["Below 60%", "60-70%", "70-80%", "80-90%", "90% and above"]

results["confidence_group"] = pd.cut(
    results["confidence"],
    bins=bins,
    labels=labels,
    right=False,
)

summary = results.groupby(
    "confidence_group",
    observed=False,
).agg(
    review_count=("correct", "size"),
    accuracy=("correct", "mean"),
    average_confidence=("confidence", "mean"),
)

print("\n=== Accuracy by Confidence Group ===")
print(summary.to_string())

# Save detailed results
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(OUTPUT_PATH, index=False)

print(f"\nDetailed results saved to: {OUTPUT_PATH}")
