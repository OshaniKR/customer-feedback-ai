
import pandas as pd
import joblib
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT_DIR / "models" / "sentiment_pipeline.joblib"
TEST_PATH = ROOT_DIR / "data" / "processed" / "test_sample.csv"
OUTPUT_PATH = ROOT_DIR / "results" / "threshold_analysis.csv"

# Load model and test data
model = joblib.load(MODEL_PATH)
test_df = pd.read_csv(TEST_PATH)

X_test = test_df["feedback"].astype(str)
y_test = test_df["sentiment"]

# Generate predictions and confidence scores
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
confidence = probabilities.max(axis=1)

results = pd.DataFrame({
    "actual": y_test,
    "predicted": predictions,
    "confidence": confidence,
})

results["correct"] = results["actual"] == results["predicted"]

# Compare different confidence thresholds
thresholds = [0.60, 0.70, 0.80, 0.90]
summary = []

for threshold in thresholds:
    accepted = results[results["confidence"] >= threshold]
    reviewed = results[results["confidence"] < threshold]

    coverage = len(accepted) / len(results) * 100

    accepted_accuracy = (
        accepted["correct"].mean() * 100
        if len(accepted) > 0
        else 0
    )

    summary.append({
        "threshold": threshold,
        "total_reviews": len(results),
        "automatic_predictions": len(accepted),
        "human_review": len(reviewed),
        "coverage_percent": round(coverage, 2),
        "accepted_accuracy_percent": round(accepted_accuracy, 2),
    })

summary_df = pd.DataFrame(summary)

print("\n=== Confidence Threshold Analysis ===")
print(summary_df.to_string(index=False))

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
summary_df.to_csv(OUTPUT_PATH, index=False)

print(f"\nResults saved to: {OUTPUT_PATH}")
