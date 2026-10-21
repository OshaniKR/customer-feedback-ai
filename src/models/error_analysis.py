
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "sentiment_pipeline.joblib"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test_sample.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load the existing model and test data
model = joblib.load(MODEL_PATH)
test_df = pd.read_csv(TEST_PATH)

X_test = test_df["feedback"]
y_test = test_df["sentiment"]

# Predict sentiment for each test review
y_pred = model.predict(X_test)

# Keep only incorrect predictions
errors_df = pd.DataFrame({
    "feedback": X_test,
    "actual_sentiment": y_test,
    "predicted_sentiment": y_pred,
})

errors_df = errors_df[
    errors_df["actual_sentiment"] != errors_df["predicted_sentiment"]
].copy()

# Save incorrect predictions for inspection
output_path = RESULTS_DIR / "misclassified_reviews.csv"
errors_df.to_csv(output_path, index=False)

print(f"Total test reviews: {len(test_df)}")
print(f"Incorrect predictions: {len(errors_df)}")
print(f"Error rate: {len(errors_df) / len(test_df):.2%}")

print("\nFirst 10 misclassified reviews:")
for _, row in errors_df.head(10).iterrows():
    print("\nReview:", row["feedback"])
    print("Actual:", row["actual_sentiment"])
    print("Predicted:", row["predicted_sentiment"])

print(f"\nAll misclassified reviews saved to: {output_path}")
