
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# Find the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Define file paths
MODEL_PATH = PROJECT_ROOT / "models" / "sentiment_pipeline.joblib"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test_sample.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# Create output directories if they do not exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Check that the required files exist
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Trained model not found. Run src/models/train_sentiment.py first."
    )

if not TEST_PATH.exists():
    raise FileNotFoundError(
        "Test dataset not found. Run src/data/load_data.py first."
    )

# Load the trained model and test data
model = joblib.load(MODEL_PATH)
test_df = pd.read_csv(TEST_PATH)

X_test = test_df["feedback"]
y_test = test_df["sentiment"]

# Generate predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy:.2%}")

# Print classification report
report = classification_report(y_test, y_pred, zero_division=0)
print("\nClassification Report:")
print(report)

# Save classification report
report_dict = classification_report(
    y_test, y_pred, output_dict=True, zero_division=0
)
pd.DataFrame(report_dict).transpose().to_csv(
    RESULTS_DIR / "evaluation_report.csv"
)

# Generate confusion matrix
labels = ["negative", "positive"]
cm = confusion_matrix(y_test, y_pred, labels=labels)

print("\nConfusion Matrix:")
print(cm)

# Plot and save confusion matrix
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels,
)
display.plot(cmap="Blues", values_format="d")

plt.title("Customer Feedback Sentiment Confusion Matrix")
plt.tight_layout()
plt.savefig(
    FIGURES_DIR / "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

print("\nEvaluation report saved to results/evaluation_report.csv")
print("Confusion matrix saved to reports/figures/confusion_matrix.png")
