
from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline


# 1. Define project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train_sample.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test_sample.csv"

MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# 2. Load training and testing data
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train = train_df["feedback"]
y_train = train_df["sentiment"]

X_test = test_df["feedback"]
y_test = test_df["sentiment"]

print("Training reviews:", len(X_train))
print("Testing reviews:", len(X_test))


# 3. Create the machine-learning pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            stop_words="english",
        ),
    ),
    (
        "classifier",
        LogisticRegression(max_iter=1000),
    ),
])


# 4. Train the model
print("\nTraining the sentiment model...")

model.fit(X_train, y_train)

print("Training completed!")


# 5. Evaluate on the unseen test dataset
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"\nTest accuracy: {accuracy:.4f}")

print("\nClassification report:")
print(classification_report(y_test, predictions))

print("\nConfusion matrix:")
print(confusion_matrix(
    y_test,
    predictions,
    labels=["negative", "positive"],
))


# 6. Save the trained model
model_path = MODEL_DIR / "sentiment_pipeline.joblib"
joblib.dump(model, model_path)

# Save evaluation metrics
report = classification_report(
    y_test,
    predictions,
    output_dict=True,
)

pd.DataFrame(report).transpose().to_csv(
    RESULTS_DIR / "classification_report.csv"
)

print(f"\nModel saved to: {model_path}")
print(f"Evaluation report saved to: {RESULTS_DIR / 'classification_report.csv'}")


# 7. Test a new review
sample_review = [
    "This product is wonderful and I absolutely love it!"
]

sample_prediction = model.predict(sample_review)[0]

print("\nSample review:", sample_review[0])
print("Predicted sentiment:", sample_prediction)
