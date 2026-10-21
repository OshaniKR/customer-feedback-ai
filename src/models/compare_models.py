
from pathlib import Path
import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Find the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train_sample.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test_sample.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Check that the datasets exist
if not TRAIN_PATH.exists() or not TEST_PATH.exists():
    raise FileNotFoundError(
        "Dataset files not found. Run src/data/load_data.py first."
    )

# Load datasets
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train = train_df["feedback"]
y_train = train_df["sentiment"]

X_test = test_df["feedback"]
y_test = test_df["sentiment"]

# Define the algorithms
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Multinomial Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC(max_iter=2000),
}

results = []

# Train and evaluate each algorithm
for model_name, classifier in models.items():
    print(f"\nTraining {model_name}...")

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=20000,
                ngram_range=(1, 2),
                stop_words="english",
            ),
        ),
        ("classifier", classifier),
    ])

    start_time = time.time()
    pipeline.fit(X_train, y_train)
    training_time = time.time() - start_time

    predictions = pipeline.predict(X_test)

    results.append({
        "model": model_name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, pos_label="positive", zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, pos_label="positive", zero_division=0
        ),
        "f1_score": f1_score(
            y_test, predictions, pos_label="positive", zero_division=0
        ),
        "training_time_seconds": round(training_time, 2),
    })

    print(f"Accuracy: {results[-1]['accuracy']:.2%}")
    print(f"Training time: {training_time:.2f} seconds")

# Sort models by F1-score
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="f1_score", ascending=False
)

# Save and display the comparison
output_path = RESULTS_DIR / "model_comparison.csv"
results_df.to_csv(output_path, index=False)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False, formatters={
    "accuracy": "{:.2%}".format,
    "precision": "{:.2%}".format,
    "recall": "{:.2%}".format,
    "f1_score": "{:.2%}".format,
}))

print(f"\nComparison saved to: {output_path}")
