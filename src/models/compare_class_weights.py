
from pathlib import Path
from time import perf_counter

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train_sample.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test_sample.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train = train_df["feedback"]
y_train = train_df["sentiment"]
X_test = test_df["feedback"]
y_test = test_df["sentiment"]

experiments = [
    ("Baseline", None),
    ("Balanced", "balanced"),
]

results = []

for name, class_weight in experiments:
    print(f"\nTraining: {name}")

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            stop_words=None,
        )),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight=class_weight,
        )),
    ])

    start = perf_counter()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    elapsed = perf_counter() - start

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=["negative", "positive"],
    )

    results.append({
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "negative_precision": report["negative"]["precision"],
        "negative_recall": report["negative"]["recall"],
        "positive_f1": report["positive"]["f1-score"],
        "false_positives": matrix[0, 1],
        "false_negatives": matrix[1, 0],
        "time_seconds": round(elapsed, 2),
    })

    print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
    print("Confusion matrix:")
    print(matrix)
    print("\nClassification report:")
    print(classification_report(y_test, predictions))

results_df = pd.DataFrame(results)
output_path = RESULTS_DIR / "class_weight_comparison.csv"
results_df.to_csv(output_path, index=False)

print("\nCLASS WEIGHT COMPARISON")
print(results_df.to_string(index=False))
print(f"\nResults saved to: {output_path}")
