
from pathlib import Path
from time import perf_counter

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

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
    {"name": "A_Current", "max_features": 20000, "ngram_range": (1, 2)},
    {"name": "B_30000_features", "max_features": 30000, "ngram_range": (1, 2)},
    {"name": "C_Trigrams", "max_features": 20000, "ngram_range": (1, 3)},
]

results = []

for experiment in experiments:
    print(f"\nRunning experiment: {experiment['name']}")

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=experiment["max_features"],
            ngram_range=experiment["ngram_range"],
            stop_words=None,
        )),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    start_time = perf_counter()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    elapsed = perf_counter() - start_time

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, pos_label="positive")

    results.append({
        "experiment": experiment["name"],
        "max_features": experiment["max_features"],
        "ngram_range": str(experiment["ngram_range"]),
        "accuracy": accuracy,
        "positive_f1": f1,
        "time_seconds": round(elapsed, 2),
    })

    print(f"Accuracy: {accuracy:.2%}")
    print(f"Positive F1-score: {f1:.4f}")
    print(f"Training and prediction time: {elapsed:.2f} seconds")

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("accuracy", ascending=False)

output_path = RESULTS_DIR / "tfidf_comparison.csv"
results_df.to_csv(output_path, index=False)

print("\nTF-IDF COMPARISON")
print(results_df.to_string(index=False))
print(f"\nResults saved to: {output_path}")
