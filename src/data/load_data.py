
from pathlib import Path

import pandas as pd


# Find the main project folder automatically
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Original Kaggle dataset files
TRAIN_FILE = PROJECT_ROOT / "data" / "raw" / "train.ft.txt"
TEST_FILE = PROJECT_ROOT / "data" / "raw" / "test.ft.txt"

# Output folder for our smaller working datasets
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_fasttext_file(file_path, limit=20000):
    """Load reviews from a FastText-formatted text file."""

    reviews = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file):
            if limit is not None and line_number >= limit:
                break

            line = line.strip()

            if not line:
                continue

            # Separate the label from the review text
            label, separator, review_text = line.partition(" ")

            if not separator or label not in {"__label__1", "__label__2"}:
                continue

            # Convert numeric labels into readable sentiment names
            sentiment = {
                "__label__1": "negative",
                "__label__2": "positive",
            }[label]

            reviews.append({
                "feedback": review_text,
                "sentiment": sentiment,
            })

    return pd.DataFrame(reviews, columns=["feedback", "sentiment"])


def main():
    # Make sure the output folder exists
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading training reviews...")
    train_df = load_fasttext_file(TRAIN_FILE, limit=20000)

    print("Loading test reviews...")
    test_df = load_fasttext_file(TEST_FILE, limit=5000)

    # Save our smaller datasets without changing the originals
    train_output = PROCESSED_DIR / "train_sample.csv"
    test_output = PROCESSED_DIR / "test_sample.csv"

    train_df.to_csv(train_output, index=False)
    test_df.to_csv(test_output, index=False)

    print("\n--- Training Dataset ---")
    print("Shape:", train_df.shape)
    print("Sentiment counts:")
    print(train_df["sentiment"].value_counts())
    print("\nFirst five reviews:")
    print(train_df.head())

    print("\n--- Test Dataset ---")
    print("Shape:", test_df.shape)
    print("Sentiment counts:")
    print(test_df["sentiment"].value_counts())

    print("\nProcessed datasets saved successfully!")
    print("Training file:", train_output)
    print("Test file:", test_output)


if __name__ == "__main__":
    main()
