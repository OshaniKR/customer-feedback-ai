
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Locate project directories
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train_sample.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load training dataset
df = pd.read_csv(DATA_PATH)

print("--- Dataset Information ---")
print(df.info())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Duplicate Reviews ---")
print(df.duplicated(subset=["feedback"]).sum())

print("\n--- Review Length Statistics ---")
df["review_length"] = df["feedback"].str.len()
print(df["review_length"].describe())

# Visualize sentiment distribution
sentiment_counts = df["sentiment"].value_counts()

sentiment_counts.plot(kind="bar")
plt.title("Distribution of Customer Sentiments")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=0)
plt.tight_layout()

output_path = OUTPUT_DIR / "sentiment_distribution.png"
plt.savefig(output_path)
plt.close()

print(f"\nChart saved to: {output_path}")
