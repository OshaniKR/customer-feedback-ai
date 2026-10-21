
from pathlib import Path

import joblib

# Locate the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "sentiment_pipeline.joblib"

# Load the trained model
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Trained model not found. Run train_sentiment.py first."
    )

model = joblib.load(MODEL_PATH)


def predict_sentiment(review):
    """Predict whether a customer review is positive or negative."""
    prediction = model.predict([review])[0]
    probabilities = model.predict_proba([review])[0]

    confidence = max(probabilities) * 100

    return prediction, confidence


if __name__ == "__main__":
    print("Customer Feedback Sentiment Predictor")
    print("Type 'exit' to quit.\n")

    while True:
        review = input("Enter a customer review: ").strip()

        if review.lower() == "exit":
            print("Goodbye!")
            break

        if not review:
            print("Please enter a review.")
            continue

        sentiment, confidence = predict_sentiment(review)

        print(f"Predicted sentiment: {sentiment}")
        print(f"Model confidence: {confidence:.2f}%\n")
