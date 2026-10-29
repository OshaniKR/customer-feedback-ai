
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Locate the project root and saved model
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "sentiment_pipeline.joblib"

# Load the trained model once when the API starts
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Customer Feedback Sentiment API")

# Allow requests from the local React development server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class ReviewRequest(BaseModel):
    review: str = Field(min_length=1, max_length=10000)


@app.get("/")
def home():
    return {"message": "Customer Feedback Sentiment API is running"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
def predict_sentiment(request: ReviewRequest):
    review = request.review.strip()

    if not review:
        raise HTTPException(
            status_code=400,
            detail="Please enter a customer review."
        )

    try:
        prediction = model.predict([review])[0]
        result = str(prediction)

        response = {
            "review": review,
            "sentiment": result,
        }

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([review])[0]
            confidence = float(max(probabilities))

            response["confidence"] = round(confidence * 100, 2)

        return response

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Unable to analyze the review."
        ) from exc
