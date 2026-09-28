"""
Spam Mail Detector — FastAPI REST API
Author: Santosh Narreddy
Deployable on Render, Railway, or AWS.
"""

import os
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

app = FastAPI(
    title="Spam Mail Detector API",
    description="REST API for real-time email & SMS spam classification using TF-IDF + Naive Bayes",
    version="1.0.0"
)

MODEL_PATH = 'model/spam_classifier.pkl'

def get_pipeline():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    from app import get_pipeline as init_pipeline
    return init_pipeline()

pipeline = get_pipeline()

class MessageRequest(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Congratulations! You have won a free iPhone. Call now to claim."
            }
        }

class PredictionResponse(BaseModel):
    text: str
    is_spam: bool
    label: str
    confidence: float
    probabilities: Dict[str, float]

@app.get("/")
def root():
    return {
        "service": "Spam Mail Detector API",
        "status": "healthy",
        "author": "Santosh Narreddy",
        "docs": "/docs"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_spam(request: MessageRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Message text cannot be empty.")

    probs = pipeline.predict_proba([request.text])[0]
    ham_prob = float(probs[0])
    spam_prob = float(probs[1])

    is_spam = spam_prob >= 0.5
    label = "spam" if is_spam else "ham"
    confidence = (spam_prob if is_spam else ham_prob) * 100.0

    return {
        "text": request.text,
        "is_spam": is_spam,
        "label": label,
        "confidence": round(confidence, 2),
        "probabilities": {
            "ham": round(ham_prob, 4),
            "spam": round(spam_prob, 4)
        }
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
