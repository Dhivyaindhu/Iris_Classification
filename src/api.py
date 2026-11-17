"""FastAPI app from Day 35, now with tests around it."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.model import predict_one, train_model

app = FastAPI(title="Iris Classifier")
_model, _ = train_model()


class IrisRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: IrisRequest):
    try:
        label, confidence = predict_one(_model, req.features)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"label": label, "confidence": round(confidence, 4)}
