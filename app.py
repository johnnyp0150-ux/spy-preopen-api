# app.py
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="SPY Preopen API")

@app.get("/")
def root():
    return {"ok": True, "message": "Use POST /predict or visit /docs"}

class PredictRequest(BaseModel):
    test_value: float = 0.0

@app.post("/predict")
def predict(req: PredictRequest):
    # This is a placeholder for testing Render
    return {
        "model_call": "Up",
        "confidence": 0.73,
        "received_test_value": req.test_value
    }
