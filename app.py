# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import os, requests, pandas as pd, numpy as np

app = FastAPI(title="SPY Preopen API")

class PredictRequest(BaseModel):
    use_live: bool = True  # toggle live Alpaca snapshot vs. fallback
    # you can add more inputs later if you want

@app.get("/")
def root():
    return {"ok": True, "msg": "See /docs for interactive API."}

@app.post("/predict")
def predict(req: PredictRequest):
    # --- example stub that just returns a fake prediction ---
    # (replace this block with your real “build df + model” code)
    return {
        "model_call": "Up",
        "confidence": 0.71,
        "premarket_gap": -0.0003,
        "vix_overnight_ret": -0.02
    }
