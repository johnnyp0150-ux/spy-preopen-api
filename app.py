import os, time, json
from functools import lru_cache
from typing import Optional, Literal

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingClassifier

APP_STARTED_AT = time.time()

ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
API_TOKEN         = os.getenv("API_TOKEN", "")  # your own simple bearer token

TRADE_BASE = "https://data.alpaca.markets/v2"
HDR = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY
}

UP_THR   = 0.0035
DOWN_THR = -0.0035
CONFIDENCE_CUTOFF = 0.60

app = FastAPI(title="SPY Pre-Open Predictor", version="1.0")

# ------------ helpers ------------
def label_day(oc_ret: float) -> Literal["Up","Down","Sideways"]:
    if oc_ret >= UP_THR:   return "Up"
    if oc_ret <= DOWN_THR: return "Down"
    return "Sideways"

def guardrail(pred: str, p: float) -> str:
    if pred == "Sideways":
        return "Guardrail: Sideways day predicted. Consider standing down."
    if p < CONFIDENCE_CUTOFF:
        return "Guardrail: low confidence (< cutoff). Consider NOT trading."
    return "OK: Meets confidence & not Sideways."

@lru_cache(maxsize=1)
def load_history(start: str = "2012-01-01") -> pd.DataFrame:
    # Basic backtest dataset (same shape as your notebook)
    spy  = yf.download("SPY",  start=start, auto_adjust=False, progress=False).rename(columns=str.title)
    vixy = yf.download("VIXY", start=start, auto_adjust=False, progress=False)
    es   = yf.download("ES=F", start=start, auto_adjust=False, progress=False)

    df = pd.DataFrame(index=spy.index)
    for c in ["Open","High","Low","Close","Volume"]:
        df[c] = spy[c]
    df["VIX_Close"] = vixy["Close"].reindex(df.index).ffill()
    df["ES_Close"]  = es["Close"].reindex(df.index).ffill()

    df["ES_overnight_ret"]  = df["ES_Close"].pct_change()
    df["VIX_overnight_ret"] = df["VIX_Close"].pct_change()
    df["Premarket_Gap"]     = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)
    df = df.dropna().copy()
    return df

@lru_cache(maxsize=1)
def train_model() -> tuple[GradientBoostingClassifier, list[str]]:
    df = load_history()
    feats = ["Premarket_Gap", "ES_overnight_ret", "VIX_overnight_ret"]

    oc_ret = df["Close"].pct_change()
    y = oc_ret.shift(-1).dropna().iloc[:-1]
    Y = y.map(label_day)
    X = df[feats].iloc[:-2].copy()
    Y = Y.reindex_like(X)

    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    Y = Y.loc[X.index]

    tscv = TimeSeriesSplit(n_splits=6)
    last_clf = None
    for tr, te in tscv.split(X):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = Y.iloc[tr], Y.iloc[te]
        clf = GradientBoostingClassifier(random_state=42)
        clf.fit(Xtr, ytr)
        last_clf = clf

    return last_clf, feats

def latest_trade_price(symbol: str) -> float:
    url = f"{TRADE_BASE}/stocks/{symbol}/trades/latest"
    r = requests.get(url, headers=HDR, timeout=10)
    r.raise_for_status()
    return float(r.json()["trade"]["p"])

def prev_daily_close(symbol: str) -> float:
    url = f"{TRADE_BASE}/stocks/{symbol}/bars"
    params = {"timeframe": "1Day", "limit": 2, "adjustment": "all"}
    r = requests.get(url, headers=HDR, params=params, timeout=10)
    r.raise_for_status()
    bars = r.json().get("bars", [])
    if len(bars) >= 2: return float(bars[-2]["c"])
    if len(bars) == 1: return float(bars[-1]["c"])
    return float("nan")

def build_live_features() -> dict:
    spy_last  = latest_trade_price("SPY")
    spy_prev  = prev_daily_close("SPY")
    vixy_last = latest_trade_price("VIXY")
    vixy_prev = prev_daily_close("VIXY")

    pre_gap = (spy_last - spy_prev) / spy_prev if np.isfinite(spy_last) and np.isfinite(spy_prev) else np.nan
    vix_ov  = (vixy_last / vixy_prev) - 1 if np.isfinite(vixy_last) and np.isfinite(vixy_prev) else np.nan

    # ES overnight ret: use yesterday’s from history (proxy, as in notebook)
    df = load_history()
    es_ov = float(df["ES_overnight_ret"].iloc[-1])

    return {
        "Premarket_Gap": float(pre_gap) if np.isfinite(pre_gap) else None,
        "ES_overnight_ret": float(es_ov),
        "VIX_overnight_ret": float(vix_ov) if np.isfinite(vix_ov) else None
    }

# ------------ schemas ------------
class PredictRequest(BaseModel):
    # Optional manual overrides if you want to simulate:
    premarket_gap: Optional[float] = None
    vix_overnight_ret: Optional[float] = None
    es_overnight_ret: Optional[float] = None

class PredictResponse(BaseModel):
    call: Literal["Up","Down","Sideways"]
    confidence: float
    guardrail: str
    probs: dict
    features: dict
    model_ready_at: float

# ------------ routes ------------
@app.get("/health")
def health():
    return {"ok": True, "uptime_s": round(time.time() - APP_STARTED_AT, 1)}

def require_token(authorization: Optional[str]):
    if not API_TOKEN:
        return  # unlocked (not recommended for public)
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")
    token = authorization.split(" ", 1)[1].strip()
    if token != API_TOKEN:
        raise HTTPException(status_code=403, detail="Bad token")

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, authorization: Optional[str] = Header(None)):
    require_token(authorization)

    # model
    clf, feats = train_model()

    # live features (with optional overrides)
    live = build_live_features()
    if req.premarket_gap is not None:
        live["Premarket_Gap"] = req.premarket_gap
    if req.vix_overnight_ret is not None:
        live["VIX_overnight_ret"] = req.vix_overnight_ret
    if req.es_overnight_ret is not None:
        live["ES_overnight_ret"] = req.es_overnight_ret

    if any(live[k] is None for k in feats):
        raise HTTPException(status_code=424, detail={"error": "Missing live features", "features": live})

    x = pd.DataFrame([live])[feats]
    proba = clf.predict_proba(x)[0]
    classes = list(clf.classes_)
    s = pd.Series(proba, index=classes).sort_values(ascending=False)

    call = s.index[0]
    conf = float(s.iloc[0])
    return PredictResponse(
        call=call,
        confidence=conf,
        guardrail=guardrail(call, conf),
        probs={k: float(v) for k, v in s.to_dict().items()},
        features=live,
        model_ready_at=APP_STARTED_AT,
    )
