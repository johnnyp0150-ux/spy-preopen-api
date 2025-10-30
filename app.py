# app.py
import os, time, json
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from fastapi import FastAPI, HTTPException, Header

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingClassifier

# -------------------- Config & Globals --------------------
APP_STARTED_AT = time.time()

ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
API_TOKEN         = os.getenv("API_TOKEN", "")  # optional bearer check

TRADE_BASE = "https://data.alpaca.markets/v2"
HDR = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
}

# Same thresholds as your notebook
UP_THR   =  0.0035
DOWN_THR = -0.0035

# Simple in-memory cache to avoid re-downloading/retraining every request
_HIST_DF: Optional[pd.DataFrame] = None
_HIST_TS: float = 0.0
_CLF = None
_FEAT_NAMES = None
_MODEL_TS: float = 0.0

# Rebuild after this many seconds (12 hours)
REFRESH_SECS = 12 * 3600

app = FastAPI(title="SPY Preopen API", version="0.1.0")


# -------------------- Schemas --------------------
from pydantic import BaseModel
class PredictIn(BaseModel):
    # keep body optional so you can POST {} from phone
    test_value: Optional[int] = None


# -------------------- Health --------------------
@app.get("/healthz")
def healthz():
    """Lightweight liveness/readiness probe (no external calls)."""
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - APP_STARTED_AT, 2),
        "model_cached": _CLF is not None,
        "history_cached": _HIST_DF is not None,
        "version": "0.1.0",
    }

@app.get("/health")
def health():
    return healthz()


# -------------------- Data helpers --------------------
def _need_refresh(ts: float) -> bool:
    return (time.time() - ts) > REFRESH_SECS


def latest_trade(symbol: str) -> float:
    """
    Live last trade:
      • Use Alpaca if keys are present
      • Fallback to yfinance intraday if not
    """
    if ALPACA_API_KEY and ALPACA_SECRET_KEY:
        url = f"{TRADE_BASE}/stocks/{symbol}/trades/latest"
        r = requests.get(url, headers=HDR, timeout=20)
        if r.status_code == 401:
            # credentials present but invalid
            raise HTTPException(502, "Alpaca auth failed (check keys)")
        r.raise_for_status()
        return float(r.json()["trade"]["p"])

    # Fallback: yfinance 1m bars for the current day
    t = yf.Ticker(symbol)
    df = t.history(period="1d", interval="1m")
    if df.empty:
        raise HTTPException(502, f"Could not fetch intraday for {symbol} via yfinance.")
    return float(df["Close"].iloc[-1])


def prev_close(symbol: str) -> float:
    """Yesterday close via yfinance (robust & free)."""
    df = yf.download(symbol, period="5d", auto_adjust=False, progress=False)
    if df.empty or len(df) < 2:
        raise HTTPException(502, f"Could not fetch recent history for {symbol}.")
    return float(df["Close"].iloc[-2])


def build_history() -> pd.DataFrame:
    """
    Same features you used in the notebook.
    Cached in memory to avoid re-downloading on every request.
    """
    global _HIST_DF, _HIST_TS
    if _HIST_DF is not None and not _need_refresh(_HIST_TS):
        return _HIST_DF

    spy = yf.download("SPY", start="2012-01-01", auto_adjust=False, progress=False)
    vix = yf.download("VIXY", start="2012-01-01", auto_adjust=False, progress=False)
    es  = yf.download("ES=F", start="2012-01-01", auto_adjust=False, progress=False)

    spy = spy.rename(columns=str.title)
    df = pd.DataFrame(index=spy.index)
    df["Open"]      = spy["Open"]
    df["Close"]     = spy["Close"]
    df["VIX_Close"] = vix["Close"].reindex(df.index).ffill()
    df["ES_Close"]  = es["Close"].reindex(df.index).ffill()

    # Features from the backtest
    df["ES_overnight_ret"]  = df["ES_Close"].pct_change()
    df["VIX_overnight_ret"] = df["VIX_Close"].pct_change()
    df["Premarket_Gap"]     = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)

    # Labels using thresholds
    oc_ret = (df["Close"] / df["Close"].shift(1)) - 1.0
    lab = np.where(oc_ret >= UP_THR, "Up",
          np.where(oc_ret <= DOWN_THR, "Down", "Sideways"))
    df["label"] = lab

    df = df.dropna(subset=["ES_overnight_ret","VIX_overnight_ret","Premarket_Gap"])

    _HIST_DF = df
    _HIST_TS = time.time()
    return df


def train_quick(df: pd.DataFrame):
    """Train a simple GBC like in your notebook (cached)."""
    global _CLF, _FEAT_NAMES, _MODEL_TS
    if _CLF is not None and _FEAT_NAMES is not None and not _need_refresh(_MODEL_TS):
        return _CLF, _FEAT_NAMES

    X = df[["ES_overnight_ret", "VIX_overnight_ret", "Premarket_Gap"]].copy()
    y = df["label"].copy()

    tscv = TimeSeriesSplit(n_splits=6)
    last_clf = None
    for tr, te in tscv.split(X):
        Xtr, ytr = X.iloc[tr], y.iloc[tr]
        last_clf = GradientBoostingClassifier(random_state=42)
        last_clf.fit(Xtr, ytr)

    _CLF = last_clf
    _FEAT_NAMES = X.columns.tolist()
    _MODEL_TS = time.time()
    return _CLF, _FEAT_NAMES


# -------------------- Routes --------------------
@app.get("/")
def root():
    return {"ok": True, "msg": "See /docs for interactive API. Health: /healthz"}


@app.post("/predict")
def predict(payload: PredictIn, authorization: Optional[str] = Header(None)):
    # Optional bearer token (set API_TOKEN in Render if you want protection)
    if API_TOKEN:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(401, "Missing Bearer token")
        if authorization.split(" ",1)[1] != API_TOKEN:
            raise HTTPException(403, "Invalid token")

    try:
        # Cache-aware history + model
        df_hist = build_history()
        clf, feat_names = train_quick(df_hist)

        # Live features
        spy_last  = latest_trade("SPY")
        vixy_last = latest_trade("VIXY")
        spy_yday  = prev_close("SPY")
        vixy_yday = prev_close("VIXY")

        premarket_gap  = (spy_last / spy_yday) - 1.0
        vix_overnight  = (vixy_last / vixy_yday) - 1.0
        es_overnight   = df_hist["ES_overnight_ret"].iloc[-1]  # keep proxy same as notebook

        x_live = pd.DataFrame([{
            "ES_overnight_ret":  es_overnight,
            "VIX_overnight_ret": vix_overnight,
            "Premarket_Gap":     premarket_gap
        }], columns=feat_names)

        proba = clf.predict_proba(x_live)[0]
        classes = clf.classes_
        conf_series = pd.Series(proba, index=classes).sort_values(ascending=False)
        pred = conf_series.index[0]
        p = float(conf_series.iloc[0])

        guidance = (
            "OK: Meets confidence & not Sideways."
            if pred != "Sideways" and p >= 0.60
            else (
                "Guardrail: Sideways day predicted. Consider standing down."
                if pred == "Sideways"
                else "Guardrail: low confidence. Consider NOT trading."
            )
        )

        return {
            "model_call": pred,
            "confidence": round(p, 4),
            "features": {
                "premarket_gap": round(float(premarket_gap), 6),
                "vix_overnight_ret": round(float(vix_overnight), 6),
                "es_overnight_ret": round(float(es_overnight), 6),
            },
            "guidance": guidance,
            "cold_start_seconds": round(time.time() - APP_STARTED_AT, 2),
            "echo": payload.dict()
        }

    except HTTPException:
        raise
    except Exception as e:
        # surface a clean error instead of HTML
        raise HTTPException(status_code=500, detail=str(e))
