# app.py
import os, time, math
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingClassifier

APP_STARTED_AT = time.time()

# --- ENV / CONFIG ---
ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
API_TOKEN         = os.getenv("API_TOKEN", "")        # optional bearer
TRADE_MODE        = os.getenv("TRADE_MODE", "equity") # "equity" now, "options" later
ALPACA_TRADE_BASE = os.getenv("ALPACA_TRADE_BASE", "https://paper-api.alpaca.markets")
ALPACA_DATA_BASE  = os.getenv("ALPACA_DATA_BASE",  "https://data.alpaca.markets/v2")

HDR_DATA = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
}
HDR_TRADE = HDR_DATA.copy()
HDR_JSON  = {"Content-Type":"application/json", **HDR_TRADE}

UP_THR   =  0.0035
DOWN_THR = -0.0035

# Regime thresholds
REG_MIN_ABS_GAP = 0.0020     # 0.20%
REG_MIN_VIX_PCT = 0.02       # +/-2% threshold
REG_MIN_ATR_PCT = 0.0080     # 0.80%

# Caches
_HIST_DF: Optional[pd.DataFrame] = None
_HIST_TS: float = 0.0
_CLF = None
_FEAT_NAMES = None
_MODEL_TS: float = 0.0

REFRESH_SECS = 12 * 3600
_LAST_HISTORY_FALLBACK = False
_LAST_MODEL_FALLBACK = False

app = FastAPI(title="SPY Preopen API", version="0.3.0")

class PredictIn(BaseModel):
    test_value: Optional[int] = None

def _need_refresh(ts: float) -> bool:
    return (time.time() - ts) > REFRESH_SECS

@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - APP_STARTED_AT, 2),
        "model_cached": _CLF is not None,
        "history_cached": _HIST_DF is not None,
        "version": "0.3.0",
    }

@app.get("/health")
def health():
    return healthz()

def latest_trade(symbol: str) -> float:
    # Prefer Alpaca market data if keys exist
    if ALPACA_API_KEY and ALPACA_SECRET_KEY:
        url = f"{ALPACA_DATA_BASE}/stocks/{symbol}/trades/latest"
        r = requests.get(url, headers=HDR_DATA, timeout=20)
        if r.status_code == 401:
            raise HTTPException(502, "Alpaca auth failed (check keys)")
        r.raise_for_status()
        return float(r.json()["trade"]["p"])
    # Fallback to yfinance 1m
    t = yf.Ticker(symbol)
    df = t.history(period="1d", interval="1m")
    if df.empty:
        raise HTTPException(502, f"Could not fetch intraday for {symbol}.")
    return float(df["Close"].iloc[-1])

def prev_close(symbol: str) -> float:
    df = yf.download(symbol, period="5d", auto_adjust=False, progress=False)
    if df.empty or len(df) < 2:
        raise HTTPException(502, f"Could not fetch recent history for {symbol}.")
    return float(df["Close"].iloc[-2])

def build_history() -> pd.DataFrame:
    global _HIST_DF, _HIST_TS, _LAST_HISTORY_FALLBACK
    _LAST_HISTORY_FALLBACK = False
    if _HIST_DF is not None and not _need_refresh(_HIST_TS):
        return _HIST_DF

    spy = yf.download("SPY", start="2012-01-01", auto_adjust=False, progress=False)
    vix = yf.download("^VIX", start="2012-01-01", auto_adjust=False, progress=False)
    es  = yf.download("ES=F", start="2012-01-01", auto_adjust=False, progress=False)

    if spy.empty or vix.empty or es.empty:
        now = pd.Timestamp.utcnow().floor("D")
        df = pd.DataFrame(
            {
                "Open":[500,501],"High":[502,503],"Low":[498,500],"Close":[501,502],
                "VIX_Close":[15,15.1], "ES_Close":[5200,5205],
            }, index=[now - pd.Timedelta(days=1), now]
        )
        _LAST_HISTORY_FALLBACK = True
    else:
        spy = spy.rename(columns=str.title)
        df = pd.DataFrame(index=spy.index)
        for c in ["Open","High","Low","Close"]:
            df[c] = spy[c]
        df["VIX_Close"] = vix["Close"].reindex(df.index).ffill()
        df["ES_Close"]  = es["Close"].reindex(df.index).ffill()

    df["ES_overnight_ret"]  = df["ES_Close"].pct_change()
    df["VIX_overnight_ret"] = df["VIX_Close"].pct_change()
    df["Premarket_Gap"]     = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)

    oc_ret = (df["Close"] / df["Close"].shift(1)) - 1.0
    df["label"] = np.where(oc_ret >= UP_THR, "Up",
                    np.where(oc_ret <= DOWN_THR, "Down", "Sideways"))

    # ATR10%
    tr = np.maximum(df["High"] - df["Low"],
         np.maximum(abs(df["High"] - df["Close"].shift(1)),
                    abs(df["Low"]  - df["Close"].shift(1))))
    df["ATR10"] = pd.Series(tr).rolling(10).mean()
    df["ATR10_pct"] = df["ATR10"] / df["Close"].shift(1)

    df = df.dropna(subset=["ES_overnight_ret","VIX_overnight_ret","Premarket_Gap","ATR10_pct"])

    if df.empty:
        now = pd.Timestamp.utcnow().floor("D")
        df = pd.DataFrame(
            {
                "Open":[500,501],"High":[502,503],"Low":[498,500],"Close":[501,502],
                "VIX_Close":[15,15.1],"ES_Close":[5200,5205],
                "ES_overnight_ret":[0.0,(5205/5200-1.0)],
                "VIX_overnight_ret":[0.0,(15.1/15.0-1.0)],
                "Premarket_Gap":[0.0,0.0],
                "ATR10_pct":[0.01,0.01],
                "label":["Sideways","Sideways"],
            }, index=[now - pd.Timedelta(days=1), now]
        )
        _LAST_HISTORY_FALLBACK = True

    _HIST_DF, _HIST_TS = df, time.time()
    return df

def train_quick(df: pd.DataFrame):
    global _CLF, _FEAT_NAMES, _MODEL_TS, _LAST_MODEL_FALLBACK
    _LAST_MODEL_FALLBACK = False
    if _CLF is not None and _FEAT_NAMES is not None and not _need_refresh(_MODEL_TS):
        return _CLF, _FEAT_NAMES

    X = df[["ES_overnight_ret","VIX_overnight_ret","Premarket_Gap"]].copy()
    y = df["label"].copy()
    if len(X) < 3:
        from sklearn.dummy import DummyClassifier
        d = DummyClassifier(strategy="most_frequent")
        d.fit(np.zeros((1,3)), ["Sideways"])
        _CLF, _FEAT_NAMES, _MODEL_TS = d, X.columns.tolist(), time.time()
        _LAST_MODEL_FALLBACK = True
        return _CLF, _FEAT_NAMES

    n_splits = max(2, min(6, len(X)-1))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    clf = None
    for tr, te in tscv.split(X):
        Xtr, ytr = X.iloc[tr], y.iloc[tr]
        clf = GradientBoostingClassifier(random_state=42)
        clf.fit(Xtr, ytr)
    _CLF, _FEAT_NAMES, _MODEL_TS = clf, X.columns.tolist(), time.time()
    return _CLF, _FEAT_NAMES

@app.get("/")
def root():
    return {"ok": True, "msg": "See /docs. Health: /healthz"}

def _auth_or_403(authorization: Optional[str]):
    if not API_TOKEN:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing Bearer token")
    if authorization.split(" ",1)[1] != API_TOKEN:
        raise HTTPException(403, "Invalid token")

@app.post("/predict")
def predict(payload: PredictIn, authorization: Optional[str] = Header(None)):
    _auth_or_403(authorization)
    try:
        df_hist = build_history()
        clf, feat_names = train_quick(df_hist)

        spy_last  = latest_trade("SPY")
        vix_last  = latest_trade("VIXY")  # proxy
        spy_yday  = prev_close("SPY")
        vix_yday  = prev_close("VIXY")

        premarket_gap = (spy_last / spy_yday) - 1.0
        vix_overnight = (vix_last / vix_yday) - 1.0
        es_overnight  = float(df_hist["ES_overnight_ret"].iloc[-1])

        x_live = pd.DataFrame([{
            "ES_overnight_ret":  es_overnight,
            "VIX_overnight_ret": vix_overnight,
            "Premarket_Gap":     premarket_gap
        }], columns=feat_names)

        proba = clf.predict_proba(x_live)[0]
        classes = clf.classes_
        conf_series = pd.Series(proba, index=classes).sort_values(ascending=False)
        pred = conf_series.index[0]
        p    = float(conf_series.iloc[0])

        atr_pct = float(df_hist["ATR10_pct"].iloc[-1])
        gap_abs = abs(float(premarket_gap))
        vix_pct = float(vix_overnight)

        vix_ok_long  = (vix_pct <= 0.0) or (abs(vix_pct) >= REG_MIN_VIX_PCT)
        vix_ok_short = (vix_pct >= 0.0) or (abs(vix_pct) >= REG_MIN_VIX_PCT)
        vix_ok = vix_ok_long if pred == "Up" else (vix_ok_short if pred == "Down" else False)

        regime = {
            "gap_abs": round(gap_abs,6),
            "atr10_pct": round(atr_pct,6),
            "vix_pct": round(vix_pct,6),
            "gap_ok": gap_abs >= REG_MIN_ABS_GAP,
            "atr_ok": atr_pct >= REG_MIN_ATR_PCT,
            "vix_ok": vix_ok,
        }
        regime_pass = bool(regime["gap_ok"] and regime["atr_ok"] and regime["vix_ok"] and pred != "Sideways")

        guidance = (
            "OK: Meets confidence & not Sideways."
            if pred != "Sideways" and p >= 0.60
            else ("Guardrail: Sideways day predicted. Consider standing down."
                  if pred == "Sideways"
                  else "Guardrail: low confidence. Consider NOT trading.")
        )

        mode = {"history_fallback": _LAST_HISTORY_FALLBACK, "model_fallback": _LAST_MODEL_FALLBACK}

        return {
            "model_call": pred,
            "confidence": round(p,4),
            "features": {
                "premarket_gap": round(float(premarket_gap),6),
                "vix_overnight_ret": round(float(vix_overnight),6),
                "es_overnight_ret": round(float(es_overnight),6),
            },
            "regime": regime,
            "regime_pass": regime_pass,
            "guidance": guidance,
            "mode": mode,
            "cold_start_seconds": round(time.time() - APP_STARTED_AT, 2),
            "echo": payload.dict()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class TradeIn(BaseModel):
    model_call: str
    confidence: float
    regime_pass: bool

@app.post("/trade")
def trade(body: TradeIn, authorization: Optional[str] = Header(None)):
    _auth_or_403(authorization)

    if not body.regime_pass or body.confidence < 0.60 or body.model_call not in ("Up","Down"):
        return {"placed": False, "reason": "Policy block"}

    side = "buy" if body.model_call == "Up" else "sell"

    if TRADE_MODE.lower() == "equity":
        # simple proof: 1 share market order
        url = f"{ALPACA_TRADE_BASE}/v2/orders"
        payload = {
            "symbol": "SPY",
            "qty": "1",
            "side": side,
            "type": "market",
            "time_in_force": "day"
        }
        r = requests.post(url, headers=HDR_JSON, json=payload, timeout=20)
        if r.status_code == 401:
            raise HTTPException(502, "Alpaca auth failed (trade).")
        if r.status_code >= 300:
            raise HTTPException(502, f"Trade error: {r.text}")
        return {"placed": True, "mode": "equity", "alpaca": r.json()}

    elif TRADE_MODE.lower() == "options":
        # placeholder until we hook an options chain source:
        # We need a contract symbol like 'SPY241101C00450000' to place an options order.
        # Once we have an options data source, we’ll pick ATM 0DTE (or spread) & submit here:
        # url = f"{ALPACA_TRADE_BASE}/v2/options/orders"
        # payload = {...}
        # r = requests.post(url, headers=HDR_JSON, json=payload, timeout=20)
        return {"placed": False, "mode": "options", "reason": "options contract selection not wired yet"}

    else:
        raise HTTPException(400, "Unknown TRADE_MODE")
