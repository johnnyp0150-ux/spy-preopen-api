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

# -------------------- Config & Globals --------------------
APP_STARTED_AT = time.time()

ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
API_TOKEN         = os.getenv("API_TOKEN", "")  # optional Bearer check

# Alpaca endpoints (paper by default)
ALPACA_TRADE_BASE = os.getenv("ALPACA_TRADE_BASE", "https://paper-api.alpaca.markets")
ALPACA_DATA_BASE  = os.getenv("ALPACA_DATA_BASE",  "https://data.alpaca.markets")

TRADE_BASE_STOCKS = f"{ALPACA_DATA_BASE}/v2"  # for latest trade fallback
HDR = {
    "APCA-API-KEY-ID": ALPACA_API_KEY or "",
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY or "",
}

# Notebook thresholds
UP_THR   =  0.0035
DOWN_THR = -0.0035

# Regime gates
REG_MIN_ABS_GAP = float(os.getenv("REG_MIN_ABS_GAP", "0.0020"))  # 0.20%
REG_MIN_VIX_PCT = float(os.getenv("REG_MIN_VIX_PCT", "0.0200"))  # 2% (magnitude, direction-aware below)
REG_MIN_ATR_PCT = float(os.getenv("REG_MIN_ATR_PCT", "0.0080"))  # 0.80%

# In-memory caches
_HIST_DF: Optional[pd.DataFrame] = None
_HIST_TS: float = 0.0
_CLF = None
_FEAT_NAMES = None
_MODEL_TS: float = 0.0

REFRESH_SECS = 12 * 3600

# Fallback flags (reported in responses)
_LAST_HISTORY_FALLBACK = False
_LAST_MODEL_FALLBACK = False

app = FastAPI(title="SPY Preopen API", version="0.4.0")


# -------------------- Schemas --------------------
class PredictIn(BaseModel):
    test_value: Optional[int] = None

class TradeIn(BaseModel):
    # This endpoint does NOT place options directly; it prepares a clean payload
    # and (optionally) hits your TRADE_WEBHOOK_URL so your workflow can place the order.
    # If you flip ENABLE_EQUITY_PROOF in Actions, it can submit a 1-share SPY paper trade via Alpaca as a sanity check.
    model_call: str
    confidence: float
    features: Dict[str, float]
    regime: Dict[str, Any]
    regime_pass: bool


# -------------------- Health --------------------
@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - APP_STARTED_AT, 2),
        "model_cached": _CLF is not None,
        "history_cached": _HIST_DF is not None,
        "version": "0.4.0",
    }

@app.get("/health")
def health():
    return healthz()


# -------------------- Helpers --------------------
def _need_refresh(ts: float) -> bool:
    return (time.time() - ts) > REFRESH_SECS


def _have_alpaca_keys() -> bool:
    return bool(ALPACA_API_KEY and ALPACA_SECRET_KEY)


def latest_trade(symbol: str) -> float:
    """
    Latest trade:
      • Use Alpaca (if keys present) via /v2/stocks/{symbol}/trades/latest
      • Else fallback to yfinance 1-minute intraday
    """
    if _have_alpaca_keys():
        url = f"{TRADE_BASE_STOCKS}/stocks/{symbol}/trades/latest"
        r = requests.get(url, headers=HDR, timeout=20)
        if r.status_code == 401:
            raise HTTPException(502, "Alpaca auth failed (check keys)")
        r.raise_for_status()
        return float(r.json()["trade"]["p"])

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
    es  = yf.download("ES=F",  start="2012-01-01", auto_adjust=False, progress=False)

    if spy.empty or vix.empty or es.empty:
        now = pd.Timestamp.utcnow().floor("D")
        df = pd.DataFrame(
            {
                "Open":       [500.0, 501.0],
                "High":       [502.0, 503.0],
                "Low":        [498.0, 500.0],
                "Close":      [501.0, 502.0],
                "VIX_Close":  [15.0,  15.1],
                "ES_Close":   [5200.0, 5205.0],
            },
            index=[now - pd.Timedelta(days=1), now],
        )
        _LAST_HISTORY_FALLBACK = True
    else:
        spy = spy.rename(columns=str.title)
        df = pd.DataFrame(index=spy.index)
        df["Open"]      = spy["Open"]
        df["High"]      = spy["High"]
        df["Low"]       = spy["Low"]
        df["Close"]     = spy["Close"]
        df["VIX_Close"] = vix["Close"].reindex(df.index).ffill()
        df["ES_Close"]  = es["Close"].reindex(df.index).ffill()

    # Features
    df["ES_overnight_ret"]  = df["ES_Close"].pct_change()
    df["VIX_overnight_ret"] = df["VIX_Close"].pct_change()
    df["Premarket_Gap"]     = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)

    # Labels
    oc_ret = (df["Close"] / df["Close"].shift(1)) - 1.0
    df["label"] = np.where(oc_ret >= UP_THR, "Up",
                   np.where(oc_ret <= DOWN_THR, "Down", "Sideways"))

    # ATR(10) %
    tr = np.maximum(df["High"] - df["Low"],
         np.maximum(abs(df["High"] - df["Close"].shift(1)), abs(df["Low"] - df["Close"].shift(1))))
    df["ATR10"] = pd.Series(tr).rolling(10).mean()
    df["ATR10_pct"] = df["ATR10"] / df["Close"].shift(1)

    df = df.dropna(subset=["ES_overnight_ret", "VIX_overnight_ret", "Premarket_Gap", "ATR10_pct"])

    if df.empty:
        now = pd.Timestamp.utcnow().floor("D")
        df = pd.DataFrame(
            {
                "Open":              [500.0, 501.0],
                "High":              [502.0, 503.0],
                "Low":               [498.0, 500.0],
                "Close":             [501.0, 502.0],
                "VIX_Close":         [15.0,  15.1],
                "ES_Close":          [5200.0, 5205.0],
                "ES_overnight_ret":  [0.0,  (5205/5200 - 1.0)],
                "VIX_overnight_ret": [0.0,  (15.1/15.0 - 1.0)],
                "Premarket_Gap":     [0.0,  0.0],
                "ATR10_pct":         [0.01, 0.01],
                "label":             ["Sideways", "Sideways"],
            },
            index=[now - pd.Timedelta(days=1), now],
        )
        _LAST_HISTORY_FALLBACK = True

    _HIST_DF, _HIST_TS = df, time.time()
    return df


def train_quick(df: pd.DataFrame):
    global _CLF, _FEAT_NAMES, _MODEL_TS, _LAST_MODEL_FALLBACK
    _LAST_MODEL_FALLBACK = False

    if _CLF is not None and _FEAT_NAMES is not None and not _need_refresh(_MODEL_TS):
        return _CLF, _FEAT_NAMES

    X = df[["ES_overnight_ret", "VIX_overnight_ret", "Premarket_Gap"]].copy()
    y = df["label"].copy()

    if len(X) < 3:
        from sklearn.dummy import DummyClassifier
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(np.zeros((1, 3)), ["Sideways"])
        _CLF, _FEAT_NAMES, _MODEL_TS = dummy, X.columns.tolist(), time.time()
        _LAST_MODEL_FALLBACK = True
        return _CLF, _FEAT_NAMES

    n_splits = max(2, min(6, len(X) - 1))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    last_clf = None
    for tr, te in tscv.split(X):
        Xtr, ytr = X.iloc[tr], y.iloc[tr]
        last_clf = GradientBoostingClassifier(random_state=42)
        last_clf.fit(Xtr, ytr)

    _CLF, _FEAT_NAMES, _MODEL_TS = last_clf, X.columns.tolist(), time.time()
    return _CLF, _FEAT_NAMES


# -------------------- Routes --------------------
@app.get("/")
def root():
    return {"ok": True, "msg": "See /docs for interactive API. Health: /healthz"}


@app.post("/predict")
def predict(payload: PredictIn, authorization: Optional[str] = Header(None)):
    if API_TOKEN:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(401, "Missing Bearer token")
        if authorization.split(" ",1)[1] != API_TOKEN:
            raise HTTPException(403, "Invalid token")

    try:
        df_hist = build_history()
        clf, feat_names = train_quick(df_hist)

        spy_last  = latest_trade("SPY")
        vixy_last = latest_trade("VIXY")  # cheap proxy for “VIX-ish” change when keys absent
        spy_yday  = prev_close("SPY")
        vixy_yday = prev_close("VIXY")

        premarket_gap  = (spy_last / spy_yday) - 1.0
        vix_overnight  = (vixy_last / vixy_yday) - 1.0
        es_overnight   = float(df_hist["ES_overnight_ret"].iloc[-1])

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

        # Regime gates
        atr_pct = float(df_hist["ATR10_pct"].iloc[-1])
        gap_abs = abs(float(premarket_gap))
        vix_pct = float(vix_overnight)

        vix_ok_long  = (vix_pct <= 0.0) or (abs(vix_pct) >= REG_MIN_VIX_PCT)
        vix_ok_short = (vix_pct >= 0.0) or (abs(vix_pct) >= REG_MIN_VIX_PCT)
        vix_ok = vix_ok_long if pred == "Up" else (vix_ok_short if pred == "Down" else False)

        regime = {
            "gap_abs": round(gap_abs, 6),
            "atr10_pct": round(atr_pct, 6),
            "vix_pct": round(vix_pct, 6),
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
            "confidence": round(p, 4),
            "features": {
                "premarket_gap": round(float(premarket_gap), 6),
                "vix_overnight_ret": round(float(vix_overnight), 6),
                "es_overnight_ret": round(float(es_overnight), 6),
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


@app.post("/trade")
def trade(body: TradeIn, authorization: Optional[str] = Header(None)):
    """
    Purpose:
      - Central place your cron (or you) can hit AFTER /predict.
      - Emits a normalized JSON payload for your trading webhook.
      - Optionally can place a tiny 1-share SPY paper order as a "proof" path (toggle in Actions).
    """
    if API_TOKEN:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(401, "Missing Bearer token")
        if authorization.split(" ",1)[1] != API_TOKEN:
            raise HTTPException(403, "Invalid token")

    try:
        payload = {
            "ts": pd.Timestamp.utcnow().isoformat(),
            "source": "preopen-trade-endpoint",
            "model_call": body.model_call,
            "confidence": body.confidence,
            "features": body.features,
            "regime": body.regime,
            "regime_pass": body.regime_pass,
        }

        # This endpoint itself doesn’t place options orders directly (we keep that in the GitHub Action via webhook),
        # but we return a clean payload so your workflow (Zapier/Make/Cloud Function) can place an OPTIONS spread.
        return {"ok": True, "ready_for_webhook": True, "payload": payload}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
