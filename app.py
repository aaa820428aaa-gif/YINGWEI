# -*- coding: utf-8 -*-
# 單檔版：EMA(4h 趨勢) + Bollinger/RSI(1h) 雙向策略，每 INTERVAL_MIN 分鐘更新
import os, asyncio
from datetime import datetime
from typing import Any, Dict, List

import httpx
import pandas as pd
from fastapi import FastAPI, Response
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# ============== 參數與幣種 ==============
SYMBOLS = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","ADAUSDT","DOGEUSDT"]
INTERVAL_MIN = int(os.getenv("INTERVAL_MIN", "10"))  # 每 10 分鐘檢查一次
BINANCE_API = "https://api.binance.com/api/v3/klines"

# ============== 抓取 K 線 ==============
async def fetch_klines(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(BINANCE_API, params=params)
        r.raise_for_status()
        data = r.json()
    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","qv","n","tb","tbv","ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    df["close"] = df["close"].astype(float)
    return df

# ============== 技術指標 ==============
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))

def bollinger_bands(series: pd.Series, period: int = 20, mult: float = 2.0):
    mid = series.rolling(window=period).mean()
    std = series.rolling(window=period).std(ddof=0)
    upper = mid + mult * std
    lower = mid - mult * std
    return mid, upper, lower

# ============== 策略：EMA 趨勢 + BB/RSI 入場（雙向） ==============
async def analyze_symbol(symbol: str) -> List[Dict[str, Any]]:
    df4h = await fetch_klines(symbol, "4h", limit=400)
    df1h = await fetch_klines(symbol, "1h", limit=400)

    df4h["ema20"] = ema(df4h["close"], 20)
    df4h["ema50"] = ema(df4h["close"], 50)
    ema20_4h = df4h["ema20"].iloc[-1]
    ema50_4h = df4h["ema50"].iloc[-1]
    trend_long = ema20_4h > ema50_4h
    trend_short = ema20_4h < ema50_4h

    df1h["rsi14"] = rsi(df1h["close"], 14)
    mid, upper, lower = bollinger_bands(df1h["close"], 20, 2.0)
    df1h["bb_up"] = upper
    df1h["bb_lo"] = lower

    last = df1h.iloc[-1]
    prev = df1h.iloc[-2]
    price = float(last["close"])
    rsi_last = float(last["rsi14"])
    rsi_prev = float(prev["rsi14"])
    up = float(last["bb_up"])
    lo = float(last["bb_lo"])

    signals: List[Dict[str, Any]] = []

    # 多單條件
    if trend_long and (price <= lo * 1.002) and (rsi_prev < 35) and (rsi_last >= 40) and (rsi_last > rsi_prev):
        signals.append({
            "symbol": symbol, "side": "LONG", "entry": round(price, 6),
            "take_profit": round(price * 1.035, 6), "stop_loss": round(price * 0.985, 6),
            "rsi": round(rsi_last, 2), "trend": "BULL"
        })

    # 空單條件
    if trend_short and (price >= up * 0.998) and (rsi_prev > 65) and (rsi_last <= 60) and (rsi_last < rsi_prev):
        signals.append({
            "symbol": symbol, "side": "SHORT", "entry": round(price, 6),
            "take_profit": round(price * 0.965, 6), "stop_loss": round(price * 1.015, 6),
            "rsi": round(rsi_last, 2), "trend": "BEAR"
        })

    return signals

async def analyze_all(symbols: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for s in symbols:
        try:
            out.extend(await analyze_symbol(s))
        except Exception as e:
            print(f"analyze {s} failed: {e}")
    return out

# ============== FastAPI 主程式 ==============
app = FastAPI(title="EMA+BB Hybrid Monitor (10m)")
STATE: Dict[str, Any] = {"last_run": None, "signals": []}

async def job():
    signals = await analyze_all(SYMBOLS)
    STATE["signals"] = signals
    STATE["last_run"] = datetime.utcnow().isoformat() + "Z"
    print(f"Updated {STATE['last_run']} | signals={len(signals)}")

@app.on_event("startup")
async def on_startup():
    scheduler = AsyncIOScheduler()
    scheduler.add_job(job, "interval", minutes=INTERVAL_MIN, next_run_time=None)
    scheduler.start()
    asyncio.create_task(job())

@app.get("/health")
async def health():
    return {"status": "ok", "last_run": STATE["last_run"]}

@app.get("/report")
async def report():
    last = STATE["last_run"] or "尚未產生"
    sigs: List[Dict[str, Any]] = STATE["signals"] or []
    def card(sig: Dict[str, Any]) -> str:
        sym = sig["symbol"]
        tv = f"https://www.tradingview.com/chart/?symbol=BINANCE:{sym}"
        side = "多頭 (LONG)" if sig["side"] == "LONG" else "空頭 (SHORT)"
        return f"<div><b>{sym}</b>｜{side}｜進場 {sig['entry']}｜止盈 {sig['take_profit']}｜止損 {sig['stop_loss']}｜<a href='{tv}' target='_blank'>TradingView</a></div>"
    body = "".join(card(s) for s in sigs) or "<p>目前沒有符合條件的幣</p>"
    return Response(f"<h2>最新更新：{last}</h2>{body}", media_type="text/html")
