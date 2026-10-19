"""
signal_engine.py - The signal engine that runs on GitHub Actions.

Loads trained model -> fetches live Binance candles -> builds features
-> scores with 9-agent orchestrator -> checks filters -> Telegram + order.

NO code or logic changes - same modules as the local platform.
"""

from __future__ import annotations
import hashlib, hmac, json, math, os, pickle, sys, time
import urllib.request, urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
BACKEND = HERE.parent / "backend"
sys.path.insert(0, str(BACKEND))

STATE_FILE = HERE / "state.json"

# mild-filter params (same as validated ETH setup)
MIN_SCORE = 0.50
MIN_AGREEMENT = 4
ATR_MULT = 1.2
SPREAD = 0.30
FEE_RATE = 0.0002
MAX_HOLD_BARS = 48
LONDON_START_UTC = 7
LONDON_END_UTC = 16
RR_PANELS = {"1": 1.0, "2": 2.0, "3": 3.0}
BE_AFTER_R = 1.0
RISK_USDT = float(os.environ.get("RISK_USDT", "10"))
MAX_NOTIONAL = float(os.environ.get("MAX_NOTIONAL", "100"))

BINANCE_KEY = os.environ.get("BINANCE_KEY", "")
BINANCE_SECRET = os.environ.get("BINANCE_SECRET", "")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
DRY_RUN = os.environ.get("DRY_RUN", "true").lower() == "true"

BINANCE_BASE = "https://fapi.binance.com"


def _get(url, params=None, timeout=15):
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _signed(method, path, params=None, timeout=15):
    params = dict(params or {})
    params["timestamp"] = int(time.time() * 1000)
    params["recvWindow"] = 10000
    qs = urllib.parse.urlencode(sorted(params.items()))
    sig = hmac.new(BINANCE_SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()
    url = f"{BINANCE_BASE}{path}?{qs}&signature={sig}"
    req = urllib.request.Request(url, method=method, headers={
        "X-MBX-APIKEY": BINANCE_KEY, "User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def fetch_klines(interval, limit=500):
    rows = _get(f"{BINANCE_BASE}/fapi/v1/klines",
                {"symbol": "ETHUSDT", "interval": interval, "limit": limit})
    data = [{"open": float(r[1]), "high": float(r[2]), "low": float(r[3]),
             "close": float(r[4]), "volume": float(r[5])} for r in rows]
    df = pd.DataFrame(data, index=pd.to_datetime(
        [r[0] for r in rows], unit="ms", utc=True))
    df.index.name = "datetime"
    return df


def get_eth_price():
    t = _get(f"{BINANCE_BASE}/fapi/v1/ticker/price", {"symbol": "ETHUSDT"})
    return float(t["price"])


def place_order(side, qty, reduce_only=False):
    params = {"symbol": "ETHUSDT", "side": side.upper(),
              "type": "MARKET", "quantity": f"{qty:.3f}"}
    if reduce_only:
        params["reduceOnly"] = "true"
    return _signed("POST", "/fapi/v1/order", params)


def tg(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = json.dumps({"chat_id": TELEGRAM_CHAT_ID, "text": text}).encode()
        req = urllib.request.Request(url, data=payload,
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"[tg] error: {e}")


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"panels": {"1": None, "2": None, "3": None},
            "journal": []}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=1, default=str))


def calc_qty(price, risk_dist):
    if risk_dist <= 0:
        return 0.0
    qty = RISK_USDT / risk_dist
    if qty * price > MAX_NOTIONAL:
        qty = MAX_NOTIONAL / price
    return round(qty, 3)


def load_model():
    from agents import make_agents
    from orchestrator import OrchestratorAgent
    with open(HERE / "model_weights.pkl", "rb") as f:
        w = pickle.load(f)
    agents = make_agents()
    orch = OrchestratorAgent(
        agents, min_agreement=w["min_agreement"],
        min_score=w["min_score"], conflict_gap=w["conflict_gap"],
        use_learned_weights=w["use_learned_weights"],
        htf_weight=w["htf_weight"])
    orch.agent_weights = w["agent_weights"]
    orch.agent_regime_weights = w["agent_regime_weights"]
    orch.session_weights = w["session_weights"]
    orch.weight_source = "loaded from model_weights.pkl"
    return orch, w


def score_bar(orch, m5, h1):
    from features import build_features, merge_higher_tf
    from regime import classify
    fs = build_features("5min", m5)
    hdf = build_features("1h", h1)
    fs = merge_higher_tf(fs, hdf)
    labels = classify(fs)
    dec = orch.score(fs, labels=labels)
    i = len(fs.df) - 1
    return {
        "side": int(dec.side.iloc[i]),
        "score": float(dec.score.iloc[i]),
        "agreement": int(dec.agreement.iloc[i]),
        "conflict": bool(dec.conflict.iloc[i]) if hasattr(dec, "conflict") else False,
        "atr": float(fs.df["atr"].iloc[i]),
        "close": float(fs.df["close"].iloc[i]),
    }


def main():
    now = datetime.now(timezone.utc)
    hour, weekday = now.hour, now.weekday()
    print(f"[engine] {now.isoformat()}  hour={hour}  weekday={weekday}  DRY_RUN={DRY_RUN}")

    orch, w = load_model()
    print(f"[engine] model: {len(w['agent_weights'])} agent weights loaded")

    state = load_state()
    price = get_eth_price()
    print(f"[engine] ETH: {price:.2f}")

    # -- manage open positions --
    for pp, pos in state["panels"].items():
        if pos is None:
            continue
        risk, entry, side = pos["risk"], pos["entry"], pos["side"]
        be, bars = pos.get("breakeven", False), pos.get("bars_held", 0)

        if not be:
            if side == 1 and price >= entry + risk * BE_AFTER_R:
                pos["stop"] = entry; pos["breakeven"] = True
                tg(f"BREAKEVEN @ entry - risk $0  Panel {pp}  ETH {price:.2f}")
            elif side == -1 and price <= entry - risk * BE_AFTER_R:
                pos["stop"] = entry; pos["breakeven"] = True
                tg(f"BREAKEVEN @ entry - risk $0  Panel {pp}  ETH {price:.2f}")

        exit_p, reason = None, None
        if side == 1:
            if price <= pos["stop"]: exit_p, reason = pos["stop"], "stop"
            for pp2 in RR_PANELS:
                tp = pos.get(f"target_{pp2}")
                if tp and price >= tp: exit_p, reason = tp, f"target_{pp2}"; break
        else:
            if price >= pos["stop"]: exit_p, reason = pos["stop"], "stop"
            for pp2 in RR_PANELS:
                tp = pos.get(f"target_{pp2}")
                if tp and price <= tp: exit_p, reason = tp, f"target_{pp2}"; break

        bars += 1; pos["bars_held"] = bars
        if exit_p is None and bars >= MAX_HOLD_BARS:
            exit_p, reason = price, "timeout"

        if exit_p is not None:
            R = (exit_p - entry) * side / risk
            R -= (exit_p * FEE_RATE * 2) / risk
            close_msg = ""
            if not DRY_RUN and BINANCE_KEY:
                try:
                    cs = "SELL" if side == 1 else "BUY"
                    resp = place_order(cs, pos.get("qty", 0.001), reduce_only=True)
                    close_msg = f"  #{resp.get('orderId','?')}"
                except Exception as e:
                    close_msg = f"  err={e}"
            label = "TARGET" if "target" in reason else "STOP" if reason=="stop" else "TIMEOUT"
            tg(f"{label} HIT  Panel {pp}  ETH {side}  {entry:.2f}->{exit_p:.2f}  R={R:+.2f}  {reason}{close_msg}")
            state["journal"].append({
                "time": now.isoformat(), "panel": pp,
                "side": "LONG" if side==1 else "SHORT",
                "entry": entry, "exit": exit_p, "stop": pos["stop"],
                "R": round(R,3), "reason": reason, "bars": bars})
            state["panels"][pp] = None

    # -- score latest bar --
    in_london = LONDON_START_UTC <= hour < LONDON_END_UTC and weekday < 5
    print(f"[engine] london={in_london}")

    if in_london:
        m5 = fetch_klines("5m", 400)
        h1 = fetch_klines("1h", 100)
        dec = score_bar(orch, m5, h1)
        print(f"[engine] side={dec['side']} score={dec['score']:.3f} agree={dec['agreement']}")

        sig = dec["side"]
        if dec["score"] < MIN_SCORE or dec["agreement"] < MIN_AGREEMENT or dec["conflict"]:
            sig = 0
        print(f"[engine] after filter: {sig}")

        if sig != 0:
            atr = dec["atr"]
            risk_dist = ATR_MULT * atr
            fill_adj = dec["close"] + dec["close"] * FEE_RATE * sig
            fill_adj = dec["close"] + dec["close"] * FEE_RATE * sig
            qty = calc_qty(fill_adj, risk_dist)

            lines = [f"SIGNAL  ETH  {'LONG' if sig==1 else 'SHORT'}  @ {fill_adj:.2f}"]
            for pp, tp_R in RR_PANELS.items():
                if state["panels"].get(pp) is not None:
                    lines.append(f"  Panel {pp}: SKIPPED (in trade)"); continue
                stop = fill_adj - sig * risk_dist
                target = fill_adj + sig * risk_dist * tp_R
                om = ""
                if not DRY_RUN and BINANCE_KEY:
                    try:
                        ss = "BUY" if sig == 1 else "SELL"
                        resp = place_order(ss, qty)
                        om = f"  #{resp.get('orderId','?')}"
                    except Exception as e:
                        om = f"  err={e}"
                state["panels"][pp] = {
                    "side": sig, "entry": round(fill_adj,2),
                    "stop": round(stop,2),
                    "target_1": round(fill_adj+sig*risk_dist*1,2),
                    "target_2": round(fill_adj+sig*risk_dist*2,2),
                    "target_3": round(fill_adj+sig*risk_dist*3,2),
                    "risk": round(risk_dist,4), "qty": qty,
                    "breakeven": False, "bars_held": 0,
                    "opened_at": now.isoformat()}
                lines.append(f"  Panel {pp} ({tp_R:.0f}R): stop {stop:.2f}  target {target:.2f}{om}")

            lines.append(f"ATR={atr:.4f}  risk={risk_dist:.4f}  qty={qty}  score={dec['score']:.2f}  agree={dec['agreement']}/9")
            tg("\n".join(lines))
            print("[engine] SIGNAL SENT")

    save_state(state)
    open_p = sum(1 for v in state["panels"].values() if v)
    print(f"[engine] done. {open_p} panels open. journal={len(state['journal'])} entries")


if __name__ == "__main__":
    main()
