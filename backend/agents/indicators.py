"""
Agent 2 - Indicator-Based Trading.

The classical oscillator/confluence voice: RSI, MACD, Stochastic, CCI, Williams
%R, Bollinger position and MFI all vote, and the agent only fires when they
*agree*.  A lone indicator never triggers a trade here - that is what separates
this from naive indicator systems.

Two modes
---------
* trend-continuation (default): indicators confirm an existing EMA-stack trend
* reversion: RSI/Stoch extremes away from the bands when the regime is ranging

The mode is chosen per bar from the regime, so the agent adapts instead of
fighting the market.
"""

from __future__ import annotations

import pandas as pd

from agents.base import Agent, make_vote, safe_bool
from agents.helpers import binary, combine, crossing_down, crossing_up, graded


class IndicatorAgent(Agent):
    name = "indicators"
    blurb = "RSI/MACD/Stoch/CCI/BB/MFI confluence, trend or reversion mode"
    regime_tags = ("trend", "range")
    styles = ("scalp", "intraday", "swing", "position")

    def __init__(self, rsi_buy=55, rsi_sell=45, min_agree=3, adx_gate=20):
        self.rsi_buy = rsi_buy
        self.rsi_sell = rsi_sell
        self.min_agree = min_agree
        self.adx_gate = adx_gate

    def score(self, fs, **ctx) -> "AgentVote":
        d = fs.df

        # ---------------- trend-continuation votes -------------------------
        trend_long = (
            safe_bool(d["ema_stack_bull"]) | (d["close"] > d["ema20"])
        ) & (
            (d["macd_hist"] > 0) | crossing_up(d["macd"], d["macd_sig"])
        ) & (
            (d["rsi"] > self.rsi_buy) | (d["rsi"] > d["rsi"].shift(1))
        ) & (d["adx"] > self.adx_gate) & (d["plus_di"] > d["minus_di"])

        trend_short = (
            safe_bool(d["ema_stack_bear"]) | (d["close"] < d["ema20"])
        ) & (
            (d["macd_hist"] < 0) | crossing_down(d["macd"], d["macd_sig"])
        ) & (
            (d["rsi"] < self.rsi_sell) | (d["rsi"] < d["rsi"].shift(1))
        ) & (d["adx"] > self.adx_gate) & (d["minus_di"] > d["plus_di"])

        # ---------------- reversion votes (ranging markets) -----------------
        rev_long = (
            (d["rsi"] < 32) | (d["stoch_k"] < 20) | (d["willr"] < -85)
        ) & (d["close"] < d["bb_low"] * 1.002) & (d["adx"] < 22)

        rev_short = (
            (d["rsi"] > 68) | (d["stoch_k"] > 80) | (d["willr"] > -15)
        ) & (d["close"] > d["bb_up"] * 0.998) & (d["adx"] < 22)

        # ---------------- count how many indicators actually agree -----------
        agree_long = (
            (d["rsi"] > 50).astype(int) + (d["macd_hist"] > 0).astype(int) +
            (d["stoch_k"] > d["stoch_d"]).astype(int) +
            (d["close"] > d["bb_mid"]).astype(int) +
            (d["cci"] > 0).astype(int) + (d["mfi"] > 50).astype(int) +
            (d["close"] > d["ema50"]).astype(int)
        )
        agree_short = (
            (d["rsi"] < 50).astype(int) + (d["macd_hist"] < 0).astype(int) +
            (d["stoch_k"] < d["stoch_d"]).astype(int) +
            (d["close"] < d["bb_mid"]).astype(int) +
            (d["cci"] < 0).astype(int) + (d["mfi"] < 50).astype(int) +
            (d["close"] < d["ema50"]).astype(int)
        )
        confluence_ok_long = agree_long >= self.min_agree
        confluence_ok_short = agree_short >= self.min_agree

        long_sig = (trend_long | rev_long) & confluence_ok_long
        short_sig = (trend_short | rev_short) & confluence_ok_short

        # ---------------- quality -------------------------------------------
        agree_q = graded(agree_long.clip(upper=7) / 7.0, self.min_agree / 7.0, 0.95)
        agree_q_s = graded(agree_short.clip(upper=7) / 7.0, self.min_agree / 7.0, 0.95)
        trend_q = graded(d["adx"], self.adx_gate, 40)
        macd_q = graded(d["macd_hist"].abs() / d["atr"], 0.02, 0.15)
        bb_q = graded(d["bb_width"] / d["bb_width"].rolling(50, min_periods=10).mean(), 0.8, 1.6)
        extreme_q = pd.concat([
            graded(d["rsi"], 28, 45, invert=True),      # oversold -> high score
            graded(d["rsi"], 55, 72),                   # overbought -> high score
        ], axis=1).max(axis=1)

        q_long = combine(agree_q, trend_q, macd_q, bb_q,
                         weights=[0.4, 0.25, 0.2, 0.15])
        q_short = combine(agree_q_s, trend_q, macd_q, bb_q,
                          weights=[0.4, 0.25, 0.2, 0.15])
        # reversion setups lean on the extreme reading instead of ADX
        q_long = q_long.where(~safe_bool(rev_long), combine(extreme_q, agree_q, weights=[0.6, 0.4]))
        q_short = q_short.where(~safe_bool(rev_short), combine(extreme_q, agree_q_s, weights=[0.6, 0.4]))

        long_score = binary(long_sig, q_long)
        short_score = binary(short_sig, q_short)

        # indicators fail in dead volatility - damp confidence there
        vol_q = graded(d["vol_regime"], 0.55, 1.0)
        conf = pd.concat([q_long, q_short], axis=1).max(axis=1) * (0.55 + 0.45 * vol_q)

        return make_vote(self, long_score, short_score, conf)
