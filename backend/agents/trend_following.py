"""
Agent 4 - Trend Following.

Classic "trade with the trend, buy pullbacks in an uptrend" logic, tightened so
it is actually testable:

* regime gate    : ADX above threshold and EMA stack aligned in the same
                   direction (no counter-trend guessing)
* entry          : pullback into the EMA9/EMA20 zone while the trend is intact,
                   or a Donchian-style new N-bar extreme in trend direction
* continuation   : MACD histogram turning back in the trend's favour
* trail logic    : not handled here - the risk manager does that

This agent is the panel's "stay long strong markets / stay short weak markets"
voice and is the one most likely to be useful for swing and position trading.
"""

from __future__ import annotations

import pandas as pd

from agents.base import Agent, make_vote, safe_bool
from agents.helpers import binary, combine, graded


class TrendFollowingAgent(Agent):
    name = "trend_following"
    blurb = "EMA-stack trends, pullbacks to EMA9/20, Donchian continuation"
    regime_tags = ("trend", "breakout")
    styles = ("intraday", "swing", "position")

    def __init__(self, adx_min: float = 22.0, pullback_atr: float = 0.8,
                 donchian: int = 20):
        self.adx_min = adx_min
        self.pullback_atr = pullback_atr
        self.donchian = donchian

    def score(self, fs, **ctx) -> "AgentVote":
        d = fs.df
        adx_ok = d["adx"] > self.adx_min

        # ---------------- trend definition ---------------------------------
        bull_trend = adx_ok & ((d["ema20"] > d["ema50"]) & (d["ema50"] > d["ema200"])) & \
                     (d["ema200_slope"] >= 0)
        bear_trend = adx_ok & ((d["ema20"] < d["ema50"]) & (d["ema50"] < d["ema200"])) & \
                     (d["ema200_slope"] <= 0)
        # softer trend definition for lower timeframes / earlier trends
        bull_trend |= adx_ok & (d["close"] > d["ema20"]) & (d["ema20"] > d["ema50"]) & \
                      safe_bool(d["trendline_bull"])
        bear_trend |= adx_ok & (d["close"] < d["ema20"]) & (d["ema20"] < d["ema50"]) & \
                      safe_bool(d["trendline_bear"])

        # ---------------- pullback into the value zone ---------------------
        # price pulled back near EMA9/20 but has not broken EMA50
        near_ema = (d["close"] - d["ema20"]).abs() < self.pullback_atr * d["atr"]
        pb_long = bull_trend & near_ema & \
                  (d["close"] > d["ema50"]) & (d["low"] <= d["ema20"] + 0.25 * d["atr"])
        pb_short = bear_trend & near_ema & \
                   (d["close"] < d["ema50"]) & (d["high"] >= d["ema20"] - 0.25 * d["atr"])

        # ---------------- Donchian continuation ----------------------------
        n = self.donchian
        dc_up = (d["close"] >= d["high"].rolling(n, min_periods=n // 2).max().shift(1))
        dc_dn = (d["close"] <= d["low"].rolling(n, min_periods=n // 2).min().shift(1))
        cont_long = bull_trend & dc_up & (d["vol_ratio"] > 1.1)
        cont_short = bear_trend & dc_dn & (d["vol_ratio"] > 1.1)

        # ---------------- MACD re-acceleration -----------------------------
        macd_long = bull_trend & (d["macd_hist"] > d["macd_hist"].shift(1)) & (d["macd_hist"] > 0)
        macd_short = bear_trend & (d["macd_hist"] < d["macd_hist"].shift(1)) & (d["macd_hist"] < 0)

        long_sig = pb_long | cont_long | macd_long
        short_sig = pb_short | cont_short | macd_short

        # ---------------- quality ------------------------------------------
        adx_q = graded(d["adx"], self.adx_min, 45)
        sep_q = graded((d["ema20"] - d["ema50"]).abs() / d["atr"], 0.3, 2.0)
        slope_q = graded(d["close_regslope_atr"].abs(), 0.2, 1.2)
        vol_q = graded(d["vol_regime"], 0.7, 1.5)
        htf_q = _htf_trend(d)
        room_q = pd.concat([graded(d["room_up_atr"], 1.0, 4.0),
                            graded(d["room_dn_atr"], 1.0, 4.0)], axis=1).max(axis=1)

        quality = combine(adx_q, sep_q, slope_q, vol_q, htf_q,
                          weights=[0.28, 0.18, 0.18, 0.14, 0.22])

        # pullback entries are higher quality than chasing a Donchian break
        quality = quality.where(~(safe_bool(pb_long) | safe_bool(pb_short)),
                                (quality * 0.85 + 0.15 * room_q).clip(0, 1))

        long_score = binary(long_sig, quality)
        short_score = binary(short_sig, quality)

        # no trend -> no vote at all, so confidence collapses outside trends
        in_trend = safe_bool(bull_trend) | safe_bool(bear_trend)
        conf = quality * in_trend.astype(float)
        return make_vote(self, long_score, short_score, conf)


def _htf_trend(d: pd.DataFrame) -> pd.Series:
    """Higher-timeframe trend agreement in [0,1]; 0.5 when unavailable."""
    cols = [c for c in d.columns if c.startswith("htf_") and c.endswith("_adx")]
    if not cols:
        return pd.Series(0.5, index=d.index)
    best = pd.concat([graded(d[c], 20, 40) for c in cols], axis=1).max(axis=1)
    return best
