"""
Agent 5 - Mean Reversion.

Trades the fade: stretched price returns toward its mean.  Deliberately the
opposite of the trend agent, and gated so hard that it only speaks in ranges.

Hard gates (all must hold, or the agent stays silent)
-----------------------------------------------------
* ADX below 22          - no reversion trades inside a strong trend
* not near a breakout   - a level break is not a reversion setup
* price statistically stretched - beyond ~1.8 SD of its regression fit, or
  outside the Bollinger band, or RSI/Stoch extreme

Entry flavours
--------------
* band fade        : close outside the Bollinger band, RSI extreme
* channel fade     : regression-channel edge touched (>= 2 SD)
* VWAP fade        : far from rolling VWAP in a range

Confidence is highest when several independent stretch measures agree, and is
crushed by any sign of trend strength or volatility expansion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from agents.base import Agent, make_vote, safe_bool
from agents.helpers import binary, combine, graded, plate


class MeanReversionAgent(Agent):
    name = "mean_reversion"
    blurb = "Fades stretched price in ranges (BB/RSI/Stoch extremes, channel edges)"
    regime_tags = ("range",)
    styles = ("scalp", "intraday", "swing")

    def __init__(self, adx_max: float = 22.0, rsi_hi: float = 72.0,
                 rsi_lo: float = 28.0, chan_sd: float = 1.8):
        self.adx_max = adx_max
        self.rsi_hi = rsi_hi
        self.rsi_lo = rsi_lo
        self.chan_sd = chan_sd

    def score(self, fs, **ctx) -> "AgentVote":
        d = fs.df

        # ---------------- hard gates ---------------------------------------
        no_trend = d["adx"] < self.adx_max
        no_break = ~(safe_bool(d["break_res_50"]) | safe_bool(d["break_sup_50"]))
        vol_ok = (d["vol_regime"] > 0.5) & (d["vol_regime"] < 1.5)
        gate_all = no_trend & no_break & vol_ok

        # ---------------- stretch measures ---------------------------------
        rsi_hi_q = graded(d["rsi"], self.rsi_hi, 88)
        rsi_lo_q = graded(d["rsi"], self.rsi_lo, 12, invert=True)
        stoch_hi_q = graded(d["stoch_k"], 82, 97)
        stoch_lo_q = graded(d["stoch_k"], 18, 3, invert=True)
        chan_hi_q = graded(d["chan_pos"], self.chan_sd, 3.2)
        chan_lo_q = graded(d["chan_pos"], -self.chan_sd, -3.2, invert=True)
        bb_hi_q = graded(d["bb_pct"], 1.0, 1.06)
        bb_lo_q = graded(d["bb_pct"], 0.0, -0.06)
        vwap_hi_q = graded(d["vwap_dist"], 1.6, 3.5)
        vwap_lo_q = graded(d["vwap_dist"], -1.6, -3.5, invert=True)
        wick_hi_q = graded(d["upper_wick_pct"], 0.35, 0.7)
        wick_lo_q = graded(d["lower_wick_pct"], 0.35, 0.7)

        # ---------------- short (fade the high) ----------------------------
        stretch_hi = pd.concat([rsi_hi_q, stoch_hi_q, chan_hi_q, bb_hi_q,
                                vwap_hi_q, wick_hi_q], axis=1)
        n_hi = (stretch_hi > 0.35).sum(axis=1)
        short_sig = gate_all & (n_hi >= 3) & (d["close_loc"] < 0.5)

        # ---------------- long (fade the low) ------------------------------
        stretch_lo = pd.concat([rsi_lo_q, stoch_lo_q, chan_lo_q, bb_lo_q,
                                vwap_lo_q, wick_lo_q], axis=1)
        n_lo = (stretch_lo > 0.35).sum(axis=1)
        long_sig = gate_all & (n_lo >= 3) & (d["close_loc"] > 0.5)

        # ---------------- quality ------------------------------------------
        agree_hi = graded(n_hi, 3, 6)
        agree_lo = graded(n_lo, 3, 6)
        depth_hi = stretch_hi.max(axis=1)
        depth_lo = stretch_lo.max(axis=1)
        # calmer markets revert more reliably than violent ones
        calm = graded(d["vol_regime"], 1.35, 0.75, invert=True)
        range_q = graded(d["range_20_atr"], 6.0, 3.0, invert=True)
        # fading into a level is better than fading mid-range
        level_lo = safe_bool(d["at_major_sup"]).astype(float) * 0.15
        level_hi = safe_bool(d["at_major_res"]).astype(float) * 0.15

        q_short = combine(agree_hi, depth_hi, calm, range_q,
                          weights=[0.32, 0.3, 0.2, 0.18]) + level_hi
        q_long = combine(agree_lo, depth_lo, calm, range_q,
                         weights=[0.32, 0.3, 0.2, 0.18]) + level_lo

        # reversion into a strong opposite trend is a trap - cut confidence
        q_short = q_short.where(~safe_bool(d["ema_stack_bear"]), q_short * 0.6)
        q_long = q_long.where(~safe_bool(d["ema_stack_bull"]), q_long * 0.6)

        long_score = binary(long_sig, q_long)
        short_score = binary(short_sig, q_short)

        conf = pd.concat([q_long, q_short], axis=1).max(axis=1) * gate_all.astype(float)
        return make_vote(self, long_score, short_score, conf)
