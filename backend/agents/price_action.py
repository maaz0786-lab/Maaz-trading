"""
Agent 1 - Price Action Trading.

Trades raw market structure: confirmed swing pivots, break of structure (BOS),
change of character (CHoCH), and pin-bar / rejection wicks at those levels.
No indicators except ATR for normalisation - this agent is deliberately the
"pure price" voice on the panel.

Rules it votes on
-----------------
BULLISH
  * break of structure up  : close takes out the last confirmed swing high
  * pullback into demand    : price returns to the prior swing-low zone after a
                              BOS up, then rejects with a lower wick
  * bullish pin bar         : long lower wick + close in the top third
BEARISH - the mirror image.

Confidence scales with wick rejection, close location in the bar, ADX support
and whether the higher timeframe agrees.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from agents.base import Agent, make_vote, safe_bool
from agents.helpers import binary, combine, graded


class PriceActionAgent(Agent):
    name = "price_action"
    blurb = "Swing structure, break of structure, rejection wicks at pivots"
    regime_tags = ("trend", "pullback", "reversal")
    styles = ("scalp", "intraday", "swing", "position")

    def __init__(self, wick_min: float = 0.45, close_loc_min: float = 0.6,
                 max_dist_atr: float = 1.2):
        self.wick_min = wick_min
        self.close_loc_min = close_loc_min
        self.max_dist_atr = max_dist_atr

    def score(self, fs, **ctx) -> "AgentVote":
        d = fs.df
        idx = d.index

        # ---- break of structure -------------------------------------------
        bos_up = (d["close"] > d["last_swing_high"]) & \
                 (d["close"].shift(1) <= d["last_swing_high"].shift(1))
        bos_dn = (d["close"] < d["last_swing_low"]) & \
                 (d["close"].shift(1) >= d["last_swing_low"].shift(1))

        # ---- pullback into the prior swing after a BOS --------------------
        # a higher low was made recently and price has come back near it
        near_demand = d["dist_to_swing_lo"].between(0.0, self.max_dist_atr)
        near_supply = d["dist_to_swing_hi"].between(0.0, self.max_dist_atr)
        recent_bos_up = bos_up.rolling(12).sum() > 0
        recent_bos_dn = bos_dn.rolling(12).sum() > 0
        pullback_long = recent_bos_up & near_demand
        pullback_short = recent_bos_dn & near_supply

        # ---- rejection wicks ----------------------------------------------
        bull_wick = (d["lower_wick_pct"] > self.wick_min) & \
                    (d["close_loc"] > self.close_loc_min)
        bear_wick = (d["upper_wick_pct"] > self.wick_min) & \
                    (d["close_loc"] < 1 - self.close_loc_min)

        # ---- liquidity sweeps reversed ------------------------------------
        sweep_long = safe_bool(d["sweep_low"])
        sweep_short = safe_bool(d["sweep_high"])

        # ---- combine -------------------------------------------------------
        long_sig = bos_up | (pullback_long & bull_wick) | (sweep_long & bull_wick)
        short_sig = bos_dn | (pullback_short & bear_wick) | (sweep_short & bear_wick)

        # ---- graded quality -------------------------------------------------
        wick_q = pd.concat([d["lower_wick_pct"], d["upper_wick_pct"]], axis=1).max(axis=1)
        wick_q = graded(wick_q, self.wick_min, 0.75)
        loc_raw = np.maximum(d["close_loc"], 1 - d["close_loc"])
        loc_q = graded(pd.Series(loc_raw, index=idx), self.close_loc_min, 0.95)
        body_q = graded(d["body_pct"], 0.35, 0.7)
        trend_q = graded(d["adx"], 15, 35)
        htf_q = _htf_agreement(d)

        quality = combine(wick_q, loc_q, body_q, trend_q, htf_q,
                          weights=[0.30, 0.25, 0.15, 0.15, 0.15])

        long_score = binary(long_sig, quality)
        short_score = binary(short_sig, quality)

        # price action needs a real trend or a real sweep; flat chop is ignored
        tradable = (~safe_bool(d["trendline_flat"]) |
                    safe_bool(d["sweep_low"]) | safe_bool(d["sweep_high"]))
        conf = quality * (0.6 + 0.4 * trend_q) * tradable.astype(float)

        return make_vote(self, long_score, short_score, conf)


def _htf_agreement(d: pd.DataFrame) -> pd.Series:
    """1 when higher timeframes all agree with the current side, 0.5 when split."""
    cols = [c for c in d.columns if c.startswith("htf_") and c.endswith("_ema200")]
    if not cols:
        return pd.Series(0.5, index=d.index)
    above = pd.concat([(d["close"] > d[c]).astype(float) for c in cols], axis=1).mean(axis=1)
    # 1.0 (all above) or 0.0 (all below) -> strong; 0.5 -> conflict
    return (above - 0.5).abs() * 2.0
