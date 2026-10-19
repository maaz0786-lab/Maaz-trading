"""
Agent 8 - Patterns & Fibonacci.

Combines harmonic/ratio geometry with the classic candlestick formations:

* Fibonacci confluence : price sitting on a 38.2/50/61.8 retracement of the last
                         significant swing, ideally with a second ratio agreeing
* golden pocket        : 50-61.8% retracement zone in a trending market
* deep retracement     : 61.8-78.6% - last-defence zone for a trend
* extensions           : price beyond 127.2%/161.8% of the prior range
* pattern filter       : only takes Fibonacci setups that a candlestick pattern
                         also supports (a hammer at the 61.8%, etc.)

The Fibonacci levels themselves come from `features.add_fibonacci`, which uses a
rolling swing high/low, so the geometric reference is always a *real*, already
completed swing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from agents.base import Agent, make_vote, safe_bool
from agents.helpers import binary, combine, graded, plate


class PatternFibAgent(Agent):
    name = "pattern_fib"
    blurb = "Fibonacci retracement/extension confluence + harmonic geometry"
    regime_tags = ("trend", "pullback", "reversal")
    styles = ("intraday", "swing", "position")

    def __init__(self, zone_atr: float = 0.35, min_zone: int = 2):
        self.zone_atr = zone_atr
        self.min_zone = min_zone

    def score(self, fs, **ctx) -> "AgentVote":
        d = fs.df

        # ---------------- how many ratios cluster here ----------------------
        near_382 = safe_bool(d["at_fib_382"])
        near_500 = safe_bool(d["at_fib_500"])
        near_618 = safe_bool(d["at_fib_618"])
        n_near = near_382.astype(int) + near_500.astype(int) + near_618.astype(int)

        # ---------------- trend context for direction -----------------------
        up_swing = safe_bool(d["trendline_bull"]) | (d["ema20"] > d["ema50"])
        dn_swing = safe_bool(d["trendline_bear"]) | (d["ema20"] < d["ema50"])

        # price is in the upper half of the swing -> buy pullbacks
        pos = d["fib_pos"]
        buy_zone = pos.between(0.5, 0.886) & up_swing
        sell_zone = pos.between(0.114, 0.5) & dn_swing
        # deep retracements are higher risk but higher reward
        deep_buy = pos.between(0.618, 0.886) & up_swing & (d["adx"] > 22)
        deep_sell = pos.between(0.114, 0.382) & dn_swing & (d["adx"] > 22)

        # ---------------- candlestick confirmation --------------------------
        bull_pat = safe_bool(d["pat_any_bull"]) | (d["lower_wick_pct"] > 0.4) | \
                   (d["close_loc"] > 0.6)
        bear_pat = safe_bool(d["pat_any_bear"]) | (d["upper_wick_pct"] > 0.4) | \
                   (d["close_loc"] < 0.4)

        # ---------------- signals -------------------------------------------
        fib_long = (buy_zone | deep_buy) & (n_near >= self.min_zone) & bull_pat
        fib_short = (sell_zone | deep_sell) & (n_near >= self.min_zone) & bear_pat

        # extension plays (continuation beyond the prior range)
        ext_long = safe_bool(d["near_extension"]) & up_swing & (d["adx"] > 25) & bull_pat
        ext_short = safe_bool(d["near_extension"]) & dn_swing & (d["adx"] > 25) & bear_pat

        long_sig = fib_long | ext_long
        short_sig = fib_short | ext_short

        # ---------------- quality ------------------------------------------
        zone_q = graded(n_near, self.min_zone, 3)
        pos_q = pd.concat([
            (1 - (pos - 0.618).abs()),      # closest to the golden ratio
            (1 - (pos - 0.382).abs()),
        ], axis=1).max(axis=1).clip(0, 1)
        pat_q = pd.concat([
            graded(d["pat_bull_score"], 1, 3),
            graded(d["pat_bear_score"], 1, 3),
        ], axis=1).max(axis=1)
        wick_q = pd.concat([d["lower_wick_pct"], d["upper_wick_pct"]], axis=1).max(axis=1)
        wick_q = graded(wick_q, 0.3, 0.7)
        trend_q = graded(d["adx"], 18, 38)
        # the swing used for the levels should be meaningful, not noise
        swing_q = graded(d["fib_range_atr"], 2.0, 8.0)

        quality = combine(zone_q, pos_q, pat_q, wick_q, trend_q, swing_q,
                          weights=[0.2, 0.22, 0.18, 0.12, 0.14, 0.14])

        long_score = binary(long_sig, quality)
        short_score = binary(short_sig, quality)

        conf = quality * (safe_bool(long_sig) | safe_bool(short_sig)).astype(float)
        return make_vote(self, long_score, short_score, conf)
