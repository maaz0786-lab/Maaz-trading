"""
Agent 9 - Chart Patterns & Candlestick Patterns.

Two layers, both genuinely detected rather than eyeballed:

Candlesticks (TA-Lib, 22 formations)
  hammer, engulfing, morning/evening star, three white soldiers, harami,
  piercing line, dark cloud cover, marubozu, doji variants, tweezers, ...

Chart patterns (geometry from confirmed swing pivots)
  double top / bottom, head & shoulders (and inverse), ascending / descending /
  symmetrical triangles, flags & pennants.

A pattern only votes when it appears **where it matters** - at support, at a
Fibonacci level, or in agreement with the prevailing structure.  A hammer in the
middle of nowhere is noise; a hammer at a double bottom is a trade.
"""

from __future__ import annotations

import pandas as pd

from agents.base import Agent, make_vote, safe_bool
from agents.helpers import binary, combine, graded


class ChartPatternAgent(Agent):
    name = "chart_pattern"
    blurb = "Candlestick formations + double tops, H&S, triangles, flags at levels"
    regime_tags = ("reversal", "range", "breakout")
    styles = ("scalp", "intraday", "swing")

    def __init__(self, min_patterns: int = 1):
        self.min_patterns = min_patterns

    def score(self, fs, **ctx) -> "AgentVote":
        d = fs.df

        # ---------------- candlestick layer --------------------------------
        bull_cd = safe_bool(d["pat_any_bull"])
        bear_cd = safe_bool(d["pat_any_bear"])
        # specific high-quality formations
        strong_bull = (safe_bool(d["pat_morning_star"]) | safe_bool(d["pat_three_white"]) |
                       safe_bool(d["pat_engulfing"]) & (d["pat_engulfing"] > 0) |
                       safe_bool(d["pat_piercing"]))
        strong_bear = (safe_bool(d["pat_evening_star"]) | safe_bool(d["pat_three_black"]) |
                       safe_bool(d["pat_darkcloud"]) | safe_bool(d["pat_shooting"]) |
                       safe_bool(d["pat_hanging"]))

        # ---------------- chart-pattern layer ------------------------------
        cp_bull = safe_bool(d["cp_bull"])
        cp_bear = safe_bool(d["cp_bear"])

        # ---------------- location filter ----------------------------------
        # patterns at a level or a fib zone carry far more information
        at_level_bull = safe_bool(d["at_major_sup"]) | safe_bool(d["at_fib_618"]) | \
                        safe_bool(d["at_fib_500"]) | safe_bool(d["tl_sup_touch"])
        at_level_bear = safe_bool(d["at_major_res"]) | safe_bool(d["at_fib_618"]) | \
                        safe_bool(d["at_fib_500"]) | safe_bool(d["tl_res_touch"])
        # or after a liquidity sweep of that level
        swept_bull = safe_bool(d["sweep_low"])
        swept_bear = safe_bool(d["sweep_high"])

        long_sig = ((bull_cd & (at_level_bull | swept_bull)) |
                    (strong_bull & (at_level_bull | swept_bull)) |
                    cp_bull)
        short_sig = ((bear_cd & (at_level_bear | swept_bear)) |
                     (strong_bear & (at_level_bear | swept_bear)) |
                     cp_bear)

        # ---------------- quality ------------------------------------------
        pat_cnt = pd.concat([d["pat_bull_score"], d["pat_bear_score"]], axis=1).max(axis=1)
        pat_q = graded(pat_cnt, 1, 3)
        strong_q = (safe_bool(strong_bull) | safe_bool(strong_bear)).astype(float)
        cp_q = (cp_bull | cp_bear).astype(float)
        loc_q = pd.concat([at_level_bull, at_level_bear], axis=1).max(axis=1).astype(float)
        sweep_q = (swept_bull | swept_bear).astype(float)
        body_q = graded(d["body_pct"], 0.4, 0.75)
        vol_q = graded(d["vol_ratio"], 1.0, 2.0)

        quality = combine(pat_q, strong_q, cp_q, loc_q, sweep_q, body_q, vol_q,
                          weights=[0.18, 0.16, 0.18, 0.18, 0.12, 0.09, 0.09])

        long_score = binary(long_sig, quality)
        short_score = binary(short_sig, quality)

        conf = quality * (safe_bool(long_sig) | safe_bool(short_sig)).astype(float)
        return make_vote(self, long_score, short_score, conf)
