"""
Agent 7 - Order Flow & Market Microstructure.

What this can and cannot do
---------------------------
Free OHLCV data has no real order book - no depth, no bid/ask ladder, no true
aggressor tags.  Rather than pretend otherwise, this agent uses the *observable
proxies* for order flow that the data genuinely supports, and says so plainly:

  tick-volume imbalance : which side of each bar volume is clustering on
  cumulative delta (CVD): signed tick volume accumulated
  buy/sell pressure     : volume-at-price within the bar's own range
  liquidity sweeps      : wicks through a swing extreme that snap back
  absorption / effort-vs-result divergence : volume without progress
  delta divergence      : delta disagreeing with price

Signals
-------
* sweep      : price wicks through a swing extreme, closes back inside, and
               volume pressure agrees (stops were run then defended)
* imbalance  : persistent pressure imbalance aligned with price, CVD confirming
* absorption : heavy volume, tiny range, at a level -> fade
* divergence : new price extreme on weaker delta -> exhaustion
"""

from __future__ import annotations

import pandas as pd

from agents.base import Agent, make_vote, safe_bool
from agents.helpers import binary, combine, graded


class OrderFlowAgent(Agent):
    name = "order_flow"
    blurb = "Tick-volume imbalance, CVD, liquidity sweeps, absorption, delta"
    regime_tags = ("breakout", "trend", "range")
    styles = ("scalp", "intraday", "swing")

    def __init__(self, imb_min: float = 0.18, sweep_wick: float = 0.4):
        self.imb_min = imb_min
        self.sweep_wick = sweep_wick

    def score(self, fs, **ctx) -> "AgentVote":
        d = fs.df

        # ---------------- liquidity sweeps ---------------------------------
        sweep_long = safe_bool(d["sweep_low"]) & (d["lower_wick_pct"] > self.sweep_wick)
        sweep_short = safe_bool(d["sweep_high"]) & (d["upper_wick_pct"] > self.sweep_wick)
        # heavy one-sided pressure into the sweep makes it more meaningful
        pressure_down = d["pressure_imb"] < -self.imb_min
        pressure_up = d["pressure_imb"] > self.imb_min
        sweep_long = sweep_long | (safe_bool(d["sweep_low"]) & pressure_down)
        sweep_short = sweep_short | (safe_bool(d["sweep_high"]) & pressure_up)

        # ---------------- pressure imbalance continuation -------------------
        imb_long = (d["pressure_imb_ma"] > self.imb_min) & (d["close"] > d["ema20"]) & \
                   (d["vol_ratio"] > 1.0)
        imb_short = (d["pressure_imb_ma"] < -self.imb_min) & (d["close"] < d["ema20"]) & \
                    (d["vol_ratio"] > 1.0)
        # CVD slope must agree, otherwise the imbalance is not persistent
        imb_long = imb_long & (safe_bool(d["cvd_slope"] >= 0) | safe_bool(d["delta_bull"]))
        imb_short = imb_short & (safe_bool(d["cvd_slope"] <= 0) | safe_bool(d["delta_bear"]))

        # ---------------- absorption -> fade -------------------------------
        absorb_long = safe_bool(d["absorption"]) & (d["close_loc"] > 0.55) & safe_bool(d["at_major_sup"])
        absorb_short = safe_bool(d["absorption"]) & (d["close_loc"] < 0.45) & safe_bool(d["at_major_res"])
        absorb_long = absorb_long | (safe_bool(d["effort_result_div"]) &
                                     safe_bool(d["at_major_sup"]) & (d["close_loc"] > 0.5))
        absorb_short = absorb_short | (safe_bool(d["effort_result_div"]) &
                                       safe_bool(d["at_major_res"]) & (d["close_loc"] < 0.5))

        # ---------------- delta divergence (exhaustion) --------------------
        new_hi = d["high"] >= d["high"].rolling(10, min_periods=4).max().shift(1)
        new_lo = d["low"] <= d["low"].rolling(10, min_periods=4).min().shift(1)
        delta_avg = d["delta"].rolling(10, min_periods=4).mean().shift(1)
        div_short = new_hi & (d["delta"] < delta_avg) & safe_bool(d["at_major_res"])
        div_long = new_lo & (d["delta"] > delta_avg) & safe_bool(d["at_major_sup"])

        long_sig = sweep_long | imb_long | absorb_long | div_long
        short_sig = sweep_short | imb_short | absorb_short | div_short

        # ---------------- quality ------------------------------------------
        imb_q = graded(d["pressure_imb"].abs(), self.imb_min, 0.6)
        wick_raw = pd.concat([d["lower_wick_pct"], d["upper_wick_pct"]], axis=1).max(axis=1)
        wick_q = graded(wick_raw, self.sweep_wick, 0.75)
        vol_q = graded(d["vol_ratio"], 1.0, 2.2)
        cvd_align = pd.concat([
            (safe_bool(d["cvd_slope"] > 0) | safe_bool(d["delta"] > 0)).astype(float),
            (safe_bool(d["cvd_slope"] < 0) | safe_bool(d["delta"] < 0)).astype(float),
        ], axis=1).max(axis=1)
        loc_raw = pd.concat([d["close_loc"], 1 - d["close_loc"]], axis=1).max(axis=1)
        loc_q = graded(loc_raw, 0.55, 0.9)
        level_q = (safe_bool(d["at_major_sup"]) | safe_bool(d["at_major_res"])).astype(float)

        q_sweep = combine(wick_q, imb_q, vol_q, loc_q, weights=[0.3, 0.3, 0.2, 0.2])
        q_imb = combine(imb_q, cvd_align, vol_q, weights=[0.4, 0.35, 0.25])
        q_absorb = combine(vol_q, loc_q, level_q, weights=[0.3, 0.3, 0.4])

        q_long = q_sweep.where(safe_bool(sweep_long),
                               q_imb.where(safe_bool(imb_long), q_absorb))
        q_short = q_sweep.where(safe_bool(sweep_short),
                                q_imb.where(safe_bool(imb_short), q_absorb))
        q_long = (q_long + 0.12 * level_q).clip(0, 1)
        q_short = (q_short + 0.12 * level_q).clip(0, 1)

        long_score = binary(long_sig, q_long)
        short_score = binary(short_sig, q_short)

        conf = pd.concat([q_long, q_short], axis=1).max(axis=1)
        return make_vote(self, long_score, short_score, conf)
