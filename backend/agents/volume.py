"""
Agent 6 - Volume-Based Trading.

Uses the feed's tick volume as the confirmation layer.  Volume does not tell you
direction on its own - it tells you *whether a move is real*, which is exactly
how this agent uses it:

* volume breakout    : price breaks a level with a volume spike (>= 1.6x)
* volume climax      : blow-off volume at an extreme -> fade candidate
* absorption         : huge volume, tiny range -> a big player is defending the
                       level, so the next move is likely against it
* dry-up after push  : volume collapses into a level after a strong move
* CVD / OBV confirm  : cumulative delta and on-balance volume agree with price
* VWAP               : volume-weighted mean as dynamic support/resistance

Volume spikes without price follow-through (effort vs result divergence) are
treated as a warning, not a signal.
"""

from __future__ import annotations

import pandas as pd

from agents.base import Agent, make_vote, safe_bool
from agents.helpers import binary, combine, graded


class VolumeAgent(Agent):
    name = "volume"
    blurb = "Volume spikes, absorption, CVD/OBV confirmation, VWAP"
    regime_tags = ("breakout", "trend", "range")
    styles = ("scalp", "intraday", "swing", "position")

    def __init__(self, spike: float = 1.6, climax: float = 2.2, dry: float = 0.55):
        self.spike = spike
        self.climax = climax
        self.dry = dry

    def score(self, fs, **ctx) -> "AgentVote":
        d = fs.df
        vr = d["vol_ratio"]

        # ---------------- volume breakout ----------------------------------
        spiked = vr > self.spike
        vol_brk_up = spiked & (safe_bool(d["break_res_20"]) | safe_bool(d["break_res_50"])) & \
                     (d["body_pct"] > 0.45)
        vol_brk_dn = spiked & (safe_bool(d["break_sup_20"]) | safe_bool(d["break_sup_50"])) & \
                     (d["body_pct"] > 0.45)
        # volume + body without a formal level break still counts on fast TFs
        vol_brk_up |= spiked & safe_bool(d["vol_breakout"]) & (d["close_loc"] > 0.75) & \
                      (d["ema20"] > d["ema50"])
        vol_brk_dn |= spiked & safe_bool(d["vol_breakout"]) & (d["close_loc"] < 0.25) & \
                      (d["ema20"] < d["ema50"])

        # ---------------- climax -> fade -----------------------------------
        climax_up = safe_bool(d["climax_up"]) | \
                    (spiked & (d["close_loc"] > 0.9) & (d["upper_wick_pct"] > 0.3))
        climax_dn = safe_bool(d["climax_down"]) | \
                    (spiked & (d["close_loc"] < 0.1) & (d["lower_wick_pct"] > 0.3))

        # ---------------- absorption at a level -----------------------------
        absorbed_up = safe_bool(d["absorption"]) & (d["close_loc"] > 0.5) & safe_bool(d["at_major_sup"])
        absorbed_dn = safe_bool(d["absorption"]) & (d["close_loc"] < 0.5) & safe_bool(d["at_major_res"])
        # effort vs result divergence: big range, no progress
        absorbed_up |= safe_bool(d["effort_result_div"]) & (d["close_loc"] > 0.55) & safe_bool(d["at_major_sup"])
        absorbed_dn |= safe_bool(d["effort_result_div"]) & (d["close_loc"] < 0.45) & safe_bool(d["at_major_res"])

        # ---------------- dry-up after a push ------------------------------
        pushed_up = d["close"].rolling(8).max() >= d["close"].rolling(20).max().shift(1)
        pushed_dn = d["close"].rolling(8).min() <= d["close"].rolling(20).min().shift(1)
        dryup_short = (vr < self.dry) & pushed_up & safe_bool(d["at_major_res"])
        dryup_long = (vr < self.dry) & pushed_dn & safe_bool(d["at_major_sup"])

        # ---------------- VWAP reclaim -------------------------------------
        above = safe_bool(d["above_vwap"])
        vwap_long = above & ~above.shift(1).fillna(False) & (d["vwap_dist"].abs() < 0.5)
        vwap_short = ~above & above.shift(1).fillna(False) & (d["vwap_dist"].abs() < 0.5)

        break_long = vol_brk_up | absorbed_up | vwap_long
        break_short = vol_brk_dn | absorbed_dn | vwap_short
        fade_long = climax_dn | dryup_long
        fade_short = climax_up | dryup_short

        long_sig = break_long | fade_long
        short_sig = break_short | fade_short

        # ---------------- CVD / OBV confirmation ---------------------------
        cvd_long = safe_bool(d["cvd_slope"] > 0) | safe_bool(d["delta_bull"])
        cvd_short = safe_bool(d["cvd_slope"] < 0) | safe_bool(d["delta_bear"])
        obv_long = safe_bool(d["obv_slope"] > 0)
        obv_short = safe_bool(d["obv_slope"] < 0)

        # ---------------- quality ------------------------------------------
        spike_q = graded(vr, self.spike, self.climax + 1.0)
        body_q = graded(d["body_pct"], 0.4, 0.75)
        loc_raw = pd.concat([d["close_loc"], 1 - d["close_loc"]], axis=1).max(axis=1)
        loc_q = graded(loc_raw, 0.55, 0.9)
        cvd_q = pd.concat([
            (cvd_long | obv_long).astype(float),
            (cvd_short | obv_short).astype(float),
        ], axis=1).max(axis=1)
        level_q = (safe_bool(d["at_major_sup"]) | safe_bool(d["at_major_res"]) |
                   safe_bool(d["break_res_20"]) | safe_bool(d["break_sup_20"])).astype(float)

        q_break = combine(spike_q, body_q, loc_q, cvd_q,
                          weights=[0.3, 0.25, 0.2, 0.25])
        q_fade = combine(spike_q, loc_q,
                         graded(d["vol_regime"], 1.4, 0.85, invert=True),
                         weights=[0.35, 0.35, 0.3])

        is_break_long = safe_bool(break_long)
        is_break_short = safe_bool(break_short)
        q_long = q_break.where(is_break_long, q_fade)
        q_short = q_break.where(is_break_short, q_fade)
        q_long = (q_long + 0.15 * level_q).clip(0, 1)
        q_short = (q_short + 0.15 * level_q).clip(0, 1)

        long_score = binary(long_sig, q_long)
        short_score = binary(short_sig, q_short)

        conf = pd.concat([q_long, q_short], axis=1).max(axis=1)
        return make_vote(self, long_score, short_score, conf)
