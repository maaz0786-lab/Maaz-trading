"""
Agent 3 - Trendlines, Support/Resistance and Breakouts.

This agent speaks the language of horizontal levels and dynamic channels:

* touch & reject  : price reaches a rolling S/R level (or a regression-channel
                    edge acting as a trendline) and closes back inside
* breakout        : a decisive close through a level **with volume expansion**
* retest          : a broken level is retested from the other side and holds
                    (broken support becomes resistance, and vice versa)

Confidence rewards (a) confluence between a horizontal level and a Fibonacci
retracement, (b) volume expansion on the break, and (c) room to run to the next
level - a breakout into immediate resistance is penalised.
"""

from __future__ import annotations

import pandas as pd

from agents.base import Agent, make_vote, safe_bool
from agents.helpers import binary, combine, graded


class SupportResistanceAgent(Agent):
    name = "sr_breakout"
    blurb = "Trendlines, S/R touches, breakouts, retests of broken levels"
    regime_tags = ("range", "breakout", "trend")
    styles = ("scalp", "intraday", "swing", "position")

    def __init__(self, touch_atr: float = 0.35, min_room_atr: float = 1.0,
                 vol_confirm: float = 1.3):
        self.touch_atr = touch_atr
        self.min_room_atr = min_room_atr
        self.vol_confirm = vol_confirm

    def score(self, fs, **ctx) -> "AgentVote":
        d = fs.df

        # ---------------- level touches ------------------------------------
        at_res = ((d["dist_res_50"].abs() < self.touch_atr) |
                  (d["dist_res_100"].abs() < self.touch_atr * 1.4) |
                  safe_bool(d["at_major_res"]))
        at_sup = ((d["dist_sup_50"].abs() < self.touch_atr) |
                  (d["dist_sup_100"].abs() < self.touch_atr * 1.4) |
                  safe_bool(d["at_major_sup"]))

        # rejection requires the bar to close back away from the level
        reject_res = at_res & (d["close_loc"] < 0.4)
        reject_sup = at_sup & (d["close_loc"] > 0.6)

        # dynamic trendline (regression-channel edge) touches
        tl_res = safe_bool(d["tl_res_touch"]) & safe_bool(d["trendline_bear"]) & (d["close_loc"] < 0.45)
        tl_sup = safe_bool(d["tl_sup_touch"]) & safe_bool(d["trendline_bull"]) & (d["close_loc"] > 0.55)

        # ---------------- breakouts ----------------------------------------
        vol_up = d["vol_ratio"] > self.vol_confirm
        brk_up = (safe_bool(d["break_res_20"]) | safe_bool(d["break_res_50"])) & vol_up
        brk_dn = (safe_bool(d["break_sup_20"]) | safe_bool(d["break_sup_50"])) & vol_up

        # a breakout out of a compression range is the strongest version
        squeeze = d["range_20_atr"] < d["range_20_ma"]
        brk_up_sq = brk_up & squeeze
        brk_dn_sq = brk_dn & squeeze

        # ---------------- retests of broken levels -------------------------
        broken_res_recent = safe_bool(d["break_res_20"]).rolling(15).sum() > 0
        broken_sup_recent = safe_bool(d["break_sup_20"]).rolling(15).sum() > 0
        retest_long = broken_res_recent & d["dist_res_20"].between(-0.4, 0.3) & (d["close_loc"] > 0.5)
        retest_short = broken_sup_recent & d["dist_sup_20"].between(-0.3, 0.4) & (d["close_loc"] < 0.5)

        long_sig = reject_sup | tl_sup | brk_up | brk_up_sq | retest_long
        short_sig = reject_res | tl_res | brk_dn | brk_dn_sq | retest_short

        # ---------------- quality ------------------------------------------
        vol_q = graded(d["vol_ratio"], self.vol_confirm, 2.5)
        loc_raw = pd.concat([d["close_loc"], 1 - d["close_loc"]], axis=1).max(axis=1)
        loc_q = graded(loc_raw, 0.55, 0.9)
        room_long = graded(d["room_up_atr"], self.min_room_atr, 4.0)
        room_short = graded(d["room_dn_atr"], self.min_room_atr, 4.0)
        # compression before the break makes a better breakout
        squeeze_q = graded(d["range_20_atr"] / d["range_20_ma"], 1.4, 0.6, invert=True)
        # horizontal level + fibonacci confluence
        fib_conf = (safe_bool(d["at_fib_618"]) | safe_bool(d["at_fib_500"]) |
                    safe_bool(d["at_fib_382"])).astype(float)
        trend_q = graded(d["adx"], 15, 35)

        q_long = combine(vol_q, loc_q, room_long, trend_q,
                         weights=[0.3, 0.2, 0.3, 0.2]) * (1.0 + 0.25 * fib_conf)
        q_short = combine(vol_q, loc_q, room_short, trend_q,
                          weights=[0.3, 0.2, 0.3, 0.2]) * (1.0 + 0.25 * fib_conf)

        q_long = q_long.where(~safe_bool(brk_up_sq), (q_long + 0.2 * squeeze_q).clip(0, 1))
        q_short = q_short.where(~safe_bool(brk_dn_sq), (q_short + 0.2 * squeeze_q).clip(0, 1))

        # penalise setups with no room to run
        q_long = q_long.where(d["room_up_atr"] >= self.min_room_atr * 0.6, q_long * 0.5)
        q_short = q_short.where(d["room_dn_atr"] >= self.min_room_atr * 0.6, q_short * 0.5)

        long_score = binary(long_sig, q_long)
        short_score = binary(short_sig, q_short)

        conf = pd.concat([q_long, q_short], axis=1).max(axis=1)
        return make_vote(self, long_score, short_score, conf)
