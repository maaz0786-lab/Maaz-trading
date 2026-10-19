"""
Market-regime classification.

Agents are not universally good - a mean-reversion agent should be trusted in a
range and ignored in a strong trend, and vice versa.  Every bar therefore gets a
regime label, and the orchestrator learns which agents work in which regime.

Regimes
-------
TREND_UP / TREND_DOWN  : directional, ADX-confirmed, EMA stack aligned
RANGE                  : low ADX, price oscillating inside the channel
BREAKOUT               : volatility expansion + range break
HIGH_VOL               : ATR far above normal (danger zone - most agents off)
QUIET                  : ATR far below normal (spread eats you alive)
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
import pandas as pd


class Regime(IntEnum):
    UNKNOWN = 0
    TREND_UP = 1
    TREND_DOWN = 2
    RANGE = 3
    BREAKOUT = 4
    HIGH_VOL = 5
    QUIET = 6


REGIME_NAMES = {
    0: "unknown", 1: "trend_up", 2: "trend_down",
    3: "range", 4: "breakout", 5: "high_vol", 6: "quiet",
}

# Which regimes are tradeable at all for each broad strategy family.
REGIME_SUITABLE = {
    "trend_following": {1, 2, 4},
    "mean_reversion": {3},
    "breakout": {4, 3},
    "range": {3},
    "any": {1, 2, 3, 4},
}


def classify(fs, adx_trend: float = 25.0, adx_range: float = 20.0,
             vol_hi: float = 1.7, vol_lo: float = 0.6) -> pd.Series:
    """Return a per-bar Regime label for a FeatureSet."""
    df = fs.df
    adx = df["adx"]
    vr = df["vol_regime"]

    trending_up = (adx > adx_trend) & (df["close"] > df["ema50"]) & (df["ema200_slope"] >= 0)
    trending_dn = (adx > adx_trend) & (df["close"] < df["ema50"]) & (df["ema200_slope"] <= 0)
    rng = adx < adx_range
    brk = (df["break_res_50"] | df["break_sup_50"]) & (df["vol_ratio"] > 1.4)
    hi_vol = vr > vol_hi
    quiet = vr < vol_lo

    out = pd.Series(Regime.RANGE, index=df.index, dtype="int8")
    out[trending_up] = Regime.TREND_UP
    out[trending_dn] = Regime.TREND_DOWN
    out[brk] = Regime.BREAKOUT
    out[rng] = Regime.RANGE
    out[hi_vol] = Regime.HIGH_VOL
    out[quiet] = Regime.QUIET

    # danger regimes win over everything else
    out[hi_vol & ~brk] = Regime.HIGH_VOL
    out[quiet & ~brk] = Regime.QUIET
    return out.where(~df["atr"].isna(), Regime.UNKNOWN)


def regime_stats(labels: pd.Series) -> pd.DataFrame:
    counts = labels.value_counts().sort_index()
    total = counts.sum()
    return pd.DataFrame({
        "regime": [REGIME_NAMES.get(i, i) for i in counts.index],
        "bars": counts.values,
        "pct": (100 * counts.values / total).round(2),
    })


def is_tradable(labels: pd.Series, allow_high_vol: bool = False,
                allow_quiet: bool = False) -> pd.Series:
    """Boolean mask of bars where a normal strategy may trade."""
    good = labels.isin([Regime.TREND_UP, Regime.TREND_DOWN, Regime.RANGE, Regime.BREAKOUT])
    if allow_high_vol:
        good |= labels == Regime.HIGH_VOL
    if allow_quiet:
        good |= labels == Regime.QUIET
    return good.fillna(False)
