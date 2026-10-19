"""
Shared helpers for the specialist agents.

`graded` maps a continuous indicator value into [0, 1] so heterogeneous signals
(RSI, ADX, wick sizes) can be combined into one quality score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def graded(series: pd.Series, lo: float, hi: float, invert: bool = False) -> pd.Series:
    """Linearly map `series` into [0,1] between lo and hi (optionally inverted)."""
    s = (series.astype(float) - lo) / (hi - lo)
    s = s.clip(0.0, 1.0)
    return (1.0 - s) if invert else s


def plate(series: pd.Series, centre: float, width: float) -> pd.Series:
    """1 when the value sits within +/- width of `centre`, tapering to 0."""
    dist = (series.astype(float) - centre).abs() / width
    return (1.0 - dist).clip(0.0, 1.0)


def combine(*parts: pd.Series, weights=None) -> pd.Series:
    """Weighted mean of quality components, NaN treated as 0."""
    if not parts:
        raise ValueError("combine() needs at least one component")
    weights = weights or [1.0] * len(parts)
    total_w = sum(weights)
    out = sum((p.fillna(0.0) * w) for p, w in zip(parts, weights))
    return out / total_w


def gate(score: pd.Series, *conditions: pd.Series) -> pd.Series:
    """Zero the score wherever any condition is False."""
    ok = pd.concat([c.fillna(False) for c in conditions], axis=1).all(axis=1)
    return score.where(ok, 0.0)


def binary(condition: pd.Series, quality: pd.Series, floor: float = 0.2) -> pd.Series:
    """Signal present -> at least `floor`, scaled by quality; else 0."""
    on = condition.fillna(False).astype(float)
    q = quality.fillna(0.0).clip(0.0, 1.0)
    return on * np.maximum(q, floor)


def shift_safe(series: pd.Series, periods: int = 1) -> pd.Series:
    return series.shift(periods)


def crossing_up(fast: pd.Series, slow: pd.Series) -> pd.Series:
    return (fast > slow) & (fast.shift(1) <= slow.shift(1))


def crossing_down(fast: pd.Series, slow: pd.Series) -> pd.Series:
    return (fast < slow) & (fast.shift(1) >= slow.shift(1))
