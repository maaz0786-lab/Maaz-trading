"""
Agent framework: the contract every one of the 9 specialist agents implements.

Design
------
Each agent is a *pure function of the feature frame* - it never sees the future
and never knows about other agents.  It returns a per-bar voting table:

    long_score  : float in [0, 1]   - how strongly it wants to be long
    short_score : float in [0, 1]   - how strongly it wants to be short
    direction   : -1 / 0 / +1       - the crisp side it votes for
    confidence  : float in [0, 1]   - how much weight the orchestrator should
                                      give this vote (setup quality)

`score()` is called once per (style, timeframe, fold).  Because it is fully
vectorised, evaluating 9 agents over ~2.8M M1 bars takes seconds, not hours.

Stops and targets are NOT decided here - the risk manager places them from ATR
so that every agent is judged on identical, timeframe-appropriate risk.  This
keeps the comparison between agents honest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class AgentVote:
    """One agent's opinion on every bar."""
    name: str
    long_score: pd.Series
    short_score: pd.Series
    direction: pd.Series          # +1 long, -1 short, 0 flat
    confidence: pd.Series
    regime: pd.Series = None      # which market regime this vote is valid in
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        idx = self.long_score.index
        if self.regime is None:
            self.regime = pd.Series(0, index=idx)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            f"{self.name}_long": self.long_score,
            f"{self.name}_short": self.short_score,
            f"{self.name}_dir": self.direction,
            f"{self.name}_conf": self.confidence,
        }, index=self.long_score.index)


class Agent:
    """Base class.  Subclasses implement `score` and set `name` / `blurb`."""

    #: short identifier used in column names and reports
    name: str = "agent"
    #: human description shown in the report
    blurb: str = ""
    #: market conditions this agent is designed for
    regime_tags: tuple[str, ...] = ()

    #: entry styles this agent is competent in (subset of STYLES keys)
    styles: tuple[str, ...] = ()

    #: minimum bars of warm-up before votes are meaningful
    warmup: int = 260

    def score(self, fs, **ctx) -> AgentVote:
        raise NotImplementedError


def make_vote(agent: "Agent", long_s, short_s, conf, regime=None, meta=None) -> AgentVote:
    """Build an AgentVote, clipping scores/confidence into [0, 1]."""
    idx = long_s.index
    long_s = long_s.clip(0, 1).astype(float)
    short_s = short_s.clip(0, 1).astype(float)
    conf = conf.clip(0, 1).astype(float)

    # crisp direction: only take a side once the score clears a small threshold
    # and the two sides are not both shouting at once.
    direction = pd.Series(0, index=idx, dtype="int8")
    both = (long_s > 0) & (short_s > 0)
    only_long = (long_s > 0) & (short_s == 0)
    only_short = (short_s > 0) & (long_s == 0)
    # when both fire, the stronger side wins and confidence is discounted
    both_long = both & (long_s > short_s)
    both_short = both & (short_s > long_s)
    direction[only_long | both_long] = 1
    direction[only_short | both_short] = -1
    conf = conf.mask(both, conf * 0.5)

    return AgentVote(
        name=agent.name,
        long_score=long_s,
        short_score=short_s,
        direction=direction,
        confidence=conf,
        regime=regime if regime is not None else pd.Series(0, index=idx),
        meta=meta or {},
    )


def graded(series: pd.Series, lo: float, hi: float, invert: bool = False) -> pd.Series:
    """Linearly map `series` into [0,1] between lo and hi (optionally inverted).

    Used to turn continuous indicator values into graded scores, e.g.
    graded(rsi, lo=30, hi=50) is 0 at RSI 30 and 1 at RSI 50.
    """
    s = (series - lo) / (hi - lo)
    s = s.clip(0, 1)
    return (1 - s) if invert else s


def safe_bool(s: pd.Series) -> pd.Series:
    """NaN-safe boolean series."""
    return s.fillna(False).astype(bool)
