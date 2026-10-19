"""
Agent 10 - the MAIN orchestrator agent.

Runs all nine specialists in parallel, then combines them the way a disciplined
desk actually would:

1. **Parallel evaluation** - every specialist scores the same feature frame
   independently (no specialist sees another's output, so there is no groupthink
   at the signal level).

2. **Learned per-agent weighting** - each agent's vote is weighted by how well
   that agent has actually performed *in the current regime, on the current
   timeframe, in the current session*, measured on training data only.  An agent
   that has been losing money in ranging markets gets its vote scaled down there.

3. **Regime gating** - agents whose competence does not match the live regime are
   muted (e.g. mean reversion is largely ignored in a strong trend).

4. **Consensus requirement** - a trade is only proposed when enough
   confidence-weighted agreement exists, measured as both a score and a vote
   count, so one loud agent cannot drag the desk into a trade.

5. **Conflict veto** - if long and short votes are both strong, the panel stands
   aside rather than guessing.

6. **Higher-timeframe alignment** - a bonus/penalty from the higher-TF trend, so
   intraday longs are not taken against a strongly bearish H4.

The output is a `PanelDecision` per bar: side, score, agreement, and each agent's
contribution - so every decision is auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from agents.base import AgentVote
from regime import Regime


@dataclass
class PanelDecision:
    """The orchestrator's verdict on every bar."""
    side: pd.Series            # +1 long, -1 short, 0 flat
    score: pd.Series           # confidence-weighted agreement, 0..1
    agreement: pd.Series       # how many agents voted for the chosen side
    conflict: pd.Series        # True when long and short votes are both strong
    voters_long: pd.Series
    voters_short: pd.Series
    weights: dict = field(default_factory=dict)      # agent -> mean weight used
    contributions: pd.DataFrame = field(default_factory=pd.DataFrame)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "side": self.side,
            "score": self.score,
            "agreement": self.agreement,
            "conflict": self.conflict,
        })


class OrchestratorAgent:
    """The main agent: runs the panel and produces a single decision per bar."""

    name = "orchestrator"
    blurb = "Runs all 9 specialists in parallel, weights them by proven edge"

    # agents whose competence is regime-specific; muted elsewhere
    REGIME_AFFINITY = {
        "mean_reversion": {Regime.RANGE},
        "trend_following": {Regime.TREND_UP, Regime.TREND_DOWN, Regime.BREAKOUT},
        "sr_breakout": {Regime.RANGE, Regime.BREAKOUT, Regime.TREND_UP, Regime.TREND_DOWN},
        "indicators": {Regime.RANGE, Regime.TREND_UP, Regime.TREND_DOWN, Regime.BREAKOUT},
        "price_action": {Regime.RANGE, Regime.TREND_UP, Regime.TREND_DOWN, Regime.BREAKOUT},
        "volume": {Regime.BREAKOUT, Regime.TREND_UP, Regime.TREND_DOWN, Regime.RANGE},
        "order_flow": {Regime.BREAKOUT, Regime.TREND_UP, Regime.TREND_DOWN, Regime.RANGE},
        "pattern_fib": {Regime.TREND_UP, Regime.TREND_DOWN, Regime.BREAKOUT, Regime.RANGE},
        "chart_pattern": {Regime.RANGE, Regime.BREAKOUT, Regime.TREND_UP, Regime.TREND_DOWN},
    }

    def __init__(self, agents, min_agreement: int = 2, min_score: float = 0.30,
                 conflict_gap: float = 0.12, use_learned_weights: bool = True,
                 htf_weight: float = 0.20):
        self.agents = list(agents)
        self.min_agreement = min_agreement
        self.min_score = min_score
        self.conflict_gap = conflict_gap
        self.use_learned_weights = use_learned_weights
        self.htf_weight = htf_weight
        self.agent_weights: dict[str, float] = {a.name: 1.0 for a in self.agents}
        self.agent_regime_weights: dict[str, dict[int, float]] = {}
        self.session_weights: dict[str, dict[str, float]] = {}
        self.weight_source = "uniform (untrained)"

    # ------------------------------------------------------------------ learning
    def learn_weights(self, stats: pd.DataFrame) -> None:
        """Set per-agent weights from measured *training* performance.

        `stats` must have columns [agent, regime, session, n, win_rate,
        profit_factor, expectancy] - exactly what `training.evaluate_agents`
        produces.  Rows with regime == -1 are the agent's overall figures and
        rows with session == 'all' are the regime totals; those feed the flat
        weight only, never the per-regime / per-session maps, otherwise the same
        evidence is counted three times.
        """
        if stats is None or stats.empty or not self.use_learned_weights:
            return

        overall = stats[(stats["regime"] == -1) & (stats["session"] == "all")]
        if not overall.empty:
            flat = {r["agent"]: _edge_weight(r["profit_factor"], r["expectancy"],
                                             int(r["n"]))
                    for _, r in overall.iterrows()}
            self.agent_weights.update({k: float(np.clip(v, 0.05, 2.5))
                                       for k, v in flat.items()})

        # per-regime and per-session maps from the *specific* cells only
        by_regime = stats[(stats["regime"] != -1) & (stats["session"] == "all")]
        by_session = stats[(stats["session"] != "all") & (stats["regime"] == -1)]

        for agent in stats["agent"].unique():
            g = by_regime[by_regime["agent"] == agent]
            if not g.empty:
                self.agent_regime_weights[agent] = {
                    int(r["regime"]): _edge_weight(r["profit_factor"],
                                                   r["expectancy"], int(r["n"]))
                    for _, r in g.iterrows()
                }
            s = by_session[by_session["agent"] == agent]
            if not s.empty:
                self.session_weights[agent] = {
                    str(r["session"]): _edge_weight(r["profit_factor"],
                                                    r["expectancy"], int(r["n"]))
                    for _, r in s.iterrows()
                }

        self.weight_source = ("learned from training data "
                              "(PF/expectancy by regime + session)")

    # ------------------------------------------------------------------- scoring
    def score(self, fs, labels: pd.Series | None = None, **ctx) -> PanelDecision:
        """Run the panel in parallel and combine into one decision per bar."""
        # 1. every specialist votes independently
        votes: list[AgentVote] = [agent.score(fs, **ctx) for agent in self.agents]

        idx = fs.df.index
        if labels is None:
            labels = pd.Series(Regime.UNKNOWN, index=idx)

        # 2. effective weight per agent per bar
        eff: dict[str, pd.Series] = {}
        sess = _session_of(fs)
        for vote in votes:
            base = self.agent_weights.get(vote.name, 1.0)
            w = pd.Series(float(base), index=idx, dtype="float64")

            regime_map = self.agent_regime_weights.get(vote.name)
            if regime_map:
                w = w * labels.map(regime_map).fillna(0.6)

            affinity = self.REGIME_AFFINITY.get(vote.name)
            if affinity is not None:
                w = w.where(labels.isin(list(affinity)), w * 0.25)

            session_map = self.session_weights.get(vote.name)
            if session_map:
                w = w * sess.map(session_map).fillna(0.85)

            eff[vote.name] = w.fillna(0.5).clip(0.0, 2.5)

        # 3. weighted long / short scores and vote counts
        long_w = pd.Series(0.0, index=idx)
        short_w = pd.Series(0.0, index=idx)
        long_cnt = pd.Series(0.0, index=idx)
        short_cnt = pd.Series(0.0, index=idx)
        contributions = {}

        # Re-centre the effective weights around 1.0 before counting votes.
        # Learning multiplies several sub-1 factors together, which would push
        # every weight below any fixed "is this agent active?" threshold and
        # silently zero the agreement count.  Scaling by the mean keeps the
        # *relative* differences (which is what matters) while staying usable.
        mean_w = float(np.mean([eff[a.name].mean() for a in votes])) or 1.0
        for vote in votes:
            w = eff[vote.name]
            w = w / mean_w
            eff[vote.name] = w
            lw = vote.long_score * vote.confidence * w
            sw = vote.short_score * vote.confidence * w
            long_w = long_w + lw
            short_w = short_w + sw
            active = (w > 0.5).astype(float)
            long_cnt = long_cnt + (vote.direction == 1).astype(float) * active
            short_cnt = short_cnt + (vote.direction == -1).astype(float) * active
            contributions[f"{vote.name}_L"] = lw
            contributions[f"{vote.name}_S"] = sw

        contrib = pd.DataFrame(contributions, index=idx)

        # 4. consensus strength.
        # Normalise by the weight of the agents that actually *voted* on this bar,
        # not by the whole panel.  Otherwise a panel of nine agents that only fires
        # on 3 bars out of 10 would produce a score of ~0.17 on a perfect 3-for-3
        # agreement, and the consensus gate would never open.
        voted = pd.Series(0.0, index=idx)
        for vote in votes:
            voted = voted + ((vote.long_score > 0) | (vote.short_score > 0)).astype(float) * \
                    eff[vote.name]
        denom = voted.where(voted > 1e-9, 1.0)
        long_score = (long_w / denom).clip(0, 1)
        short_score = (short_w / denom).clip(0, 1)

        # 5. higher-timeframe alignment modifier
        htf = _htf_bias(fs)
        long_score = (long_score * (1 + self.htf_weight * htf.clip(0, 1))).clip(0, 1)
        short_score = (short_score * (1 + self.htf_weight * (-htf).clip(0, 1))).clip(0, 1)

        # 6. side, agreement, score, conflict
        side = pd.Series(0, index=idx, dtype="int8")
        side[long_score > short_score] = 1
        side[short_score > long_score] = -1

        agreement = pd.concat([long_cnt.where(side == 1, 0.0),
                               short_cnt.where(side == -1, 0.0)], axis=1).max(axis=1)
        score = pd.concat([long_score.where(side == 1, 0.0),
                           short_score.where(side == -1, 0.0)], axis=1).max(axis=1)

        both_strong = (long_score > 0.15) & (short_score > 0.15)
        conflict = both_strong & ((long_score - short_score).abs() < self.conflict_gap)

        # 7. consensus gate
        enough = (agreement >= self.min_agreement) & (score >= self.min_score)
        side = side.where(enough & ~conflict, 0)

        return PanelDecision(
            side=side, score=score, agreement=agreement, conflict=conflict,
            voters_long=long_cnt, voters_short=short_cnt,
            weights={k: float(v.mean()) for k, v in eff.items()},
            contributions=contrib,
        )


# ---------------------------------------------------------------------- helpers
def _edge_weight(profit_factor: float, expectancy: float, n: float) -> float:
    """Translate measured performance into a vote weight.

    PF 1.0 (breakeven) -> 0.5, PF 1.5 -> ~0.9, PF 2.0 -> ~1.3, PF 3.0 -> ~1.9.
    Thin evidence is shrunk toward 0.5 so a lucky 10-trade sample is not trusted.
    """
    if n < 20 or not np.isfinite(profit_factor):
        return 0.5
    pf = max(0.0, float(profit_factor))
    w = 0.5 + 0.9 * np.log(pf) if pf > 0 else 0.0
    w = float(np.clip(w, 0.0, 2.0))
    w += float(np.clip(expectancy, -0.2, 0.2)) * 0.5
    shrink = n / (n + 100.0)
    w = 0.5 + (w - 0.5) * shrink
    return float(np.clip(w, 0.0, 2.5))


def _session_of(fs) -> pd.Series:
    """Label each bar with its dominant trading session (UTC)."""
    hour = fs.df.index.hour
    out = pd.Series("asia", index=fs.df.index, dtype=object)
    out[(hour >= 7) & (hour < 12)] = "london_am"
    out[(hour >= 12) & (hour < 16)] = "overlap"
    out[(hour >= 16) & (hour < 21)] = "newyork_pm"
    out[(hour >= 21) | (hour < 0)] = "sydney"
    return out


def _htf_bias(fs) -> pd.Series:
    """Higher-timeframe trend bias in [-1, 1] from the merged htf_ columns."""
    cols = [c for c in fs.df.columns
            if c.startswith("htf_") and c.endswith("_ema50")]
    if not cols:
        return pd.Series(0.0, index=fs.df.index)
    above = pd.concat([(fs.df["close"] > fs.df[c]).astype(float) for c in cols],
                      axis=1).mean(axis=1)
    return (above - 0.5) * 2.0
