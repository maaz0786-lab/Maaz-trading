"""
Training: measure each agent's real edge, then teach the orchestrator.

This is the "train" half of the system and it runs on the training period only.

What it does
------------
1. Every specialist agent votes on the training features.
2. Each agent's votes are backtested **individually** - its own signals, with the
   same risk model, spread and resolution bars the panel will use.
3. Those results are broken down by market regime and trading session into a
   table of [agent, regime, session, n, win_rate, profit_factor, expectancy].
4. `orchestrator.learn_weights()` turns that table into vote weights, so an agent
   that loses money in ranging markets is quietly outvoted there, while an agent
   with PF 2 in trending markets leads the discussion.

Why score agents separately first
---------------------------------
If we only ever measured the panel we could not tell which agent was contributing
and which was adding noise.  Scoring agents individually on **training data
only** makes the weighting principled rather than a guess - and because the test
period is never touched here, the weights are genuinely out-of-sample when they
are applied later.
"""

from __future__ import annotations

import pandas as pd

from agents.base import AgentVote
from backtest import run_backtest, score_trades
from orchestrator import OrchestratorAgent, PanelDecision, _session_of

METRIC_KEYS = ("n", "win_rate", "profit_factor", "expectancy", "net_R")


def _decision_from_vote(vote: AgentVote, min_score: float = 0.2) -> PanelDecision:
    """Wrap a single agent's vote as a PanelDecision the backtester understands."""
    side = vote.direction.where(
        vote.long_score.gt(min_score) | vote.short_score.gt(min_score), 0)
    score = pd.concat([vote.long_score, vote.short_score], axis=1).max(axis=1)
    agreement = (vote.direction != 0).astype(float)
    zeros = pd.Series(False, index=side.index)
    return PanelDecision(side=side, score=score, agreement=agreement,
                         conflict=zeros, voters_long=agreement,
                         voters_short=agreement)


def _cell_stats(R) -> dict:
    """n / win rate / profit factor / expectancy for an array of R multiples."""
    R = pd.Series(R).to_numpy(dtype=float)
    if len(R) == 0:
        return {"n": 0, "win_rate": 0.0, "profit_factor": 0.0,
                "expectancy": 0.0, "net_R": 0.0}
    wins, losses = R[R > 0], R[R <= 0]
    gw, gl = float(wins.sum()), float(-losses.sum())
    return {
        "n": int(len(R)),
        "win_rate": 100.0 * len(wins) / len(R),
        "profit_factor": (gw / gl) if gl > 0 else float("inf"),
        "expectancy": float(R.mean()),
        "net_R": float(R.sum()),
    }


def evaluate_agents(agents, fs, resolve: pd.DataFrame, labels: pd.Series, *,
                    start: str, end: str, tp_R: float = 2.0,
                    atr_mult: float = 1.5, spread: float = 0.30,
                    fee_rate: float = 0.0,
                    max_hold_bars: int | None = None,
                    friday_cutoff_hour: int | None = 20,
                    min_agent_score: float = 0.2) -> pd.DataFrame:
    """Backtest each agent on its own and summarise by regime and session.

    Returns one row per (agent, regime, session) cell that had trades, plus an
    overall row where regime == -1 and session == 'all'.
    """
    sess = _session_of(fs)
    rows = []

    for agent in agents:
        vote = agent.score(fs)
        dec = _decision_from_vote(vote, min_score=min_agent_score)
        res = run_backtest(dec, fs, resolve, tp_R=tp_R, atr_mult=atr_mult,
                           spread=spread, fee_rate=fee_rate,
                           max_hold_bars=max_hold_bars,
                           friday_cutoff_hour=friday_cutoff_hour,
                           labels=labels, start=start, end=end)
        tr = res.frame
        if tr.empty:
            continue

        entry_times = pd.DatetimeIndex(tr["entry_time"])
        tr = tr.assign(regime=labels.reindex(entry_times).to_numpy(),
                       session=sess.reindex(entry_times).to_numpy())

        rows.append({"agent": agent.name, "regime": -1, "session": "all",
                     **_cell_stats(tr["R"])})

        for r, g in tr.groupby("regime"):
            rows.append({"agent": agent.name, "regime": int(r),
                         "session": "all", **_cell_stats(g["R"])})

        for s, g in tr.groupby("session"):
            rows.append({"agent": agent.name, "regime": -1,
                         "session": str(s), **_cell_stats(g["R"])})

    return (pd.DataFrame(rows)
            .sort_values(["agent", "regime", "session"])
            .reset_index(drop=True))


def summarise_agents(stats: pd.DataFrame) -> pd.DataFrame:
    """One tidy row per agent: overall edge, best and worst regime."""
    if stats.empty:
        return stats
    rows = []
    for agent, g in stats[stats["regime"] == -1].groupby("agent"):
        o = g[g["session"] == "all"]
        if o.empty:
            continue
        by_reg = stats[(stats["agent"] == agent) & (stats["regime"] != -1) &
                       (stats["session"] == "all")].sort_values("profit_factor")
        best = by_reg.iloc[-1] if not by_reg.empty else None
        worst = by_reg.iloc[0] if not by_reg.empty else None
        rows.append({
            "agent": agent,
            "n": int(o["n"].iloc[0]),
            "win_rate": round(o["win_rate"].iloc[0], 1),
            "profit_factor": round(o["profit_factor"].iloc[0], 3),
            "expectancy": round(o["expectancy"].iloc[0], 4),
            "net_R": round(o["net_R"].iloc[0], 1),
            "best_regime": int(best["regime"]) if best is not None else None,
            "best_PF": round(best["profit_factor"], 2) if best is not None else None,
            "worst_regime": int(worst["regime"]) if worst is not None else None,
            "worst_PF": round(worst["profit_factor"], 2) if worst is not None else None,
        })
    return (pd.DataFrame(rows)
            .sort_values("profit_factor", ascending=False)
            .reset_index(drop=True))


def train(agents, fs, resolve, labels, style_cfg: dict, train_start: str,
          train_end: str, tp_R: float = 2.0, friday_cutoff_hour: int | None = 20,
          fee_rate: float = 0.0):
    """Full training pass: measure agents, learn weights, return a ready panel."""
    orch = OrchestratorAgent(
        agents,
        min_agreement=style_cfg.get("min_agreement", 2),
        min_score=style_cfg.get("min_score", 0.35),
    )
    stats = evaluate_agents(agents, fs, resolve, labels,
                            start=train_start, end=train_end, tp_R=tp_R,
                            atr_mult=style_cfg["atr_mult"],
                            spread=style_cfg["spread"], fee_rate=fee_rate,
                            max_hold_bars=style_cfg.get("max_hold_bars"),
                            friday_cutoff_hour=friday_cutoff_hour)
    orch.learn_weights(stats)
    return orch, stats, summarise_agents(stats)

