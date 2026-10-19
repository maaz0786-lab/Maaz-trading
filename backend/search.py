"""
Setup search: find the best configuration for every R:R target and style.

This is the "find the best setup" engine.  It is a grid search over the knobs that
genuinely change the economics of a setup, evaluated honestly:

* **Train, then verify.**  Every configuration is scored on the training window,
  ranked, and the survivors are re-scored on the test window.  A configuration
  that only works in-sample is reported as such and never promoted.
* **Minimum sample size.**  Configurations with too few trades are discarded -
  a PF of 6.0 on 4 trades is noise, not an edge.
* **Multiple-comparison guard.**  Because we test hundreds of combinations, the
  report states how many were tried.  A single best-of-hundreds result on the
  test set is still just one result, and the code refuses to hide that.

Search dimensions
-----------------
entry timeframe, resolution timeframe, ATR stop multiple, consensus threshold,
minimum agent agreement, session filter, regime filter, exit engine
(fixed TP / breakeven / ATR trail).
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from backtest import run_backtest, score_trades
from regime import Regime

# Exit engines: label -> kwargs for run_backtest
EXIT_ENGINES = {
    "fixed": {},
    "breakeven_1R": {"be_after_R": 1.0},
    "trail_atr": {"trail_atr": 2.0},
}

SESSION_FILTERS = {
    "all": None,
    "london": (7, 16),
    "ny_overlap": (12, 16),
    "london_am": (7, 12),
}


def _session_mask(fs, window) -> pd.Series | None:
    if window is None:
        return None
    lo, hi = window
    hour = fs.df.index.hour
    return pd.Series((hour >= lo) & (hour < hi), index=fs.df.index)


def evaluate_config(decision, fs, resolve, labels, *, tp_R, atr_mult, spread,
                    max_hold_bars, min_score, min_agreement, session_window,
                    allowed_regimes, **exit_kwargs):
    """Backtest one configuration and return (trades_frame, metrics)."""
    side = decision.side.copy()
    if min_score > 0:
        side = side.where(decision.score >= min_score, 0)
    if min_agreement > 0:
        side = side.where(decision.agreement >= min_agreement, 0)

    mask = side != 0
    if session_window is not None:
        sm = _session_mask(fs, session_window)
        mask = mask & sm.reindex(side.index).fillna(False)
    if allowed_regimes is not None:
        lr = labels.reindex(side.index)
        mask = mask & lr.isin(list(allowed_regimes))

    filtered_side = side.where(mask, 0)

    from orchestrator import PanelDecision
    dec2 = PanelDecision(side=filtered_side, score=decision.score,
                         agreement=decision.agreement,
                         conflict=decision.conflict,
                         voters_long=decision.voters_long,
                         voters_short=decision.voters_short)

    res = run_backtest(dec2, fs, resolve, tp_R=tp_R, atr_mult=atr_mult,
                       spread=spread, max_hold_bars=max_hold_bars,
                       labels=labels, **exit_kwargs)
    return res, score_trades(res)


REGIME_SETS = {
    "all": None,
    "trending": {Regime.TREND_UP, Regime.TREND_DOWN},
    "ranging": {Regime.RANGE},
    "trend_break": {Regime.TREND_UP, Regime.TREND_DOWN, Regime.BREAKOUT},
}


def search_setups(orch, fs, resolve, labels, *, train_start, train_end,
                  test_start=None, test_end=None, rr_targets=(1.0, 2.0, 3.0),
                  atr_mults=(1.0, 1.5, 2.0), min_scores=(0.30, 0.45, 0.60),
                  min_agreements=(2, 3), sessions=("all", "london", "ny_overlap"),
                  regimes=("all", "trending", "ranging"),
                  exits=("fixed", "breakeven_1R", "trail_atr"),
                  spread=0.30, fee_rate: float = 0.0, max_hold_bars=None, min_trades=100,
                  style="intraday", friday_cutoff_hour: int | None = 20):
    """Grid-search configurations on the training window, verify on the test window.

    Returns a DataFrame sorted by training profit factor, with the out-of-sample
    columns attached for every configuration.
    """
    # score the panel once - the grid only changes risk/filters, not the signals
    decision = orch.score(fs, labels=labels)

    combos = list(itertools.product(rr_targets, atr_mults, min_scores,
                                    min_agreements, sessions, regimes, exits))
    rows = []
    n_evals = 0

    for tp_R, atr_mult, min_score, min_agr, sess_name, reg_name, exit_name in combos:
        common = dict(
            tp_R=tp_R, atr_mult=atr_mult, spread=spread, fee_rate=fee_rate,
            max_hold_bars=max_hold_bars, min_score=min_score,
            min_agreement=min_agr,
            session_window=SESSION_FILTERS[sess_name],
            allowed_regimes=REGIME_SETS[reg_name],
            friday_cutoff_hour=friday_cutoff_hour,
            **EXIT_ENGINES[exit_name],
        )
        # ---------------- in-sample ----------------
        _, tr_m = evaluate_config(decision, fs, resolve, labels,
                                  start=train_start, end=train_end, **common)
        n_evals += 1
        if tr_m["n"] < min_trades:
            continue

        row = {
            "style": style, "rr_target": tp_R, "atr_mult": atr_mult,
            "min_score": min_score, "min_agreement": min_agr,
            "session": sess_name, "regime": reg_name, "exit": exit_name,
            "train_n": tr_m["n"], "train_wr": tr_m["win_rate"],
            "train_pf": tr_m["profit_factor"], "train_exp": tr_m["expectancy"],
            "train_netR": tr_m["net_R"], "train_maxdd": tr_m["max_dd_R"],
            "train_realised_rr": tr_m["realised_rr"],
        }

        # ---------------- out-of-sample ----------------
        if test_start:
            _, te_m = evaluate_config(decision, fs, resolve, labels,
                                      start=test_start, end=test_end, **common)
            row.update({
                "test_n": te_m["n"], "test_wr": te_m["win_rate"],
                "test_pf": te_m["profit_factor"], "test_exp": te_m["expectancy"],
                "test_netR": te_m["net_R"], "test_maxdd": te_m["max_dd_R"],
                "test_realised_rr": te_m["realised_rr"],
            })
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df, n_evals
    return df.sort_values("train_pf", ascending=False).reset_index(drop=True), n_evals

