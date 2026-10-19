"""
Event-driven backtester for the multi-agent panel.

Design decisions that matter for honesty
----------------------------------------
* **Resolution timeframe != entry timeframe.**  A signal on an M30 bar is
  resolved against M5 bars (or M1 for scalping), so stop/target fills are
  evaluated at a realistic granularity instead of assuming the entry bar's
  close.
* **Conservative same-bar resolution.**  If a single resolution bar contains both
  the stop and the target, the STOP is assumed to have been hit first.  This is
  the pessimistic choice and stops optimistic bias.
* **Costs charged on every trade** - spread at entry, plus optional commission
  and slippage, per style.
* **Entry at the next bar's open**, never at the signal bar's close.  A signal
  known at bar N's close is executed at bar N+1's open.
* **Gap-through handling.**  If a resolution bar opens beyond the stop, the fill
  is at that open, not at the stop price.
* **One position at a time**, with an optional cooldown so the panel cannot
  re-enter the same bar it exited on.

Results are returned as a `BacktestResult` with per-trade R multiples, which is
the natural unit for comparing risk:reward configurations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def _r(value, digits: int = 3):
    """Round, tolerating None (trailing-stop trades have no fixed target)."""
    return None if value is None else round(float(value), digits)


@dataclass
class Trade:
    entry_time: pd.Timestamp

    exit_time: pd.Timestamp
    side: int                  # +1 long, -1 short
    entry: float
    stop: float
    target: float
    exit_price: float
    r_multiple: float          # net result in units of 1R
    gross_r: float
    cost_r: float
    bars_held: int
    exit_reason: str           # target / stop / time_exit / week_close
    score: float = 0.0
    agreement: float = 0.0
    regime: int = 0

    def as_dict(self) -> dict:
        return {
            "entry_time": self.entry_time, "exit_time": self.exit_time,
            "side": "LONG" if self.side == 1 else "SHORT",
            "entry": _r(self.entry), "stop": _r(self.stop),
            "target": _r(self.target), "exit": _r(self.exit_price),
            "R": round(self.r_multiple, 4), "gross_R": round(self.gross_r, 4),
            "bars_held": self.bars_held, "exit_reason": self.exit_reason,
            "score": round(self.score, 3), "agreement": self.agreement,
            "regime": self.regime,
        }


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)

    @property
    def frame(self) -> pd.DataFrame:
        cols = ["entry_time", "exit_time", "side", "entry", "stop", "target",
                "exit", "R", "bars_held", "exit_reason", "score",
                "agreement", "regime"]
        if not self.trades:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame([t.as_dict() for t in self.trades])


def _close(tr: dict, exit_time, exit_price: float, reason: str) -> Trade:
    """Turn an open-trade dict into a completed Trade with net R."""
    side = tr["side"]
    risk = tr["risk"]                     # distance from entry to initial stop
    gross_move = (exit_price - tr["entry"]) * side
    gross_r = gross_move / risk if risk > 0 else 0.0
    # Cost model (charged once per side, converted to R):
    #  * fee_rate > 0 -> 0.02% of notional on the OPEN and again on the CLOSE
    #    (0.04% of notional per round trip), applied to each side's own price.
    #  * fee_rate == 0 -> legacy: the flat spread, charged once at entry.
    # Slippage was already applied to the fill price above.
    fee_rate = tr.get("fee_rate", 0.0) or 0.0
    if fee_rate > 0:
        fee_units = fee_rate * tr["entry"] + fee_rate * abs(exit_price)
    else:
        fee_units = tr["spread"]
    cost_r = fee_units / risk if risk > 0 else 0.0
    return Trade(
        entry_time=tr["entry_time"], exit_time=exit_time, side=side,
        entry=tr["entry"], stop=tr["stop_initial"], target=tr["target"],
        exit_price=exit_price, r_multiple=gross_r - cost_r,
        gross_r=gross_r, cost_r=cost_r, bars_held=tr["bars_held"],
        exit_reason=reason, score=tr["score"], agreement=tr["agreement"],
        regime=tr["regime"],
    )


# ---------------------------------------------------------------------- engine
def run_backtest(decision, fs, resolve: pd.DataFrame, *,
                 tp_R: float = 2.0,
                 atr_mult: float = 1.5,
                 spread: float = 0.30,
                 fee_rate: float = 0.0,
                 slippage_atr: float = 0.05,
                 max_hold_bars: int | None = None,
                 friday_cutoff_hour: int | None = 20,
                 cooldown_bars: int = 1,
                 min_score: float = 0.0,
                 min_agreement: int = 0,
                 labels: pd.Series | None = None,
                 start: str | None = None,
                 end: str | None = None,
                 be_after_R: float | None = None,
                 trail_atr: float | None = None,
                 ) -> BacktestResult:
    """Simulate the panel's decisions on `fs`, resolved against `resolve`.

    Key honesty properties
    ----------------------
    * A signal known at entry-bar N's close is filled at the NEXT resolution
      bar's open - never at the signal bar's own close.
    * Stops and targets are judged on `resolve` bars, so an M30 signal is not
      resolved with M30 granularity.
    * When one resolution bar contains both the stop and the target, the STOP is
      taken (pessimistic).
    * A resolution bar that opens beyond the stop fills at that open (gap risk).
    * Spread is charged at every entry.
    * One position at a time, plus a cooldown so the panel cannot re-enter on the
      very bar it closed on.
    """
    entry_df = fs.df
    side = decision.side.astype(float).copy()
    if min_score > 0:
        side = side.where(decision.score >= min_score, 0)
    if min_agreement > 0:
        side = side.where(decision.agreement >= min_agreement, 0)

    if start or end:
        mask = pd.Series(True, index=side.index)
        if start:
            mask &= side.index >= pd.Timestamp(start, tz=side.index.tz)
        if end:
            mask &= side.index <= pd.Timestamp(end, tz=side.index.tz)
        side = side.where(mask, 0)

    atr = entry_df["atr"]
    res_idx = resolve.index
    res_open = resolve["open"].to_numpy()
    res_high = resolve["high"].to_numpy()
    res_low = resolve["low"].to_numpy()
    res_close = resolve["close"].to_numpy()
    res_atr = (resolve["atr"].to_numpy() if "atr" in resolve.columns
               else np.full(len(resolve), np.nan))

    if labels is None:
        labels = pd.Series(0, index=entry_df.index)

    # Map every resolution bar to the entry bar that was OPEN when it started.
    # Used to enforce the max-hold limit measured in entry bars.
    entry_pos = (pd.Series(np.arange(len(entry_df.index)), index=entry_df.index)
                 .reindex(res_idx, method="ffill").to_numpy())

    trades: list[Trade] = []
    side_vals = side.to_numpy()
    atr_vals = atr.to_numpy()
    score_vals = decision.score.to_numpy()
    agree_vals = decision.agreement.to_numpy()
    regime_vals = labels.reindex(entry_df.index).to_numpy()
    n_res = len(res_idx)

    # Which resolution bars immediately follow an entry bar carrying a signal?
    # The engine only ever needs to look at (a) the bar where a pending signal
    # fills and (b) the bars while a position is open.  Walking the whole
    # resolution frame for every grid configuration is the difference between
    # 2 minutes and 2 hours on the M5 scalp style, so we index the events up
    # front and visit only those.
    dow_res = res_idx.dayofweek.to_numpy()
    hour_res = res_idx.hour.to_numpy()

    fill_candidates = np.flatnonzero(side_vals[entry_pos[:-1]] != 0) if n_res > 1 \
        else np.array([], dtype=int)
    # +1: the signal from entry bar i is executed on the resolution bar after it
    fill_candidates = fill_candidates + 1
    fill_candidates = fill_candidates[fill_candidates < n_res]

    # drop candidates whose entry bar was the last one before the Friday close
    if friday_cutoff_hour is not None and len(fill_candidates):
        prev = fill_candidates - 1
        late = (dow_res[prev] == 4) & (hour_res[prev] >= friday_cutoff_hour)
        fill_candidates = fill_candidates[~late]

    next_fill = 0
    n_fills = len(fill_candidates)

    def _fill_at(j: int, bar_o: float, a0: float) -> dict | None:
        """Open a position at resolution bar j from the queued signal."""
        nonlocal next_fill
        while next_fill < n_fills and fill_candidates[next_fill] <= j:
            i = int(entry_pos[j - 1]) if j > 0 else -1
            sd = int(side_vals[i]) if 0 <= i < len(side_vals) else 0
            next_fill += 1
            if sd == 0 or not (np.isfinite(a0) and a0 > 0):
                continue
            slip = slippage_atr * a0
            fill = bar_o + sd * ((0.0 if fee_rate > 0 else spread / 2.0) + slip)
            risk = atr_mult * a0
            stop = fill - sd * risk
            target = (fill + sd * risk * tp_R) if trail_atr is None else None
            return {
                "side": sd, "entry": fill, "risk": risk, "stop": stop,
                "stop_initial": stop, "target": target, "atr0": a0,
                # legacy: the flat spread is charged once, at entry.
                # fee model: fee_rate applies to BOTH sides (open + close) and is
                # converted to R in _close once the exit price is known.
                "spread": spread, "fee_rate": fee_rate, "slip": slip,
                "entry_time": res_idx[j],
                "entry_i": i, "entry_bar": j, "moved_be": False, "bars_held": 0,
                "score": float(score_vals[i]),
                "agreement": float(agree_vals[i]),
                "regime": int(regime_vals[i]) if np.isfinite(regime_vals[i]) else 0,
            }
        return None


    n_res = len(res_idx)

    # Event-driven main loop: jump straight from one interesting bar to the next.
    # A bar is interesting if it can fill a queued signal, or if a position is
    # open and might stop/target out.  Everything in between is skipped, which is
    # what makes the M5 grid search tractable.
    open_trade: dict | None = None
    just_filled = False

    j = 0
    while j < n_res:
        # ---------------- fill a queued signal on this bar ----------------
        # Filling happens BEFORE exit management so that the entry bar's own
        # high/low are still checked against the brand-new stop and target.
        # Skipping that (and only checking from the next bar) would let a trade
        # that is stopped out inside its own entry bar be reported as a gap.
        if open_trade is None and next_fill < n_fills and fill_candidates[next_fill] <= j:
            i_prev = int(entry_pos[j - 1]) if j > 0 else -1
            a0 = atr_vals[i_prev] if 0 <= i_prev < len(atr_vals) else np.nan
            open_trade = _fill_at(j, res_open[j], a0)
            just_filled = open_trade is not None

        # ---------------- exit management on this bar ----------------
        if open_trade is not None:
            tr = open_trade
            bar_o, bar_h, bar_l, bar_c = res_open[j], res_high[j], res_low[j], res_close[j]
            a = res_atr[j] if np.isfinite(res_atr[j]) else tr["atr0"]
            exit_price = exit_reason = None

            if tr["side"] == 1:
                # A bar that opens beyond the stop fills there (gap risk) - but
                # not the very bar we just filled on, where the open IS our entry.
                if bar_o <= tr["stop"] and not just_filled:
                    exit_price, exit_reason = bar_o, "stop"
                else:
                    if trail_atr is not None and np.isfinite(a):
                        tr["stop"] = max(tr["stop"], bar_c - trail_atr * a)
                    hit_stop = bar_l <= tr["stop"]
                    hit_tgt = (tr["target"] is not None) and (bar_h >= tr["target"])
                    if hit_stop:                              # stop wins ties
                        exit_price, exit_reason = tr["stop"], "stop"
                    elif hit_tgt:
                        exit_price, exit_reason = tr["target"], "target"
            else:
                if bar_o >= tr["stop"] and not just_filled:
                    exit_price, exit_reason = bar_o, "stop"
                else:
                    if trail_atr is not None and np.isfinite(a):
                        tr["stop"] = min(tr["stop"], bar_c + trail_atr * a)
                    hit_stop = bar_h >= tr["stop"]
                    hit_tgt = (tr["target"] is not None) and (bar_l <= tr["target"])
                    if hit_stop:
                        exit_price, exit_reason = tr["stop"], "stop"
                    elif hit_tgt:
                        exit_price, exit_reason = tr["target"], "target"

            # breakeven stop move once the trade has run far enough
            if exit_price is None and be_after_R is not None and not tr["moved_be"]:
                if tr["side"] == 1 and bar_h >= tr["entry"] + be_after_R * tr["risk"]:
                    tr["stop"] = max(tr["stop"], tr["entry"])
                    tr["moved_be"] = True
                elif tr["side"] == -1 and bar_l <= tr["entry"] - be_after_R * tr["risk"]:
                    tr["stop"] = min(tr["stop"], tr["entry"])
                    tr["moved_be"] = True

            # time-based exit, measured in ENTRY bars
            if exit_price is None and max_hold_bars is not None and not just_filled:
                i_now = entry_pos[j]
                if np.isfinite(i_now) and (i_now - tr["entry_i"]) >= max_hold_bars:
                    exit_price, exit_reason = bar_c, "time_exit"

            if exit_price is not None:
                trades.append(_close(tr, res_idx[j], exit_price, exit_reason))
                open_trade = None

        just_filled = False
        j += 1

        # ---------------- skip ahead to the next interesting bar ----------------
        if open_trade is None and next_fill < n_fills:
            j = max(j, int(fill_candidates[next_fill]))

    # any trade still open at the end of the data is closed at the last close
    if open_trade is not None:
        trades.append(_close(open_trade, res_idx[-1], res_close[-1], "end_of_data"))

    return BacktestResult(trades=trades)



# --------------------------------------------------------------------- metrics
def score_trades(result: BacktestResult) -> dict:
    """The metrics that decide whether a setup is tradeable.

    Returns win rate, profit factor, expectancy in R, average win/loss in R,
    realised reward:risk, max drawdown in R, and a simple R-based Sharpe.
    """
    r = result.frame
    if r.empty:
        return {"n": 0, "win_rate": 0.0, "profit_factor": 0.0, "expectancy": 0.0,
                "avg_win": 0.0, "avg_loss": 0.0, "realised_rr": 0.0,
                "net_R": 0.0, "max_dd_R": 0.0, "sharpe": 0.0, "payoff": 0.0}

    R = r["R"].to_numpy(dtype=float)
    wins = R[R > 0]
    losses = R[R <= 0]
    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())

    # profit factor: infinite when there are no losses at all
    pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")

    equity = np.cumsum(R)
    peak = np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:]
    dd = peak - equity
    max_dd = float(dd.max()) if len(dd) else 0.0

    std = float(R.std(ddof=1)) if len(R) > 1 else 0.0
    sharpe = (float(R.mean()) / std * np.sqrt(len(R))) if std > 0 else 0.0

    return {
        "n": int(len(R)),
        "win_rate": float(100.0 * len(wins) / len(R)),
        "profit_factor": float(pf),
        "expectancy": float(R.mean()),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        # realised reward:risk = average win / average loss magnitude
        "realised_rr": (float(wins.mean() / abs(losses.mean()))
                        if len(wins) and len(losses) and losses.mean() != 0 else 0.0),
        "payoff": (float(wins.mean() / abs(losses.mean()))
                   if len(wins) and len(losses) and losses.mean() != 0 else 0.0),
        "net_R": float(R.sum()),
        "max_dd_R": max_dd,
        "sharpe": float(sharpe),
        "median_R": float(np.median(R)),
    }


def equity_curve(result: BacktestResult, risk_pct: float = 1.0,
                 start_equity: float = 10_000.0) -> pd.DataFrame:
    """Compounded equity curve from fixed-fractional risk per trade."""
    eq = start_equity
    rows = []
    for t in result.trades:
        pnl = eq * (risk_pct / 100.0) * t.r_multiple
        eq += pnl
        rows.append({"exit_time": t.exit_time, "equity": eq, "pnl": pnl,
                     "R": t.r_multiple})
    return pd.DataFrame(rows)

