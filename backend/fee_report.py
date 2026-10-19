"""Re-price the scalp strategy under a FLAT 0.02%-of-notional fee model.

The original backtests charged a fixed $ spread per trade (e.g. $1 on BTC,
$0.03 on silver) baked into the fill price + R-multiple.  This script removes
that and instead charges 0.02% of the notional entry value as a per-trade fee,
then reports whether each instrument / panel / selected setup is still
profitable (year by year), and shows the old-vs-new comparison.

Because the 9-agent panel decision does NOT depend on costs, we reuse the
cached decision frames for BTC / ETH / silver and only recompute gold's (it has
no cache yet).  No grid search is re-run: the previously OOS-validated setups
are simply re-priced under the new fee.

    python fee_report.py                    # all four, spread vs fee
    python fee_report.py --instruments BTCUSDT XAGUSD
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd

from agents import make_agents
from backtest import run_backtest
from config import RESULTS_DIR, STYLES
from full_report_multi import totals_row, window_table, yearly_breakdown
from run_scalp_multi import SYMBOLS, build_frames
from search import EXIT_ENGINES, REGIME_SETS, SESSION_FILTERS, evaluate_config
from training import train

pd.set_option("display.width", 260)

FEE_RATE = 0.0002          # 0.02% of notional per SIDE (0.04% round trip)
RR_VALUES = {"1:1": 1.0, "1:2": 2.0, "1:3": 3.0}

# instrument -> summary path
GOLD_SUMMARY = RESULTS_DIR / "scalp" / "summary.json"


def _load_decision_cache(path: Path):
    if not path.exists():
        return None
    from orchestrator import PanelDecision
    d = pd.read_parquet(path)
    return PanelDecision(
        side=d["side"], score=d["score"], agreement=d["agreement"],
        conflict=d["conflict"], voters_long=d["voters_long"],
        voters_short=d["voters_short"])


def _cache_decision(dec, path: Path) -> None:
    pd.DataFrame({
        "side": dec.side, "score": dec.score, "agreement": dec.agreement,
        "conflict": dec.conflict, "voters_long": dec.voters_long,
        "voters_short": dec.voters_short,
    }).to_parquet(path)


def prepare(instrument: str) -> dict:
    """Return everything needed to re-price one instrument's scalp run."""
    cfg = STYLES["scalp"]

    if instrument == "GOLD":
        from run import build_style_frames
        fs, resolve, labels, info = build_style_frames("scalp")
        out_dir = RESULTS_DIR / "scalp"
        with open(GOLD_SUMMARY, encoding="utf-8") as f:
            summ = json.load(f)
        spread = STYLES["scalp"]["spread"]
        dec_cache = out_dir / "XAUUSD_decision.parquet"
        decision = _load_decision_cache(dec_cache)
        if decision is None:
            train_style = dict(cfg, spread=spread)
            orch, _s, _a = train(
                make_agents(), fs, resolve, labels, train_style,
                summ["train_window"][0], summ["train_window"][1], tp_R=2.0,
                friday_cutoff_hour=20)
            decision = orch.score(fs, labels=labels)
            _cache_decision(decision, dec_cache)
        display = "Gold (XAU/USD)"
        fch = 20
    else:
        fs, resolve, labels, info = build_frames(instrument)
        sym = SYMBOLS[instrument]
        out_dir = RESULTS_DIR / "multi_scalp" / instrument
        with open(out_dir / "summary.json", encoding="utf-8") as f:
            summ = json.load(f)
        decision = _load_decision_cache(out_dir / "decision.parquet")
        spread = sym["spread"]
        display = sym["display"]
        fch = sym.get("friday_cutoff_hour", 20)

    return {
        "instrument": instrument, "display": display,
        "cfg": cfg, "fs": fs, "resolve": resolve, "labels": labels,
        "decision": decision, "tz": fs.df.index.tz, "out_dir": out_dir,
        "summ": summ, "spread": spread, "friday_cutoff_hour": fch,
    }


def run_pricing(ctx: dict, tp_R, *, min_score=0.0, min_agreement=0,
                session=None, regime=None, exit_name="fixed", cost,
                atr_mult=None):
    """Backtest one configuration under a given cost model (spread | fee)."""
    common = dict(
        tp_R=float(tp_R),
        atr_mult=(ctx["cfg"]["atr_mult"] if atr_mult is None else float(atr_mult)),
        spread=ctx["spread"],
        max_hold_bars=ctx["cfg"]["max_hold_bars"],
        min_score=min_score, min_agreement=min_agreement,
        session_window=SESSION_FILTERS.get(session),
        allowed_regimes=REGIME_SETS.get(regime),
        friday_cutoff_hour=ctx["friday_cutoff_hour"],
        **EXIT_ENGINES[exit_name])
    if cost == "fee":
        common["fee_rate"] = FEE_RATE
    res, _ = evaluate_config(ctx["decision"], ctx["fs"], ctx["resolve"],
                             ctx["labels"], **common)
    return res.frame


def _print_year_block(title, tr_fee, tr_old, tz, test_start):
    """Print year-by-year for the FEE model, plus TRAIN/TEST PF old-vs-new."""
    def split(tr):
        et = pd.DatetimeIndex(tr["entry_time"])
        cut = pd.Timestamp(test_start, tz=tz) if tz else pd.Timestamp(test_start)
        return tr[et < cut], tr[et >= cut]

    print(f"\n  {title}")
    f_tr, f_te = split(tr_fee)
    o_tr, o_te = split(tr_old)
    print("    --- 0.02% FEE model (TRAIN) ---")
    print("    " + window_table(f_tr, "TRAIN").to_string(index=False).replace("\n", "\n    "))
    print("    --- 0.02% FEE model (TEST) ---")
    print("    " + window_table(f_te, "TEST").to_string(index=False).replace("\n", "\n    "))

    def pf_net(sub):
        if sub.empty:
            return ("n=0", "-", "-")
        t = totals_row(sub)
        return (f"n={t['trades']}", f"PF {t['PF']}", f"NetR {t['Net R']}")

    o_trT, o_teT = pf_net(o_tr), pf_net(o_te)
    f_trT, f_teT = pf_net(f_tr), pf_net(f_te)
    print(f"    OLD spread : TRAIN {o_trT[0]} {o_trT[1]} {o_trT[2]}  |  "
          f"TEST {o_teT[0]} {o_teT[1]} {o_teT[2]}")
    print(f"    NEW 0.02%  : TRAIN {f_trT[0]} {f_trT[1]} {f_trT[2]}  |  "
          f"TEST {f_teT[0]} {f_teT[1]} {f_teT[2]}")


def report_instrument(instrument: str) -> None:
    ctx = prepare(instrument)
    summ = ctx["summ"]
    test_start = summ["test_window"][0]
    best = summ["best_setup"]

    print("\n" + "=" * 78)
    print(f" FEE-MODEL RE-PRICING  -  {ctx['display']}  ({instrument})")
    print("=" * 78)
    print(f"  train {summ['train_window'][0]}..{summ['train_window'][1]} | "
          f"test {summ['test_window'][0]}..{summ['test_window'][1]}")
    print(f"  old spread ${ctx['spread']}  ->  NEW flat fee "
          f"{FEE_RATE*100:.2f}% of notional")

    out_dir = ctx["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # panels 1:1 / 1:2 / 1:3 (no filters, fixed exit)
    for label, tp in RR_VALUES.items():
        t = time.time()
        fee = run_pricing(ctx, tp, exit_name="fixed", cost="fee")
        old = run_pricing(ctx, tp, exit_name="fixed", cost="spread")
        if not fee.empty:
            fee.to_csv(out_dir / f"trades_fee_panel_{label.replace(':', '_')}.csv",
                       index=False)
        _print_year_block(f"PANEL 1:{int(tp)}", fee, old, ctx["tz"], test_start)

    # selected setup (with filters + its exit engine + its own stop size)
    t = time.time()
    kw = dict(min_score=float(best["min_score"]),
              min_agreement=int(best["min_agreement"]),
              session=str(best["session"]), regime=str(best["regime"]),
              exit_name=str(best["exit"]), atr_mult=float(best["atr_mult"]))
    fee = run_pricing(ctx, best["rr_target"], cost="fee", **kw)
    old = run_pricing(ctx, best["rr_target"], cost="spread", **kw)
    if not fee.empty:
        fee.to_csv(out_dir / "trades_fee_selected.csv", index=False)
    _print_year_block(
        f"SELECTED SETUP (rr={best['rr_target']} atr={best['atr_mult']} "
        f"score>={best['min_score']} agree>={best['min_agreement']} "
        f"{best['session']}/{best['regime']}/{best['exit']})",
        fee, old, ctx["tz"], test_start)
    print(f"\n  [{time.time()-t:.1f}s]  logs -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fee-model re-pricing report")
    ap.add_argument("--instruments", nargs="+",
                    default=["GOLD", "BTCUSDT", "ETHUSDT", "XAGUSD"])
    args = ap.parse_args()

    print("=" * 78)
    print(" SCALP FEE-MODEL RE-PRICING  (0.02% of notional vs old $ spread)")
    print("=" * 78)
    print(f" instruments : {', '.join(args.instruments)}", flush=True)

    for inst in args.instruments:
        report_instrument(inst)

    print("\ndone. Fee trade logs saved to results_multiagent/.../trades_fee_*.csv")


if __name__ == "__main__":
    main()

