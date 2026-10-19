"""
Multi-asset scalp backtest: run the FULL scalp pipeline (9-agent panel +
2,916-configuration setup grid) on BTC, ETH and silver.

    python run_scalp_multi.py                       # all three symbols
    python run_scalp_multi.py --symbols BTCUSDT     # one symbol

Per-symbol differences from the gold pipeline (all handled here):

    BTCUSDT / ETHUSDT  Binance Vision M1, timestamps already UTC, trades 24/7
                       -> NO weekend stripping, NO Friday-evening cutoff.
    XAGUSD             Dukascopy M1 (bid), UTC, forex week
                       -> weekend stripping + Friday cutoff like gold.

Everything else (features, regimes, 9 agents, orchestrator, backtest engine,
grid search, train/test clamping) is the exact gold code path.  Results land in
results_multiagent/multi_scalp/<SYMBOL>/.
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
from backtest import equity_curve, run_backtest, score_trades
from config import (PF_TARGET, RESULTS_DIR, RISK_PER_TRADE_PCT, START_EQUITY,
                    STYLES, TEST_END, TEST_START, TRAIN_END, TRAIN_START)
from data_loader import RESAMPLE_RULES, _strip_weekends, resample
from features import build_features, merge_higher_tf
from regime import classify, regime_stats
from search import search_setups
from training import train
from volume_proxy import PROXY, attach_volume, has_real_volume

RR_VALUES = {"1:1": 1.0, "1:2": 2.0, "1:3": 3.0}

# --------------------------------------------------------------------- symbols
SYMBOLS: dict[str, dict] = {
    "BTCUSDT": {
        "display": "Bitcoin (BTC/USDT)",
        "csv": "BTCUSDT_M1.csv",
        "strip_weekends": False,          # crypto trades 24/7
        "friday_cutoff_hour": None,       # no Friday close for crypto
        "spread": 1.00,                   # USD, conservative retail/spot spread
        "data_source": "Binance Vision BTCUSDT 1m monthly klines (spot)",
        "volume_label": "real traded volume (Binance spot)",
    },
    "ETHUSDT": {
        "display": "Ethereum (ETH/USDT)",
        "csv": "ETHUSDT_M1.csv",
        "strip_weekends": False,
        "friday_cutoff_hour": None,
        "spread": 0.30,                   # USD
        "data_source": "Binance Vision ETHUSDT 1m monthly klines (spot)",
        "volume_label": "real traded volume (Binance spot)",
    },
    "XAGUSD": {
        "display": "Silver (XAG/USD)",
        "csv": "XAGUSD_M1.csv",
        "strip_weekends": True,           # forex week, same as gold
        "friday_cutoff_hour": 20,
        "spread": 0.03,                   # USD
        "data_source": "Dukascopy XAGUSD M1 (bid, real tick volume)",
        "volume_label": "real tick volume (Dukascopy)",
    },
}


# ------------------------------------------------------------------- data load
def load_m1_for(symbol: str, use_cache: bool = True) -> tuple[pd.DataFrame, dict]:
    """M1 OHLCV in UTC for one symbol, with a per-symbol parquet cache."""
    cfg = SYMBOLS[symbol]
    sym_volume_label = cfg["volume_label"]
    csv_path = Path(__file__).parent.parent / "data" / cfg["csv"]
    cache_dir = RESULTS_DIR / "cache" / f"m1_{symbol.lower()}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / "m1_utc.parquet"
    if use_cache and cache.exists():
        df = pd.read_parquet(cache)
        # attrs don't survive parquet - probe the raw CSV for the label
        probe = pd.read_csv(csv_path, usecols=["volume"])
        source = (sym_volume_label if has_real_volume(probe) else PROXY)
    else:
        raw = pd.read_csv(csv_path, parse_dates=["datetime"])
        df = raw.set_index("datetime").sort_index()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        df = df[~df.index.duplicated(keep="first")]
        if cfg["strip_weekends"]:
            df = _strip_weekends(df)
        df = df[(df["high"] >= df["low"]) & (df["close"] > 0) & (df["open"] > 0)]
        df = df[["open", "high", "low", "close", "volume"]].astype("float64")
        df, _src = attach_volume(df, df)
        # relabel Dukascopy's generic "real" string for the Binance feed
        source = sym_volume_label if _src != PROXY else PROXY
        df.attrs["volume_source"] = source
        df.to_parquet(cache)

    # ------------------------------------------------------- data inventory
    idx = df.index
    span_min = int((idx[-1] - idx[0]).total_seconds() // 60) + 1
    if cfg["strip_weekends"]:
        full = pd.date_range(idx[0], idx[-1], freq="1min", tz="UTC")
        expected = len(_strip_weekends(pd.DataFrame(index=full)))
    else:
        expected = span_min
    inventory = {
        "m1_bars": len(df),
        "m1_start": str(idx[0]),
        "m1_end": str(idx[-1]),
        "m1_expected": expected,
        "m1_coverage_pct": round(100.0 * len(df) / expected, 2) if expected else 0.0,
        "volume_source": str(source),
        "csv": cfg["csv"],
        "csv_bytes": csv_path.stat().st_size,
    }
    return df, inventory


def build_frames(symbol: str, use_cache: bool = True):
    """(entry FeatureSet, resolve frame, labels, info dict) for the scalp style."""
    cfg_style = STYLES["scalp"]
    sym = SYMBOLS[symbol]

    m1, inventory = load_m1_for(symbol, use_cache=use_cache)
    frames = {"1min": m1}
    needed = {cfg_style["entry_tf"], cfg_style["resolve_tf"], *cfg_style["higher_tf"]}
    for tf in needed:
        if tf == "1min":
            continue
        frames[tf] = resample(m1, RESAMPLE_RULES[tf])

    entry_df = frames["5min"].copy()
    entry_df.attrs["volume_source"] = inventory["volume_source"]
    fs = build_features("5min", entry_df)
    for htf in cfg_style["higher_tf"]:
        hfs = build_features(htf, frames[htf])
        fs = merge_higher_tf(fs, hfs)

    resolve = frames["1min"]
    if "atr" not in resolve.columns:
        rfs = build_features("1min", resolve.copy())
        resolve = rfs.df
    # run_backtest only needs OHLC + atr on the resolution frame; keeping all
    # ~90 feature columns on millions of M1 rows would triple peak memory.
    resolve = resolve[["open", "high", "low", "close", "atr"]].copy()

    labels = classify(fs)
    info = {
        "symbol": symbol,
        "display": sym["display"],
        "style": "scalp",
        "description": cfg_style["desc"],
        "data_source": sym["data_source"],
        **inventory,
        "entry_tf": "5min",
        "resolve_tf": "1min",
        "higher_tf": list(cfg_style["higher_tf"]),
        "m5_bars": len(frames["5min"]),
        "m15_bars": len(frames["15min"]),
        "h1_bars": len(frames["1h"]),
        "entry_bars": len(fs.df),
        "resolve_bars": len(resolve),
        "start": str(fs.df.index[0]),
        "end": str(fs.df.index[-1]),
        "atr_mult": cfg_style["atr_mult"],
        "spread": sym["spread"],
        "friday_cutoff_hour": sym["friday_cutoff_hour"],
        "max_hold_bars": cfg_style["max_hold_bars"],
        "columns": len(fs.df.columns),
    }
    return fs, resolve, labels, info


# ------------------------------------------------------------------ one symbol
def run_symbol(symbol: str, rr_targets: tuple[str, ...] = ("1:1", "1:2", "1:3"),
               verbose: bool = True) -> dict:
    t0 = time.time()
    cfg_style = STYLES["scalp"]
    sym = SYMBOLS[symbol]
    out_dir = RESULTS_DIR / "multi_scalp" / symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)

    log("")
    log("=" * 78)
    log(f" MULTI-ASSET SCALP  -  {sym['display']}  ({symbol})")
    log("=" * 78)

    fs, resolve, labels, info = build_frames(symbol)
    log(f"  data        : {info['data_source']}")
    log(f"  M1          : {info['m1_bars']:,} bars  "
        f"{info['m1_start'][:10]} .. {info['m1_end'][:10]}  "
        f"({info['m1_coverage_pct']}% of expected, volume={info['volume_source']})")
    log(f"  entry 5min  : {info['m5_bars']:,} bars / {info['entry_bars']:,} with features")
    log(f"  resolution  : M1 {info['resolve_bars']:,} bars | "
        f"M15 {info['m15_bars']:,} | H1 {info['h1_bars']:,}")
    log(f"  spread      : ${sym['spread']}  friday_cutoff={sym['friday_cutoff_hour']}")

    rs = regime_stats(labels)
    log("  regimes     : " + ", ".join(f"{r.regime} {r.pct:.0f}%"
                                       for r in rs.itertuples()))

    # ------------------------------------------- train/test window (clamped)
    first, last = fs.df.index[0], fs.df.index[-1]
    tz = fs.df.index.tz

    def _ts(s):
        return pd.Timestamp(s).tz_localize(tz) if tz is not None else pd.Timestamp(s)

    want_train_start = max(TRAIN_START, str(first.date()))
    want_train_end = min(TRAIN_END, str(last.date()))
    want_test_start = max(TEST_START, str(first.date()))

    if _ts(want_test_start) >= last or _ts(want_train_end) <= _ts(want_train_start):
        split = first + (last - first) * 0.70
        train_start = str(first.date())
        train_end = str(split.date())
        test_start = str((split + pd.Timedelta(minutes=1)).date())
        test_end = str(last.date())
        log("  NOTE: history shorter than the configured windows - 70/30 split")
    else:
        train_start = want_train_start
        train_end = want_train_end
        test_start = want_test_start
        test_end = min(TEST_END, str(last.date()))
        if _ts(train_end) >= _ts(test_start):
            train_end = str((_ts(test_start) - pd.Timedelta(days=1)).date())
    log(f"  train       : {train_start} .. {train_end}")
    log(f"  test        : {test_start} .. {test_end}")

    train_style = dict(cfg_style, spread=sym["spread"])

    # ---------------------------------------------------------------- TRAIN
    t = time.time()
    orch, stats, agent_summary = train(
        make_agents(), fs, resolve, labels, train_style, train_start, train_end,
        tp_R=2.0, friday_cutoff_hour=sym["friday_cutoff_hour"])
    log(f"\n  [TRAIN] 9 agents scored in {time.time() - t:.1f}s")
    log(f"  weights: {orch.weight_source}")
    if not agent_summary.empty:
        log("")
        log(agent_summary.to_string(index=False))
    agent_summary.to_csv(out_dir / "agent_performance.csv", index=False)
    stats.to_csv(out_dir / "agent_regime_stats.csv", index=False)

    # ----------------------------------------------------------------- TEST
    decision = orch.score(fs, labels=labels)
    base_kwargs = dict(atr_mult=cfg_style["atr_mult"], spread=sym["spread"],
                       max_hold_bars=cfg_style["max_hold_bars"], labels=labels,
                       friday_cutoff_hour=sym["friday_cutoff_hour"])
    rows = []
    for label in rr_targets:
        tp = RR_VALUES[label]
        tr = run_backtest(decision, fs, resolve, tp_R=tp,
                          start=train_start, end=train_end, **base_kwargs)
        te = run_backtest(decision, fs, resolve, tp_R=tp,
                          start=test_start, end=test_end, **base_kwargs)
        s_tr, s_te = score_trades(tr), score_trades(te)
        rows.append({
            "rr": label, "tp_R": tp,
            "train_n": s_tr["n"], "train_wr": s_tr["win_rate"],
            "train_pf": s_tr["profit_factor"], "train_netR": s_tr["net_R"],
            "test_n": s_te["n"], "test_wr": s_te["win_rate"],
            "test_pf": s_te["profit_factor"], "test_netR": s_te["net_R"],
            "test_exp": s_te["expectancy"], "test_realised_rr": s_te["realised_rr"],
            "test_maxdd": s_te["max_dd_R"],
        })
    rr_table = pd.DataFrame(rows)
    log("\n  PANEL vs R:R  (train / test)")
    log(rr_table.to_string(index=False))
    rr_table.to_csv(out_dir / "rr_table.csv", index=False)

    # --------------------------------------------------------------- SEARCH
    log("\n  [SEARCH] full grid - 2,916 configurations (ranked in-sample, verified OOS)")
    t = time.time()
    grid, n_evals = search_setups(
        orch, fs, resolve, labels,
        train_start=train_start, train_end=train_end,
        test_start=test_start, test_end=test_end,
        rr_targets=tuple(RR_VALUES[l] for l in rr_targets),
        atr_mults=(1.0, cfg_style["atr_mult"], 2.0, 2.5),
        min_scores=(0.35, 0.5, 0.65), min_agreements=(2, 3, 4),
        sessions=("all", "london", "ny_overlap"),
        regimes=("all", "trending", "ranging"),
        exits=("fixed", "breakeven_1R", "trail_atr"),
        spread=sym["spread"], max_hold_bars=cfg_style["max_hold_bars"],
        min_trades=max(30, cfg_style["min_trades"] // 5), style="scalp",
        friday_cutoff_hour=sym["friday_cutoff_hour"])
    grid_s = time.time() - t
    log(f"  {n_evals} configurations evaluated in {grid_s:.1f}s, "
        f"{len(grid)} had enough trades")

    best = None
    best_rows: dict = {}
    if not grid.empty:
        grid.to_csv(out_dir / "setup_grid.csv", index=False)
        cols = ["rr_target", "atr_mult", "min_score", "min_agreement", "session",
                "regime", "exit", "train_n", "train_wr", "train_pf",
                "test_n", "test_wr", "test_pf"]
        log("\n  top 10 by IN-SAMPLE profit factor:")
        log(grid.head(10)[cols].to_string(index=False))

        solid = grid[grid["test_n"] >= max(30, cfg_style["min_trades"] // 3)]
        pool = solid if not solid.empty else grid
        ranked = pool.sort_values("train_pf", ascending=False)
        best = ranked.iloc[0]
        log("\n  SELECTED SETUP (best in-sample among usable OOS samples):")
        log(ranked.head(1)[cols].to_string(index=False))

        # best per R:R target for the report (strings kept as strings!)
        _str_keys = ("session", "regime", "exit")
        _keys = ("rr_target", "atr_mult", "min_score", "min_agreement",
                 "session", "regime", "exit", "train_n", "train_wr",
                 "train_pf", "test_n", "test_wr", "test_pf")
        for rr in (1.0, 2.0, 3.0):
            sub = pool[pool["rr_target"] == rr]
            if not sub.empty:
                b = sub.sort_values("train_pf", ascending=False).iloc[0]
                row = {}
                for k in _keys:
                    if k not in b.index:
                        row[k] = None
                    elif k in _str_keys:
                        row[k] = str(b[k])
                    else:
                        v = float(b[k])
                        row[k] = v if v == v else None      # NaN -> null
                best_rows[str(rr)] = row

    summary = {
        **info,
        "train_window": [train_start, train_end],
        "test_window": [test_start, test_end],
        "rr_table": rr_table.to_dict("records"),
        "configs_evaluated": int(n_evals),
        "configs_usable": int(len(grid)) if not grid.empty else 0,
        "search_seconds": round(grid_s, 1),
        "best_setup": best.to_dict() if best is not None else None,
        "best_per_rr": best_rows,
        "pf_target": PF_TARGET,
        "risk_per_trade_pct": RISK_PER_TRADE_PCT,
        "start_equity": START_EQUITY,
        "runtime_s": round(time.time() - t0, 1),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    # equity artefact for the selected setup (mirrors run.py)
    if best is not None:
        from search import (EXIT_ENGINES, REGIME_SETS, SESSION_FILTERS,
                            evaluate_config)
        dec_kwargs = dict(
            tp_R=float(best["rr_target"]), atr_mult=float(best["atr_mult"]),
            spread=sym["spread"], max_hold_bars=cfg_style["max_hold_bars"],
            min_score=float(best["min_score"]),
            min_agreement=int(best["min_agreement"]),
            session_window=SESSION_FILTERS[str(best["session"])],
            allowed_regimes=REGIME_SETS[str(best["regime"])],
            friday_cutoff_hour=sym["friday_cutoff_hour"],
            **EXIT_ENGINES[str(best["exit"])])
        te_res, _ = evaluate_config(decision, fs, resolve, labels,
                                    start=test_start, end=test_end, **dec_kwargs)
        eq = equity_curve(te_res)
        if not eq.empty:
            eq.to_csv(out_dir / "equity_curve_test.csv", index=False)

    log(f"\n  saved -> {out_dir}   ({summary['runtime_s']}s)")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-asset scalp backtest")
    ap.add_argument("--symbols", nargs="+", default=list(SYMBOLS),
                    choices=list(SYMBOLS))
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    print("=" * 78)
    print(" MULTI-ASSET SCALP BACKTEST  -  full 2,916-config grid per symbol")
    print("=" * 78)
    print(f" symbols : {', '.join(args.symbols)}", flush=True)

    for sym in args.symbols:
        run_symbol(sym)


if __name__ == "__main__":
    main()
