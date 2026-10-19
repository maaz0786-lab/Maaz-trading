"""
Load the M1 gold series, convert to UTC, and resample every timeframe the
strategy needs.  Results are cached as parquet so repeated runs are instant.

Timezone handling
-----------------
HistData publishes M1 timestamps in US/Eastern *standard* time (fixed UTC-5,
no DST).  We localise as UTC-5 and convert to UTC, then drop weekends.

    python -c "from data_loader import load_all; d = load_all(); print({k: v.shape for k,v in d.items()})"
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (ALL_TIMEFRAMES, FEED_TZ_OFFSET_HOURS, M1_CSV, RESULTS_DIR,
                    ROOT, WEEK_CLOSE_DOW, WEEK_CLOSE_HOUR, WEEK_OPEN_DOW,
                    WEEK_OPEN_HOUR)

OHLCV = ["open", "high", "low", "close", "volume"]
CACHE_DIR = RESULTS_DIR / "cache"


def _strip_weekends(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the forex trading week (Sun 21:00 UTC -> Fri 21:00 UTC)."""
    dow, hour = df.index.dayofweek, df.index.hour
    keep = np.ones(len(df), dtype=bool)
    keep &= ~((dow == WEEK_CLOSE_DOW) & (hour >= WEEK_CLOSE_HOUR))   # Fri 21:00+
    keep &= ~(dow == 5)                                              # Saturday
    keep &= ~((dow == WEEK_OPEN_DOW) & (hour < WEEK_OPEN_HOUR))      # Sun < 21:00
    return df[keep]


def load_m1(use_cache: bool = True) -> pd.DataFrame:
    """M1 OHLCV in UTC, weekends removed, sorted, de-duplicated."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / "M1_utc.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    df = pd.read_csv(M1_CSV, parse_dates=["datetime"])
    # naive -> fixed UTC-5 -> UTC
    df["datetime"] = (
        df["datetime"].dt.tz_localize(f"Etc/GMT+{-FEED_TZ_OFFSET_HOURS}").dt.tz_convert("UTC")
    )
    df = df.set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = _strip_weekends(df)

    # HistData occasionally emits rows with zero/negative range; drop them.
    df = df[(df["high"] >= df["low"]) & (df["close"] > 0) & (df["open"] > 0)]
    df = df[["open", "high", "low", "close", "volume"]].astype(
        {"open": "float64", "high": "float64", "low": "float64",
         "close": "float64", "volume": "float64"}
    )

    # HistData's free M1 files carry NO volume.  Attach a tick-activity proxy so
    # the volume / order-flow agents have something real to work with.
    from volume_proxy import attach_volume
    df, source = attach_volume(df, df)
    df.attrs["volume_source"] = source
    df.to_parquet(cache)
    return df


# pandas resample rule + which timestamps we allow a bar to be labelled with
RESAMPLE_RULES = {
    "5min": "5min", "15min": "15min", "30min": "30min",
    "1h": "1h", "4h": "4h", "1D": "1D",
}


def resample(m1: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate M1 into a higher timeframe, keeping only complete bars."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last",
           "volume": "sum"}
    if rule == "1D":
        # label daily bars by the UTC session date (Sun open belongs to Monday)
        g = m1.resample("24h", origin="start_day", offset="21h", label="left")
    else:
        g = m1.resample(rule, label="left", closed="left")
    out = g.agg(agg).dropna(subset=["open"])
    # if the M1 volume was a tick-activity proxy, the summed value is already the
    # right higher-TF activity figure - no re-normalisation needed here.
    out["volume"] = out["volume"].fillna(0.0)
    return out


def load_all(use_cache: bool = True, timeframes: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Return {timeframe: DataFrame} - always includes '1min'."""
    tfs = timeframes or ALL_TIMEFRAMES
    m1 = load_m1(use_cache=use_cache)
    out: dict[str, pd.DataFrame] = {}

    for tf in tfs:
        safe = tf.replace("/", "_")
        cache = CACHE_DIR / f"{safe}.parquet"
        if use_cache and cache.exists():
            out[tf] = pd.read_parquet(cache)
            continue
        if tf == "1min":
            out[tf] = m1
        else:
            out[tf] = resample(m1, RESAMPLE_RULES[tf])
        out[tf].to_parquet(cache)

    if "1min" in out:
        out["1min"].to_parquet(CACHE_DIR / "1min.parquet")
    return out


# ------------------------------------------------------------------ 8-year H1
# HistData rate-limits its free M1 feed, so the M1 series only covers the most
# recent ~21 months.  The earlier gold-strategy project downloaded 8 full years
# of XAUUSD H1 from Dukascopy (70,128 bars, 2018-10 -> 2026-09).  That series is
# real, verified gold data, so we reuse it to give the slower styles (intraday,
# swing, position) the full 8-year history they need for training.
#
# IMPORTANT: the M1-derived chain and this H1 chain come from different vendors,
# so they are never mixed inside one run.  A style either uses the Dukascopy
# chain (H1/H4/D1, 8 years) or the HistData M1 chain (M5..H1, ~21 months).
LEGACY_H1_CANDIDATES = [
    ROOT.parent / "gold-strategy" / "data" / "XAUUSD_H1.csv",
    ROOT / "data" / "XAUUSD_H1_dukascopy.csv",
]


def load_legacy_h1() -> pd.DataFrame | None:
    """8 years of Dukascopy XAUUSD H1 (UTC), weekends removed, or None."""
    for path in LEGACY_H1_CANDIDATES:
        if not path.exists():
            continue
        df = pd.read_csv(path, parse_dates=["datetime"])
        df = df.rename(columns={"datetime": "time"}).set_index("time").sort_index()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        df = df[~df.index.duplicated(keep="first")]
        df = _strip_weekends(df)
        df = df[(df["high"] >= df["low"]) & (df["close"] > 0)]
        return df[["open", "high", "low", "close", "volume"]].astype("float64")
    return None


def load_dukascopy_chain(use_cache: bool = True,
                         timeframes=("1h", "4h", "1D")) -> dict[str, pd.DataFrame] | None:
    """H1 plus resampled H4/D1 from the 8-year Dukascopy series."""
    cache = CACHE_DIR / "duka_h1.parquet"
    if use_cache and cache.exists():
        h1 = pd.read_parquet(cache)
    else:
        h1 = load_legacy_h1()
        if h1 is None:
            return None
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        h1.to_parquet(cache)

    out = {"1h": h1}
    for tf in timeframes:
        if tf == "1h":
            continue
        safe = f"duka_{tf}".replace("/", "_")
        c = CACHE_DIR / f"{safe}.parquet"
        if use_cache and c.exists():
            out[tf] = pd.read_parquet(c)
        else:
            out[tf] = resample(h1, RESAMPLE_RULES[tf])
            out[tf].to_parquet(c)
    return out


def load_higher_tf_context(entry_tf: str) -> list[str]:
    """Timeframes above `entry_tf`, for higher-timeframe confluence."""
    order = ALL_TIMEFRAMES
    i = order.index(entry_tf)
    return order[i + 1:]


def slice_period(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df.loc[start:end].copy()


if __name__ == "__main__":
    data = load_all()
    for k, v in data.items():
        print(f"{k:>6}: {len(v):>9,} bars  {v.index[0]} -> {v.index[-1]}")
    print("\nM1 price sanity:", data["1min"]["close"].iloc[0],
          "->", data["1min"]["close"].iloc[-1])
