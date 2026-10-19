"""
Central configuration for the multi-agent gold strategy.

Everything tunable lives here so the agents, orchestrator and backtester all
agree on one set of numbers.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parent
PKG_ROOT = ROOT.parent
DATA_DIR = PKG_ROOT / "data"
M1_CSV = DATA_DIR / "ETHUSDT_M1.csv"
RESULTS_DIR = PKG_ROOT / "database"

# --------------------------------------------------------------------- data
# HistData M1 feed is US/Eastern; shift to UTC.
FEED_TZ_OFFSET_HOURS = -5

# Forex week: Sun 21:00 UTC open -> Fri 21:00 UTC close.
WEEK_OPEN_DOW, WEEK_OPEN_HOUR = 6, 21
WEEK_CLOSE_DOW, WEEK_CLOSE_HOUR = 4, 21

# ------------------------------------------------------------------ regimes
TRADING_SESSIONS_UTC = {
    "sydney": (21, 6),
    "tokyo": (0, 9),
    "london": (7, 16),
    "newyork": (12, 21),
}

# ------------------------------------------------------------------- styles
# Each style runs on its own data chain.  Two chains exist:
#
#   duka   : 8 years of Dukascopy XAUUSD H1, resampled to H1/H4/D1 (50k H1 bars)
#   m1     : ~21 months of HistData M1, resampled to M5/M15/M30/H1
#
# A style NEVER mixes chains: entries, stops, targets and higher-timeframe
# context all come from one vendor's series, so the two feeds are never
# compared against each other inside a single backtest.
#
# name -> entry tf, resolution tf, data chain, ATR stop multiple, max hold
#        (in ENTRY bars), spread in USD, min trades for a trustworthy verdict.
STYLES: dict[str, dict] = {
    "scalp": {
        "entry_tf": "5min",
        "resolve_tf": "1min",
        "higher_tf": ("15min", "1h"),
        "chain": "m1",
        "atr_mult": 1.2,
        "max_hold_bars": 48,        # 48 x 5min = 4 hours
        "spread": 0.25,
        "commission": 0.0,
        "min_trades": 200,
        "desc": "M5 entries, M1 stop/target resolution, held <= 4h",
    },
    "scalp_m1": {
        "entry_tf": "1min",
        "resolve_tf": "1min",
        "higher_tf": ("5min", "15min"),
        "chain": "m1",
        "atr_mult": 1.0,
        "max_hold_bars": 60,        # 60 minutes
        "spread": 0.20,
        "commission": 0.0,
        "min_trades": 200,
        "desc": "M1 entries, held <= 1h - true scalping",
    },
    "intraday": {
        "entry_tf": "30min",
        "resolve_tf": "5min",
        "higher_tf": ("4h",),
        "chain": "m1",
        "atr_mult": 1.5,
        "max_hold_bars": 32,        # 32 x 30min = 16h
        "spread": 0.30,
        "commission": 0.0,
        "min_trades": 200,
        "desc": "M30 entries, M5 resolution, closed same/next day",
    },
    "intraday_h1": {
        "entry_tf": "1h",
        "resolve_tf": "1h",
        "higher_tf": ("4h",),
        "chain": "duka",
        "atr_mult": 1.5,
        "max_hold_bars": 24,        # 24 hours
        "spread": 0.30,
        "commission": 0.0,
        "min_trades": 300,
        "desc": "H1 entries, H1 resolution - the 8-year workhorse",
    },
    "swing": {
        "entry_tf": "4h",
        "resolve_tf": "4h",
        "higher_tf": ("1D",),
        "chain": "duka",
        "atr_mult": 2.0,
        "max_hold_bars": 42,        # 42 x 4h = 7 days
        "spread": 0.40,
        "commission": 0.0,
        "min_trades": 150,
        "desc": "H4 entries, H4 resolution, held up to a week",
    },
    "position": {
        "entry_tf": "1D",
        "resolve_tf": "4h",
        "higher_tf": (),
        "chain": "duka",
        "atr_mult": 2.5,
        "max_hold_bars": 14,        # 14 days
        "spread": 0.50,
        "commission": 0.0,
        "min_trades": 100,
        "desc": "Daily entries, H4 resolution, multi-week holds",
    },
}

# Which styles the runner evaluates by default.
DEFAULT_STYLES = ["scalp", "intraday_h1", "swing", "position"]


# Extra timeframes cached for every style so higher-TF context is available.
ALL_TIMEFRAMES = ["1min", "5min", "15min", "30min", "1h", "4h", "1D"]

# ------------------------------------------------------------ risk:reward set
# Each is (label, take-profit multiple of 1R, breakeven-after-R or None).
RR_TARGETS = {
    "1:1": {"tp_R": 1.0, "be_after_R": None, "trail": None},
    "1:2": {"tp_R": 2.0, "be_after_R": None, "trail": None},
    "1:3": {"tp_R": 3.0, "be_after_R": None, "trail": None},
}

# ------------------------------------------------------------------ training
TRAIN_START = "2018-10-01"
TRAIN_END = "2023-09-30"      # 5 years in-sample
TEST_START = "2023-10-01"     # 3 years out-of-sample
TEST_END = "2026-09-30"

# Walk-forward folds (each trains on everything before, tests on the window).
WALK_FORWARD_FOLDS = [
    ("2018-10-01", "2020-09-30", "2020-10-01", "2021-09-30"),
    ("2018-10-01", "2021-09-30", "2021-10-01", "2022-09-30"),
    ("2018-10-01", "2022-09-30", "2022-10-01", "2023-09-30"),
    ("2018-10-01", "2023-09-30", "2023-10-01", "2024-09-30"),
    ("2018-10-01", "2024-09-30", "2024-10-01", "2025-09-30"),
    ("2018-10-01", "2025-09-30", "2025-10-01", "2026-09-30"),
]

# ---------------------------------------------------------------- acceptance
# The brief asks for profit factor > 2.  We report against this bar honestly.
PF_TARGET = 2.0
MIN_PROFIT_FACTOR_FLOOR = 1.0   # anything below this is discarded outright

# Risk model
RISK_PER_TRADE_PCT = 1.0
START_EQUITY = 10_000.0
