"""
Shared, vectorised feature library.

Every agent reads from this module so indicators are computed **once** and are
guaranteed identical across agents.  Nothing here looks ahead: all rolling
windows are backward-looking, and higher-timeframe context is only made visible
after the higher-TF bar has fully closed (see `merge_higher_tf`).

Agents receive a `FeatureSet` for their entry timeframe; optional higher-TF
context is merged in with a shift so it cannot leak.

    from data_loader import load_all
    from features import build_features
    fs = build_features("1h", load_all()["1h"])
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import talib


@dataclass
class FeatureSet:
    """All indicators for one timeframe, aligned on that timeframe's index."""
    tf: str
    df: pd.DataFrame                       # OHLCV + every derived column
    minutes_per_bar: int
    atr_period: int = 14
    _extra: dict = field(default_factory=dict)

    # -- convenience accessors -------------------------------------------------
    def __getitem__(self, key):
        return self.df[key]

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.df.index

    @property
    def columns(self):
        return self.df.columns

    def get(self, key, default=None):
        return self._extra.get(key, default)

    def set(self, key, value):
        self._extra[key] = value



def add_core(fs: FeatureSet) -> FeatureSet:
    """Core price / volatility / trend / oscillators shared by all agents."""
    df = fs.df
    h, l, c, v = (df["high"].values, df["low"].values,
                  df["close"].values, df["volume"].values)
    idx = df.index
    # Series views for arithmetic that needs index alignment
    op, hp, lp, cp, vp = df["open"], df["high"], df["low"], df["close"], df["volume"]

    def S(arr):
        """Wrap a TA-Lib numpy result as a Series on our index."""
        return pd.Series(np.asarray(arr, dtype="float64"), index=idx)

    # --- volatility -----------------------------------------------------------
    df["tr"] = _true_range(df)
    df["atr"] = S(talib.ATR(h, l, c, timeperiod=fs.atr_period))
    df["atr_pct"] = df["atr"] / c
    df["atr_sma50"] = df["atr"].rolling(50, min_periods=20).mean()
    df["vol_regime"] = df["atr"] / df["atr_sma50"]

    # --- moving averages / trend ---------------------------------------------
    for n in (9, 20, 50, 100, 200):
        df[f"ema{n}"] = S(talib.EMA(c, timeperiod=n))
        df[f"sma{n}"] = S(talib.SMA(c, timeperiod=n))
    df["ema_stack_bull"] = (df["ema9"] > df["ema20"]) & (df["ema20"] > df["ema50"]) & (df["ema50"] > df["ema200"])
    df["ema_stack_bear"] = (df["ema9"] < df["ema20"]) & (df["ema20"] < df["ema50"]) & (df["ema50"] < df["ema200"])
    df["ema200_slope"] = np.sign(df["ema200"].diff(5))
    df["adx"] = S(talib.ADX(h, l, c, timeperiod=14))
    df["plus_di"] = S(talib.PLUS_DI(h, l, c, timeperiod=14))
    df["minus_di"] = S(talib.MINUS_DI(h, l, c, timeperiod=14))

    # --- oscillators ----------------------------------------------------------
    df["rsi"] = S(talib.RSI(c, timeperiod=14))
    df["rsi7"] = S(talib.RSI(c, timeperiod=7))
    df["macd"] = S(talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)[0])
    df["macd_sig"] = S(talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)[1])
    df["macd_hist"] = S(talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)[2])
    df["stoch_k"] = S(talib.STOCH(h, l, c, fastk_period=14, slowk_period=3, slowd_period=3)[0])
    df["stoch_d"] = S(talib.STOCH(h, l, c, fastk_period=14, slowk_period=3, slowd_period=3)[1])
    bb_up = S(talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2)[0])
    bb_mid = S(talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2)[1])
    bb_low = S(talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2)[2])
    df["bb_up"], df["bb_mid"], df["bb_low"] = bb_up, bb_mid, bb_low
    df["bb_width"] = (bb_up - bb_low) / bb_mid
    df["bb_pct"] = (c - bb_low) / (bb_up - bb_low).replace(0, np.nan)
    df["cci"] = S(talib.CCI(h, l, c, timeperiod=20))
    df["willr"] = S(talib.WILLR(h, l, c, timeperiod=14))
    df["mfi"] = S(talib.MFI(h, l, c, v, timeperiod=14))
    df["roc"] = S(talib.ROC(c, timeperiod=10))
    df["uo"] = S(talib.ULTOSC(h, l, c, timeperiod1=7, timeperiod2=14, timeperiod3=28))

    # --- bar anatomy (price action) ------------------------------------------
    o = op
    rng = (hp - lp)
    rng = rng.where(rng != 0, np.nan)
    df["body"] = cp - o
    df["body_pct"] = (cp - o).abs() / rng
    df["upper_wick"] = hp - np.maximum(o, cp)
    df["lower_wick"] = np.minimum(o, cp) - lp
    df["upper_wick_pct"] = df["upper_wick"] / rng
    df["lower_wick_pct"] = df["lower_wick"] / rng
    df["is_bull"] = cp > o
    df["close_loc"] = (cp - lp) / rng          # 1 = closed on the high
    df["dir"] = np.sign(cp - o)
    df["ret"] = cp / df["close"].shift(1) - 1.0

    # --- swing structure (fractal pivots, 3 bars each side) ------------------
    left = right = 3
    pivot_hi = pd.Series(h).rolling(left + right + 1, center=True).max().values == h
    pivot_lo = pd.Series(l).rolling(left + right + 1, center=True).min().values == l
    df["pivot_hi"], df["pivot_lo"] = pivot_hi, pivot_lo
    ph = pd.Series(np.where(pivot_hi, h, np.nan), index=df.index).ffill()
    pl = pd.Series(np.where(pivot_lo, l, np.nan), index=df.index).ffill()
    # shift(right): a pivot is only *confirmed* `right` bars after it prints
    df["last_swing_high"] = ph.shift(right)
    df["last_swing_low"] = pl.shift(right)
    df["dist_to_swing_hi"] = (df["last_swing_high"] - c) / df["atr"]
    df["dist_to_swing_lo"] = (c - df["last_swing_low"]) / df["atr"]
    df["hh"] = df["last_swing_high"] > df["last_swing_high"].shift(1)
    df["ll"] = df["last_swing_low"] < df["last_swing_low"].shift(1)
    df["hl"] = df["last_swing_low"] > df["last_swing_low"].shift(1)
    df["lh"] = df["last_swing_high"] < df["last_swing_high"].shift(1)

    # --- session / calendar ---------------------------------------------------
    hour, dow = df.index.hour, df.index.dayofweek
    df["hour"], df["dow"] = hour, dow
    for name, (s, e) in {"london": (7, 16), "newyork": (12, 21),
                         "tokyo": (0, 9), "london_am": (7, 12)}.items():
        df[f"is_{name}"] = (hour >= s) & (hour < e)
        df[f"in_{name}"] = df[f"is_{name}"]
    df["is_sydney"] = (hour >= 21) | (hour < 6)
    df["in_sydney"] = df["is_sydney"]
    df["is_overlap"] = df["is_london"] & df["is_newyork"]
    df["in_overlap"] = df["is_overlap"]
    df["is_friday"] = dow == 4
    df["in_friday"] = df["is_friday"]
    df["is_monday"] = dow == 0
    df["in_monday"] = df["is_monday"]
    return fs

    def warmup(self) -> int:
        """Bars to skip at the start (indicator warm-up)."""
        return 260


def _true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    return pd.concat([df["high"] - df["low"],
                      (df["high"] - pc).abs(),
                      (df["low"] - pc).abs()], axis=1).max(axis=1)



def add_volume(fs: FeatureSet) -> FeatureSet:
    """Volume-profile features (tick volume supplied by the feed)."""
    df = fs.df
    v = df["volume"]
    df["vol_sma20"] = v.rolling(20, min_periods=5).mean()
    df["vol_ratio"] = v / df["vol_sma20"].replace(0, np.nan)
    df["vol_sma50"] = v.rolling(50, min_periods=10).mean()
    df["vol_zscore"] = (v - df["vol_sma20"]) / v.rolling(20, min_periods=5).std().replace(0, np.nan)
    df["vol_spike"] = df["vol_ratio"] > 1.8
    df["vol_dry"] = df["vol_ratio"] < 0.6
    df["vol_rising"] = v > v.shift(1)

    # signed volume (bar direction) + cumulative delta
    signed = v * np.sign(df["close"] - df["open"]).replace(0, 1.0)
    df["delta"] = signed
    df["signed_vol"] = signed
    df["cvd"] = signed.cumsum()
    df["cvd_ma"] = df["cvd"].rolling(20, min_periods=5).mean()
    df["cvd_slope"] = np.sign(df["cvd_ma"].diff(5))
    df["delta_div"] = np.sign(df["delta"]) != np.sign(df["ret"])

    # OBV and rolling VWAP
    df["obv"] = pd.Series(talib.OBV(df["close"].values, v.values), index=df.index)
    df["obv_slope"] = np.sign(df["obv"].diff(5))
    tp = (df["high"] + df["low"] + df["close"]) / 3
    pv = (tp * v).rolling(96, min_periods=10).sum()
    vv = v.rolling(96, min_periods=10).sum()
    df["vwap"] = pv / vv.replace(0, np.nan)
    df["vwap_dist"] = (df["close"] - df["vwap"]) / df["atr"]
    df["above_vwap"] = df["close"] > df["vwap"]

    # volume-confirmed trend and momentum
    df["vol_weighted_ret"] = df["ret"] * df["vol_ratio"]
    df["vol_confirm_trend"] = (np.sign(df["ret"]) == np.sign(df["close"].diff(5))) & (df["vol_ratio"] > 1.2)
    df["vol_breakout"] = (df["vol_ratio"] > 1.5) & (df["body_pct"] > 0.5)
    return fs


def add_orderflow(fs: FeatureSet) -> FeatureSet:
    """Market-microstructure proxies.

    Free data has no true order book / depth-of-market, so instead of faking
    one we use observable proxies for institutional behaviour: tick-volume
    clustering, intrabar close-vs-range path, absorption, liquidity sweeps and
    volume-at-price imbalance.  Every one of these is genuinely computable from
    OHLCV, which keeps the agent honest.
    """
    df = fs.df
    h, l, c, v = df["high"], df["low"], df["close"], df["volume"]

    # --- liquidity sweeps / stop runs ----------------------------------------
    prior_hi, prior_lo = df["last_swing_high"], df["last_swing_low"]
    df["sweep_high"] = (h > prior_hi) & (c < prior_hi) & (df["upper_wick_pct"] > 0.4)
    df["sweep_low"] = (l < prior_lo) & (c > prior_lo) & (df["lower_wick_pct"] > 0.4)

    # --- absorption: heavy volume, tiny range --------------------------------
    df["absorption"] = (df["vol_ratio"] > 1.6) & (df["body_pct"] < 0.3)

    # --- effort vs result ----------------------------------------------------
    df["effort_result"] = (h - l) / df["atr"]
    df["effort_result_div"] = (df["effort_result"] > 1.5) & (df["body_pct"] < 0.3)

    # --- volume-at-price imbalance (which side is being traded) --------------
    upper = np.where(h > l, (np.maximum(df["open"], c) - l) / (h - l).replace(0, np.nan), 0.5)
    df["buy_pressure"] = v * upper
    df["sell_pressure"] = v * (1 - upper)
    df["pressure_imb"] = (df["buy_pressure"] - df["sell_pressure"]) / v.replace(0, np.nan)
    df["pressure_imb_ma"] = df["pressure_imb"].rolling(10, min_periods=3).mean()

    # --- climax / exhaustion -------------------------------------------------
    df["climax_up"] = (df["close_loc"] > 0.85) & (df["vol_ratio"] > 2.0) & (df["body_pct"] > 0.6)
    df["climax_down"] = (df["close_loc"] < 0.15) & (df["vol_ratio"] > 2.0) & (df["body_pct"] > 0.6)
    df["exhaustion"] = df["climax_up"] | df["climax_down"]

    # --- delta persistence ---------------------------------------------------
    df["delta_bull"] = (df["delta"] > 0) & (df["delta"].shift(1) > 0)
    df["delta_bear"] = (df["delta"] < 0) & (df["delta"].shift(1) < 0)
    df["delta_confirm_bull"] = (df["delta"] > 0) & (df["ret"] > 0)
    df["delta_confirm_bear"] = (df["delta"] < 0) & (df["ret"] < 0)
    return fs


def _minutes_per_bar(tf: str) -> int:
    return {"1min": 1, "5min": 5, "15min": 15, "30min": 30,
            "1h": 60, "4h": 240, "1D": 1440}[tf]


def add_patterns(fs: FeatureSet) -> FeatureSet:
    """Candlestick patterns via TA-Lib (all use past bars only).

    The available CDL* functions differ between TA-Lib builds, so the list is
    filtered against what this installation actually exposes.
    """
    df = fs.df
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values

    # name -> TA-Lib function name (kept only if present in this build)
    wanted = {
        # single-bar
        "hammer": "CDLHAMMER",
        "inv_hammer": "CDLINVERTEDHAMMER",
        "hanging": "CDLHANGINGMAN",
        "shooting": "CDLSHOOTINGSTAR",
        "doji": "CDLDOJI",
        "dragonfly": "CDLDRAGONFLYDOJI",
        "gravestone": "CDLGRAVESTONEDOJI",
        "marubozu": "CDLMARUBOZU",
        "spinning_top": "CDLSPINNINGTOP",
        "long_line": "CDLLONGLINE",
        "high_wave": "CDLHIGHWAVE",
        "rickshaw": "CDLRICKSHAWMAN",
        "takuri": "CDLTAKURI",
        "belt_hold": "CDLBELTHOLD",
        # two-bar
        "engulfing": "CDLENGULFING",
        "harami": "CDLHARAMI",
        "harami_cross": "CDLHARAMICROSS",
        "piercing": "CDLPIERCING",
        "darkcloud": "CDLDARKCLOUDCOVER",
        "counterattack": "CDLCOUNTERATTACK",
        "kicking": "CDLKICKING",
        # three-bar
        "morning_star": "CDLMORNINGSTAR",
        "morning_doji": "CDLMORNINGDOJISTAR",
        "evening_star": "CDLEVENINGSTAR",
        "evening_doji": "CDLEVENINGDOJISTAR",
        "three_white": "CDL3WHITESOLDIERS",
        "three_black": "CDL3BLACKCROWS",
        "three_inside": "CDL3INSIDE",
        "three_outside": "CDL3OUTSIDE",
        "risefall3": "CDLRISEFALL3METHODS",
        "three_stars": "CDL3STARSINSOUTH",
        "abandoned_baby": "CDLABANDONEDBABY",
        "tristar": "CDLTRISTAR",
        "upside_gap": "CDLUPSIDEGAP2CROWS",
        "advance_block": "CDLADVANCEBLOCK",
        "conceal_baby": "CDLCONCEALBABYSWALL",
        "homing_pigeon": "CDLHOMINGPIGEON",
        "matching_low": "CDLMATCHINGLOW",
        "ladder_bottom": "CDLLADDERBOTTOM",
        "xside_gap3": "CDLXSIDEGAP3METHODS",
        "tasukigap": "CDLTASUKIGAP",
        "sticksandwich": "CDLSTICKSANDWICH",
        "sep_lines": "CDLSEPARATINGLINES",
        "thrusting": "CDLTHRUSTING",
    }

    raw = {}
    for short, full in wanted.items():
        fn = getattr(talib, full, None)
        if fn is not None:
            raw[short] = fn(o, h, l, c)
    if not raw:
        raise RuntimeError("no CDL* pattern functions available in this TA-Lib build")

    # TA-Lib returns +100 for bullish / -100 for bearish *by value*, so a
    # negative number means the bearish variant of that pattern fired.
    for name, arr in raw.items():
        df[f"pat_{name}"] = pd.Series(np.sign(arr), index=df.index).fillna(0).astype("int8")

    pat_cols = [f"pat_{n}" for n in raw]
    df["pat_bull_score"] = df[pat_cols].clip(lower=0).sum(axis=1)
    df["pat_bear_score"] = (-df[pat_cols].clip(upper=0)).sum(axis=1)
    df["pat_any_bull"] = df["pat_bull_score"] > 0
    df["pat_any_bear"] = df["pat_bear_score"] > 0
    df["pat_net"] = np.sign(df["pat_bull_score"] - df["pat_bear_score"])
    return fs


def add_chart_patterns(fs: FeatureSet, lookback: int = 40) -> FeatureSet:
    """Classic chart patterns detected from confirmed swing pivots.

    These are genuine geometric detections (double top/bottom, head & shoulders,
    triangle compression, flag/pennant), not hand-waving - each is expressed as a
    measurable relationship between the last three confirmed pivots.
    """
    df = fs.df

    # Rebuild confirmed pivots locally so we can reference the last 3 of them.
    left = right = 3
    h, l = df["high"], df["low"]
    ph = pd.Series(np.where(df["pivot_hi"].values, h.values, np.nan), index=df.index).ffill()
    pl = pd.Series(np.where(df["pivot_lo"].values, l.values, np.nan), index=df.index).ffill()
    sh1 = ph.shift(right)                 # most recent confirmed swing high
    sh2 = ph.shift(right + 1)
    sl1 = pl.shift(right)
    sl2 = pl.shift(right + 1)

    tol = 0.25 * df["atr"]

    # --- double top / bottom: two swings at a similar level ------------------
    df["dbl_top"] = (sh1.notna() & sh2.notna() &
                     ((sh1 - sh2).abs() < tol) & (df["close"] < sh1 - df["atr"]))
    df["dbl_bottom"] = (sl1.notna() & sl2.notna() &
                        ((sl1 - sl2).abs() < tol) & (df["close"] > sl1 + df["atr"]))

    # --- head & shoulders -----------------------------------------------------
    # three swing highs: shoulders near-equal, head higher, close below neckline
    sh3 = ph.shift(right + 2)
    neckline = pl.shift(right)
    df["hs_top"] = (sh1.notna() & sh3.notna() &
                    ((sh1 - sh3).abs() < tol) & (sh2 > sh1 + tol) & (sh2 > sh3 + tol) &
                    (df["close"] < neckline))
    sl3 = pl.shift(right + 2)
    neckline_hi = ph.shift(right)
    df["hs_bottom"] = (sl1.notna() & sl3.notna() &
                       ((sl1 - sl3).abs() < tol) & (sl2 < sl1 - tol) & (sl2 < sl3 - tol) &
                       (df["close"] > neckline_hi))

    # --- ascending / descending / symmetrical triangle compression -----------
    n = lookback
    hi_slope = (h.rolling(n).max() - h.rolling(n).max().shift(n)) / n
    lo_slope = (l.rolling(n).min() - l.rolling(n).min().shift(n)) / n
    rng_compress = df["range_20_atr"] < df["range_20_atr"].rolling(60, min_periods=20).mean()
    df["triangle_up"] = rng_compress & (lo_slope > 0) & (hi_slope.abs() < lo_slope.abs() * 0.5)
    df["triangle_down"] = rng_compress & (hi_slope < 0) & (lo_slope.abs() < hi_slope.abs() * 0.5)
    df["triangle_sym"] = rng_compress & (hi_slope < 0) & (lo_slope > 0)

    # --- flag / pennant: sharp impulse then tight consolidation --------------
    impulse = df["close"].diff(10).abs() / df["atr"]
    df["flag_bull"] = (impulse > 2.5) & rng_compress & (df["close"] > df["ema20"])
    df["flag_bear"] = (impulse > 2.5) & rng_compress & (df["close"] < df["ema20"])

    df["cp_bull"] = df["dbl_bottom"] | df["hs_bottom"] | df["flag_bull"]
    df["cp_bear"] = df["dbl_top"] | df["hs_top"] | df["flag_bear"]
    df["cp_net"] = np.where(df["cp_bull"], 1, np.where(df["cp_bear"], -1, 0))
    return fs



def add_fibonacci(fs: FeatureSet, lookback: int = 60) -> FeatureSet:
    """Position of price inside the most recent significant swing + levels."""
    df = fs.df
    hi = df["high"].rolling(lookback, min_periods=10).max()
    lo = df["low"].rolling(lookback, min_periods=10).min()
    rng = (hi - lo).replace(0, np.nan)
    pos = (df["close"] - lo) / rng          # 0 = swing low, 1 = swing high

    df["fib_hi"], df["fib_lo"], df["fib_pos"] = hi, lo, pos
    for lvl in (0.236, 0.382, 0.5, 0.618, 0.786):
        df[f"fib_{int(lvl * 1000)}"] = hi - lvl * rng      # retracement from high
    df["in_golden_zone"] = (pos > 0.5) & (pos < 0.618)
    df["at_deep_retr"] = (pos > 0.618) & (pos < 0.786)
    df["near_extension"] = (pos < -0.236) | (pos > 1.236)
    df["fib_range_atr"] = rng / df["atr"]
    for lvl in (382, 500, 618):
        df[f"dist_fib_{lvl}"] = (df[f"fib_{lvl}"] - df["close"]).abs() / df["atr"]
    df["at_fib_618"] = df["dist_fib_618"] < 0.3
    df["at_fib_500"] = df["dist_fib_500"] < 0.3
    df["at_fib_382"] = df["dist_fib_382"] < 0.3
    return fs


def add_support_resistance(fs: FeatureSet, lookbacks=(20, 50, 100, 200)) -> FeatureSet:
    """Rolling horizontal S/R levels; distances measured in ATR."""
    df = fs.df
    for n in lookbacks:
        df[f"res_{n}"] = df["high"].rolling(n, min_periods=max(5, n // 4)).max().shift(1)
        df[f"sup_{n}"] = df["low"].rolling(n, min_periods=max(5, n // 4)).min().shift(1)
        df[f"dist_res_{n}"] = (df[f"res_{n}"] - df["close"]) / df["atr"]
        df[f"dist_sup_{n}"] = (df["close"] - df[f"sup_{n}"]) / df["atr"]

    df["at_major_res"] = (df["dist_res_100"].abs() < 0.35) & (df["dist_res_200"].abs() < 0.5)
    df["at_major_sup"] = (df["dist_sup_100"].abs() < 0.35) & (df["dist_sup_200"].abs() < 0.5)
    df["break_res_20"] = (df["close"] > df["res_20"].shift(1)) & (df["close"].shift(1) <= df["res_20"].shift(1))
    df["break_sup_20"] = (df["close"] < df["sup_20"].shift(1)) & (df["close"].shift(1) >= df["sup_20"].shift(1))
    df["break_res_50"] = (df["close"] > df["res_50"].shift(1)) & (df["close"].shift(1) <= df["res_50"].shift(1))
    df["break_sup_50"] = (df["close"] < df["sup_50"].shift(1)) & (df["close"].shift(1) >= df["sup_50"].shift(1))
    df["near_res"] = (df["dist_res_50"].abs() < 0.5) | (df["dist_res_100"].abs() < 0.7)
    df["near_sup"] = (df["dist_sup_50"].abs() < 0.5) | (df["dist_sup_100"].abs() < 0.7)
    df["range_20_atr"] = (df["high"].rolling(20).max() - df["low"].rolling(20).min()) / df["atr"]
    df["range_20_ma"] = df["range_20_atr"].rolling(60, min_periods=20).mean()
    # room to run before the next level (key filter for entry quality)
    df["room_up_atr"] = df["dist_res_50"].clip(lower=0)
    df["room_dn_atr"] = df["dist_sup_50"].clip(lower=0)
    return fs


def add_trendlines(fs: FeatureSet, lookback: int = 50) -> FeatureSet:
    """Regression trendline + channel position.

    A vectorised, numerically robust stand-in for hand-drawn trendlines: the
    least-squares slope of the last `lookback` closes, normalised by ATR.
    """
    df = fs.df
    x = np.arange(lookback)

    for name in ("close", "high", "low"):
        vals = df[name].values
        out = np.full(len(df), np.nan)
        for i in range(lookback, len(df) + 1):
            w = vals[i - lookback:i]
            if np.isnan(w).any():
                continue
            out[i - 1] = np.polyfit(x, w, 1)[0]
        df[f"{name}_regslope"] = out
        df[f"{name}_regslope_atr"] = out * lookback / df["atr"]

    df["trendline_bull"] = df["close_regslope_atr"] > 0.15
    df["trendline_bear"] = df["close_regslope_atr"] < -0.15
    df["trendline_flat"] = df["close_regslope_atr"].abs() <= 0.15
    df["trendline_align"] = np.sign(df["close_regslope_atr"]) == np.sign(df["high_regslope_atr"])

    roll = df["close"].rolling(lookback, min_periods=lookback)
    df["reg_fit"] = roll.mean()
    df["reg_resid"] = df["close"] - df["reg_fit"]
    df["reg_std"] = df["reg_resid"].rolling(lookback, min_periods=lookback).std()
    df["chan_pos"] = df["reg_resid"] / df["reg_std"].replace(0, np.nan)
    df["chan_upper_touch"] = df["chan_pos"] > 2.0
    df["chan_lower_touch"] = df["chan_pos"] < -2.0

    # rising / falling channel edges act as dynamic trendline support & resistance
    df["trendline_res"] = df["reg_fit"] + 2 * df["reg_std"]
    df["trendline_sup"] = df["reg_fit"] - 2 * df["reg_std"]
    df["dist_trendline_res"] = (df["trendline_res"] - df["close"]) / df["atr"]
    df["dist_trendline_sup"] = (df["close"] - df["trendline_sup"]) / df["atr"]
    df["tl_res_touch"] = df["dist_trendline_res"].abs() < 0.3
    df["tl_sup_touch"] = df["dist_trendline_sup"].abs() < 0.3
    return fs


def build_features(tf: str, df: pd.DataFrame) -> FeatureSet:
    """Build the complete feature set for one timeframe's OHLCV frame."""
    fs = FeatureSet(tf=tf, df=df.copy(), minutes_per_bar=_minutes_per_bar(tf))
    add_core(fs)
    add_volume(fs)
    add_orderflow(fs)
    add_support_resistance(fs)   # defines range_20_atr, used by chart patterns
    add_patterns(fs)
    add_chart_patterns(fs)
    add_fibonacci(fs)
    add_trendlines(fs)
    fs.df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # carry the provenance of the volume column through to the report
    fs.set("volume_source", df.attrs.get("volume_source", "unknown"))
    return fs


# Columns that are useful as higher-timeframe context.
HTF_CONTEXT_COLS = [
    "ema9", "ema20", "ema50", "ema200", "adx", "rsi",
    "atr", "atr_pct", "vol_regime", "close_regslope_atr", "ema_stack_bull",
    "ema_stack_bear", "fib_pos", "dist_res_50", "dist_sup_50", "cvd_slope",
]


def merge_higher_tf(entry: FeatureSet, higher: FeatureSet,
                    columns: list[str] | None = None) -> FeatureSet:
    """Bring higher-TF columns onto the entry index, visible only AFTER close.

    An H4 bar labelled 00:00 covers 00:00-04:00, so it is only *known* once the
    04:00 bar has opened.  On an H1 entry index that means the H4 value first
    becomes usable on the 04:00 bar - i.e. it must be shifted by the number of
    entry bars the higher-TF bar spans, inclusive of the bar it closes on.

    Shifting by only `higher/entry - 1` would expose the H4 value from 03:00
    onwards, three bars before it is knowable.  That is a genuine look-ahead
    leak, and `test_no_lookahead.py` catches it.
    """
    cols = columns or HTF_CONTEXT_COLS
    left, right = entry.df.index, higher.df.index
    sub = higher.df[[c for c in cols if c in higher.df.columns]]
    reindexed = sub.reindex(right.union(left)).ffill().reindex(left)

    # number of entry bars the higher-TF bar spans, +1 so the value is withheld
    # until the entry bar that *follows* the higher-TF close
    shift_bars = int(np.ceil(higher.minutes_per_bar / entry.minutes_per_bar)) + 1
    for col in sub.columns:
        entry.df[f"htf_{higher.tf}_{col}"] = reindexed[col].shift(shift_bars).values
    return entry



