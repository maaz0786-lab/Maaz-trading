"""
Volume handling - and being honest about it.

The problem
-----------
Only one of our two sources actually carries volume:

* **Dukascopy H1** (8 years) - real tick volume. ~5.9% of bars are zero (dead
  hours), the rest are genuine.
* **HistData M1** (~21 months) - the free "Generic ASCII" files carry **zero
  volume for every single bar**.  Verified: 0 of 613,577 bars have any volume.

Naively computing volume features on the M1 chain would produce NaNs, and the
volume / order-flow agents would silently vote on nothing.  That is exactly the
kind of quiet failure that makes a backtest look meaningful when it is not.

The fix
-------
`tick_activity_volume()` derives a volume proxy from what M1 data *does* contain.
The idea is standard in FX microstructure: when you have no tick count, the
number of distinct price updates and the length of the price path are the best
available stand-ins.  We combine:

* `n_moves`    - how many M1 bars inside the aggregated bar actually moved price
* `path_len`   - summed absolute 1-minute return (the path the price walked)
* `sign_flips` - how often the direction reversed (churn)

The result is a **relative activity index**, normalised to a mean of ~1, which is
exactly what the volume features need (`vol_ratio`, `vol_sma20`, `obv` etc. work
on ratios, not absolute tick counts).

Honesty rules enforced here
---------------------------
* If real volume exists, it is used unchanged - nothing is synthesised.
* If it does not, the proxy is built, and the feature frame records that fact
  (`fs.set("volume_source", ...)`) so every report can state it plainly.
* The proxy is *not* claimed to be volume.  Reports label it "tick-activity
  proxy".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REAL = "real tick volume (Dukascopy)"
PROXY = "tick-activity proxy derived from M1 microstructure (no volume in source)"


def has_real_volume(df: pd.DataFrame, min_nonzero_frac: float = 0.02) -> bool:
    """True when the frame's volume column carries usable information."""
    v = df["volume"]
    if v.isna().all():
        return False
    return float((v > 0).mean()) >= min_nonzero_frac


def tick_activity_volume(m1: pd.DataFrame) -> pd.Series:
    """Per-M1-bar tick-activity index.

    Combines 'did the bar move?', 'how far did price travel?' and 'did direction
    flip?' into one positive activity number per minute bar.
    """
    c = m1["close"]
    h, l, o = m1["high"], m1["low"], m1["open"]

    moved = (c != c.shift(1)).astype(float)                     # price updated?
    rng = (h - l).where(h > l, 0.0)                             # intrabar travel
    body = (c - o).abs()
    path = rng + body                                           # total path length
    flip = (np.sign(c - o) != np.sign((c - o).shift(1))).astype(float)

    # scale path by its own median so the index is dimensionless
    scale = float(path.median()) or 1.0
    activity = moved * (1.0 + path / scale) + flip * 0.25
    return activity.fillna(0.0)


def attach_volume(m1: pd.DataFrame, target: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Make sure `target` has a usable volume column; return (frame, source).

    `target` is an aggregated (resampled) frame.  If its volume is dead, we
    aggregate the M1 tick-activity series up to the same index instead.
    """
    if has_real_volume(target):
        target = target.copy()
        target["volume"] = target["volume"].fillna(0.0)
        return target, REAL

    activity = tick_activity_volume(m1)
    agg = activity.resample(_rule_for(target.index), label="left",
                            closed="left").sum()
    agg = agg.reindex(target.index).fillna(0.0)

    # normalise so mean ~ 1, which is what the volume ratios expect
    mean = float(agg.mean()) or 1.0
    out = target.copy()
    out["volume"] = (agg / mean).astype(float)
    return out, PROXY


def _rule_for(index: pd.DatetimeIndex) -> str:
    """Infer the pandas resample rule from a frame's timestamp spacing."""
    if len(index) < 3:
        return "1h"
    step = index[1] - index[0]
    minutes = step.total_seconds() / 60.0
    if minutes >= 1440:
        return "1D"
    return f"{int(round(minutes))}min"
