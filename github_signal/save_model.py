"""
save_model.py - Train the 9-agent orchestrator and save the learned weights.

Run this ONCE on your laptop:
    cd backend
    python save_model.py

Output: model_weights.pkl (in the same folder)

This pickle contains ONLY the learned numbers (agent weights, regime weights,
session weights) - NOT the agents themselves. The agents are pure Python code
and get rebuilt from source on GitHub, guaranteeing identical logic.
"""

from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from agents import make_agents
from config import STYLES, TRAIN_START, TRAIN_END
from orchestrator import OrchestratorAgent
from training import train


def main():
    from run_scalp_multi import SYMBOLS, load_m1_for
    print("[save_model] loading M1 data ...")
    m1, info = load_m1_for("ETHUSDT")
    print(f"[save_model] M1: {len(m1):,} bars  {m1.index[0]} .. {m1.index[-1]}")

    from data_loader import resample
    m5 = resample(m1, "5min")
    print(f"[save_model] M5: {len(m5):,} bars")

    from features import build_features, merge_higher_tf
    print("[save_model] building features ...")
    fs = build_features("5min", m5)
    for htf_name in ("15min", "1h"):
        htf = resample(m1, htf_name)
        hdf = build_features(htf_name, htf)
        fs = merge_higher_tf(fs, hdf)
    print(f"[save_model] features: {fs.df.shape[1]} cols, {len(fs.df):,} bars")

    from regime import classify
    labels = classify(fs)
    print("[save_model] regimes classified")

    fs_m1 = build_features("1min", m1)
    for htf_name in ("5min", "15min"):
        htf = resample(m1, htf_name)
        hdf = build_features(htf_name, htf)
        fs_m1 = merge_higher_tf(fs_m1, hdf)
    labels_m1 = classify(fs_m1)
    print(f"[save_model] M1 resolution: {fs_m1.df.shape[1]} cols")

    cfg = STYLES["scalp"]
    sym = SYMBOLS["ETHUSDT"]
    print("[save_model] training 9-agent orchestrator ...")
    orch, s, a = train(
        make_agents(), fs, fs_m1, labels,
        dict(cfg, spread=sym["spread"]),
        TRAIN_START, TRAIN_END,
        tp_R=2.0,
        friday_cutoff_hour=sym.get("friday_cutoff_hour"),
    )
    print(f"[save_model] trained. agents={s}")

    weights = {
        "agent_weights":         orch.agent_weights,
        "agent_regime_weights":  orch.agent_regime_weights,
        "session_weights":       orch.session_weights,
        "min_agreement":         orch.min_agreement,
        "min_score":             orch.min_score,
        "conflict_gap":          orch.conflict_gap,
        "use_learned_weights":   orch.use_learned_weights,
        "htf_weight":            orch.htf_weight,
        "train_start":           str(TRAIN_START),
        "train_end":             str(TRAIN_END),
        "symbol":                "ETHUSDT",
        "atr_mult":              cfg["atr_mult"],
        "spread":                sym["spread"],
    }

    out = Path(__file__).parent / "model_weights.pkl"
    with open(out, "wb") as f:
        pickle.dump(weights, f)

    print(f"\n[save_model] saved -> {out}")
    print(f"[save_model] {len(weights['agent_weights'])} agent weights")
    print(f"[save_model] {len(weights['agent_regime_weights'])} regime maps")
    print(f"[save_model] {len(weights['session_weights'])} session maps")
    print(f"[save_model] {out.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
