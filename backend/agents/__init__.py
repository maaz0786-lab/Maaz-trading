"""
The nine specialist agents.

Import `ALL_AGENTS` to get one instance of each, ready to score a feature frame.
"""

from __future__ import annotations

from agents.chart_pattern import ChartPatternAgent
from agents.indicators import IndicatorAgent
from agents.mean_reversion import MeanReversionAgent
from agents.order_flow import OrderFlowAgent
from agents.pattern_fib import PatternFibAgent
from agents.price_action import PriceActionAgent
from agents.sr_breakout import SupportResistanceAgent
from agents.trend_following import TrendFollowingAgent
from agents.volume import VolumeAgent

# Order matters only for reporting; the orchestrator is order-independent.
ALL_AGENT_CLASSES = [
    PriceActionAgent,
    IndicatorAgent,
    SupportResistanceAgent,
    TrendFollowingAgent,
    MeanReversionAgent,
    VolumeAgent,
    OrderFlowAgent,
    PatternFibAgent,
    ChartPatternAgent,
]


def make_agents() -> list:
    """One instance of each of the nine specialist agents."""
    return [cls() for cls in ALL_AGENT_CLASSES]


ALL_AGENTS = make_agents()

__all__ = [
    "PriceActionAgent", "IndicatorAgent", "SupportResistanceAgent",
    "TrendFollowingAgent", "MeanReversionAgent", "VolumeAgent",
    "OrderFlowAgent", "PatternFibAgent", "ChartPatternAgent",
    "ALL_AGENT_CLASSES", "ALL_AGENTS", "make_agents",
]
