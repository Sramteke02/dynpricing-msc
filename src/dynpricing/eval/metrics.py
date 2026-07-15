"""Evaluation metrics (E5).

For each episode we record cumulative revenue, gross profit, average market
share, and pricing stability (the standard deviation of period-to-period price
*changes* — lower is more stable). Results are averaged across seeds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EpisodeMetrics:
    agent: str
    seed: int
    scenario: str
    revenue: float
    gross_profit: float
    market_share: float
    pricing_stability: float
    n_steps: int
    prices: list = field(default_factory=list)

    def to_row(self) -> dict:
        return {
            "agent": self.agent,
            "seed": self.seed,
            "scenario": self.scenario,
            "revenue": self.revenue,
            "gross_profit": self.gross_profit,
            "market_share": self.market_share,
            "pricing_stability": self.pricing_stability,
            "n_steps": self.n_steps,
        }


def pricing_stability(prices) -> float:
    """Std of consecutive price changes; 0 for a perfectly flat price path."""
    p = np.asarray(prices, dtype=float)
    if p.size < 2:
        return 0.0
    return float(np.std(np.diff(p)))


def aggregate(rows: list[dict]) -> list[dict]:
    """Mean +/- std of each metric per (agent, scenario) across seeds."""
    import pandas as pd

    df = pd.DataFrame(rows)
    if df.empty:
        return []
    metrics = ["revenue", "gross_profit", "market_share", "pricing_stability"]
    grouped = df.groupby(["agent", "scenario"])[metrics].agg(["mean", "std"])
    out = []
    for (agent, scenario), record in grouped.iterrows():
        entry = {"agent": agent, "scenario": scenario}
        for m in metrics:
            entry[f"{m}_mean"] = float(record[(m, "mean")])
            entry[f"{m}_std"] = float(0.0 if np.isnan(record[(m, "std")]) else record[(m, "std")])
        out.append(entry)
    return out
