"""Data-driven dynamic pricing in a competitive e-commerce simulation.

A custom, data-calibrated, Gymnasium-compatible market simulation used to
compare rule-based, gradient-boosting and LLM pricing agents.
"""

from __future__ import annotations

__version__ = "0.1.0"

from dynpricing.env.config import EnvConfig
from dynpricing.env.market_env import MarketEnv, MarketState

__all__ = ["EnvConfig", "MarketEnv", "MarketState", "__version__"]
