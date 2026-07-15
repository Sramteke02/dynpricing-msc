"""Layer 2: decision agents, all obeying the shared interface in ``base``."""

from dynpricing.agents.base import Agent, action_to_reach_price
from dynpricing.agents.baselines import (
    FixedPriceAgent,
    CostPlusAgent,
    CompetitorMatchingAgent,
    RandomAgent,
    OracleAgent,
)
from dynpricing.agents.gbm_agent import GradientBoostingAgent
from dynpricing.agents.llm_agent import LLMAgent
from dynpricing.agents.oracle_dp import BackwardInductionOracle
from dynpricing.agents.registry import build_agent, AGENT_NAMES

__all__ = [
    "Agent",
    "action_to_reach_price",
    "FixedPriceAgent",
    "CostPlusAgent",
    "CompetitorMatchingAgent",
    "RandomAgent",
    "OracleAgent",
    "GradientBoostingAgent",
    "LLMAgent",
    "BackwardInductionOracle",
    "build_agent",
    "AGENT_NAMES",
]
