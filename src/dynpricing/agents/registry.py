"""Factory for building agents by name (used by the CLI and harness)."""

from __future__ import annotations

from dynpricing.env.config import EnvConfig
from dynpricing.env.demand import DemandModel
from dynpricing.agents.base import Agent
from dynpricing.agents.baselines import (
    FixedPriceAgent,
    CostPlusAgent,
    CompetitorMatchingAgent,
    RandomAgent,
    OracleAgent,
)
from dynpricing.agents.gbm_agent import GradientBoostingAgent
from dynpricing.agents.gbm_uniform_agent import UniformExplorationGBMAgent
from dynpricing.agents.llm_agent import LLMAgent
from dynpricing.agents.oracle_dp import BackwardInductionOracle

AGENT_NAMES = (
    "fixed",
    "cost_plus",
    "competitor_match",
    "random",
    "gbm",
    "gbm_uniform",
    "llm",
    "oracle",
)


def build_agent(name: str, cfg: EnvConfig, *, seed: int = 0, **kwargs) -> Agent:
    """Construct an agent by name, wiring up any dependencies it needs."""
    name = name.lower()
    if name == "fixed":
        return FixedPriceAgent()
    if name == "cost_plus":
        return CostPlusAgent(markup=kwargs.get("markup", 1.6))
    if name == "competitor_match":
        return CompetitorMatchingAgent(undercut=kwargs.get("undercut", 0.0))
    if name == "random":
        return RandomAgent(seed=seed)
    if name == "oracle":
        return OracleAgent(demand=DemandModel(cfg))
    if name == "oracle_dp":
        return BackwardInductionOracle(
            cfg, inv_buckets=kwargs.get("inv_buckets", 400)
        )
    if name == "gbm":
        return GradientBoostingAgent(
            exploration_episodes=kwargs.get("exploration_episodes", 40), seed=seed
        )
    if name == "gbm_uniform":
        return UniformExplorationGBMAgent(
            exploration_episodes=kwargs.get("exploration_episodes", 40),
            seed=seed,
            uniform_exploration=kwargs.get("uniform_exploration", True),
            clamp_to_train_max=kwargs.get("clamp_to_train_max", True),
            verbose=kwargs.get("verbose", False),
        )
    if name == "llm":
        llm_kwargs = {}
        if "llm_model" in kwargs:
            llm_kwargs["model"] = kwargs["llm_model"]
        if "llm_provider" in kwargs:
            llm_kwargs["provider"] = kwargs["llm_provider"]
        if "llm_temperature" in kwargs:
            llm_kwargs["temperature"] = kwargs["llm_temperature"]
        if "llm_template_name" in kwargs:
            name_ = kwargs["llm_template_name"]
            from dynpricing.agents.llm_agent import TEMPLATES
            llm_kwargs["template_name"] = name_
            llm_kwargs["user_template"] = TEMPLATES.get(name_)
        return LLMAgent(
            mode=kwargs.get("llm_mode", "api"),
            force_fallback=kwargs.get("force_fallback", False),
            **llm_kwargs,
        )
    raise ValueError(f"unknown agent {name!r}; choose from {AGENT_NAMES}")
