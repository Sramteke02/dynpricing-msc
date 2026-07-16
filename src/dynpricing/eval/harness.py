"""The shared evaluation harness (E5).

A single runner drives *any* agent through identical scenarios and seeds and
records the outcomes in one place. Because every agent speaks the same Layer-3
interface, one loop drives all of them, which is what makes the comparison fair.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from dynpricing.env.config import EnvConfig
from dynpricing.env.market_env import MarketEnv
from dynpricing.agents.base import Agent
from dynpricing.agents.registry import build_agent
from dynpricing.eval.metrics import EpisodeMetrics, pricing_stability


@dataclass
class Scenario:
    """A named environment configuration (a market condition to test under)."""

    name: str
    config: EnvConfig

    def make_env(self) -> MarketEnv:
        return MarketEnv(self.config)


def default_scenarios(base: EnvConfig) -> list[Scenario]:
    """A small, sensible set of market conditions derived from the base config."""
    from copy import deepcopy

    def variant(**overrides) -> EnvConfig:
        d = base.to_dict()
        d.update(overrides)
        return EnvConfig.from_dict(d)

    return [
        Scenario("baseline", base),
        Scenario("high_competition", variant(
            d=base.d * 1.6,
            competitor_drift=base.competitor_drift * 1.5,
        )),
        Scenario("scarce_inventory", variant(
            init_inventory=max(1000, int(base.init_inventory * 0.3)),
        )),
        Scenario("strong_seasonality", variant(
            seasonal_amplitude=min(0.6, base.seasonal_amplitude * 2.0),
            holiday_days=tuple(range(40, 46)),
        )),
    ]


def run_episode(agent: Agent, env: MarketEnv, seed: int,
                scenario: str = "baseline") -> EpisodeMetrics:
    """Drive one agent through one episode and collect metrics."""
    _, info = env.reset(seed=seed)
    state = info["state"]
    agent.reset(state)

    revenue = gross_profit = 0.0
    shares: list[float] = []
    prices: list[float] = [state.own_price]
    steps = 0
    done = False
    while not done:
        action = agent.act(state)
        _, reward, terminated, truncated, info = env.step(action)
        next_state = info["state"]
        agent.observe(state, action, reward, next_state, info)

        revenue += info["revenue"]
        gross_profit += info["gross_profit"]
        shares.append(info["market_share"])
        prices.append(info["price"])
        steps += 1
        state = next_state
        done = terminated or truncated

    return EpisodeMetrics(
        agent=agent.name,
        seed=seed,
        scenario=scenario,
        revenue=revenue,
        gross_profit=gross_profit,
        market_share=float(np.mean(shares)) if shares else 0.0,
        pricing_stability=pricing_stability(prices),
        n_steps=steps,
        prices=prices,
    )


def evaluate_agent(
    agent_name: str,
    cfg: EnvConfig,
    scenario: Scenario,
    seeds,
    *,
    agent_kwargs: dict | None = None,
) -> list[EpisodeMetrics]:
    """Build and evaluate one agent over many seeds in one scenario.

    Training agents (e.g. the GBM agent) are trained once, on the *baseline*
    config, before evaluation — they never see the true demand function.
    """
    agent_kwargs = agent_kwargs or {}
    seeds = list(seeds)
    results: list[EpisodeMetrics] = []

    for seed in seeds:
        agent = build_agent(agent_name, scenario.config, seed=seed, **agent_kwargs)
        if getattr(agent, "requires_training", False):
            agent.train(make_env=lambda: MarketEnv(cfg), seed=seed)
        env = scenario.make_env()
        results.append(run_episode(agent, env, seed=seed, scenario=scenario.name))
    return results


def run_experiment(
    cfg: EnvConfig,
    agent_names,
    seeds,
    scenarios: list[Scenario] | None = None,
    *,
    agent_kwargs: dict | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[EpisodeMetrics]:
    """Run the full grid of agents x scenarios x seeds."""
    scenarios = scenarios or [Scenario("baseline", cfg)]
    out: list[EpisodeMetrics] = []
    for scenario in scenarios:
        for agent_name in agent_names:
            if progress:
                progress(f"  {scenario.name:>20} | {agent_name}")
            out.extend(
                evaluate_agent(
                    agent_name, cfg, scenario, seeds, agent_kwargs=agent_kwargs
                )
            )
    return out
