"""The five rule-based baseline agents (E3).

These anchor the evaluation ladder. The simplest baselines mark the floor; the
oracle, which alone may use the true demand function, marks the ceiling (the
theoretical optimum within the discrete action set).
"""

from __future__ import annotations

import numpy as np

from dynpricing.agents.base import Agent, action_to_reach_price
from dynpricing.env.market_env import ACTIONS, MarketState
from dynpricing.env.demand import DemandModel


class FixedPriceAgent(Agent):
    """Holds the opening price forever (always chooses ``hold``)."""

    name = "fixed"

    def act(self, state: MarketState) -> int:
        return 0


class CostPlusAgent(Agent):
    """Targets a fixed markup over unit cost."""

    name = "cost_plus"

    def __init__(self, markup: float = 1.6):
        self.markup = float(markup)

    def act(self, state: MarketState) -> int:
        target = state.unit_cost * self.markup
        return action_to_reach_price(state, target)


class CompetitorMatchingAgent(Agent):
    """Tracks the mean competitor price, optionally undercutting slightly."""

    name = "competitor_match"

    def __init__(self, undercut: float = 0.0):
        self.undercut = float(undercut)

    def act(self, state: MarketState) -> int:
        target = state.competitor_mean * (1.0 - self.undercut)
        return action_to_reach_price(state, target)


class RandomAgent(Agent):
    """Chooses a uniformly random legal action (a naive lower anchor)."""

    name = "random"

    def __init__(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed)

    def reset(self, state: MarketState) -> None:
        pass

    def act(self, state: MarketState) -> int:
        return int(self._rng.integers(0, len(ACTIONS)))


class OracleAgent(Agent):
    """Inventory-aware optimum using the *true* demand model (the ceiling).

    The oracle is the only agent permitted to see the environment's true demand
    function. Because it has perfect knowledge it should never be beaten; an
    agent that beats it signals a bug in the environment or harness.

    It does **not** price greedily per period: with finite inventory over a
    finite horizon, dumping stock early at the static profit-maximising price is
    suboptimal. Instead the oracle uses the classic fluid (rate-control) result
    from revenue management:

    * compute the static profit-maximising price ``p*`` and its expected demand;
    * if that demand can be sustained over the remaining horizon without
      stocking out, target ``p*`` (inventory is not binding);
    * otherwise raise the price to the *pacing* price at which expected demand
      equals the sustainable sell-through rate ``remaining_inventory /
      remaining_days``. Selling all stock evenly at the highest such price
      maximises total profit under the constraint.

    The oracle then takes the discrete action that moves nearest this target.
    """

    name = "oracle"

    def __init__(self, demand: DemandModel, grid: int = 120):
        self.demand = demand
        self.grid = int(grid)

    def _target_price(self, state: MarketState) -> float:
        static_opt = self.demand.optimal_price(
            state.competitor_prices, state.day, grid=self.grid
        )
        remaining_days = max(1, state.horizon - state.day)
        if state.inventory <= 0:
            return static_opt
        target_rate = state.inventory / remaining_days
        units_at_static = self.demand.expected_units(
            static_opt, state.competitor_prices, state.day
        )
        if units_at_static <= target_rate:
            return static_opt

        prices = np.linspace(static_opt, state.price_max, self.grid)
        for p in prices:
            if self.demand.expected_units(p, state.competitor_prices, state.day) <= target_rate:
                return float(p)
        return float(state.price_max)

    def act(self, state: MarketState) -> int:
        target = self._target_price(state)
        best_idx, best_err = 0, float("inf")
        for idx, (_, mult) in enumerate(ACTIONS):
            price = float(np.clip(
                state.own_price * mult, state.price_min, state.price_max
            ))
            err = abs(price - target)
            if err < best_err:
                best_idx, best_err = idx, err
        return best_idx
