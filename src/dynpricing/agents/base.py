"""The shared agent interface (Layer 3).

Every agent implements the same contract::

    action_index = agent.act(state)

where ``state`` is a :class:`~dynpricing.env.market_env.MarketState` and the
returned value is a discrete action index into
:data:`~dynpricing.env.market_env.ACTIONS`. This uniform contract is what lets a
single runner drive every agent and makes the comparison fair.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from dynpricing.env.market_env import ACTIONS, MarketState


def action_to_reach_price(state: MarketState, target_price: float) -> int:
    """Return the action whose resulting price is closest to ``target_price``.

    Many agents reason in terms of a *desired price*; the environment only
    accepts discrete moves. This helper bridges the two by picking the move that
    lands nearest the target (after clipping to the legal price band).
    """
    target = float(np.clip(target_price, state.price_min, state.price_max))
    best_idx, best_err = 0, float("inf")
    for idx, (_, mult) in enumerate(ACTIONS):
        reached = float(np.clip(state.own_price * mult, state.price_min, state.price_max))
        err = abs(reached - target)
        if err < best_err:
            best_idx, best_err = idx, err
    return best_idx


class Agent(ABC):
    """Base class for all pricing agents."""

    name: str = "agent"
    #: whether the runner should call :meth:`learn` before evaluation
    requires_training: bool = False

    def reset(self, state: MarketState) -> None:
        """Hook called at the start of each episode. Default: no-op."""

    @abstractmethod
    def act(self, state: MarketState) -> int:
        """Choose a discrete action index given the market state."""

    def observe(self, state: MarketState, action: int, reward: float,
                next_state: MarketState, info: dict) -> None:
        """Optional learning-from-interaction hook. Default: no-op."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(name={self.name!r})"
