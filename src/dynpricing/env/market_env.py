"""The Gymnasium market environment (Layer 4).

Pricing is modelled as a Markov Decision Process:

* **State**   own price, competitor prices, current demand level, remaining
              inventory, day of week, season.
* **Action**  a small discrete set of price moves: hold, +5%, -5%, -10%.
* **Reward**  gross profit = (price - unit cost) x units sold.
* **Transition**  the chosen price drives the demand model; inventory is
              decremented and time advances.
* **Episode** one selling horizon, ending at the horizon or at stockout.

The environment exposes a rich :class:`MarketState` (the "shared interface" of
Layer 3) to agents via the ``info`` dict, and a flat Box observation for
Gymnasium/RL compatibility.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces

    _GYM_BASE = gym.Env
except Exception:  # pragma: no cover - exercised only without gymnasium
    gym = None
    spaces = None
    _GYM_BASE = object

from dynpricing.env.config import EnvConfig
from dynpricing.env.demand import DemandModel

ACTIONS: tuple[tuple[str, float], ...] = (
    ("hold", 1.00),
    ("raise_5", 1.05),
    ("lower_5", 0.95),
    ("lower_10", 0.90),
)


@dataclass
class MarketState:
    """Interpretable market state handed to agents (Layer 3 interface).

    Agents receive this object and return a discrete action index. It contains
    everything a seller could reasonably observe; it deliberately does **not**
    contain the true demand parameters (those are owned by the environment).
    """

    own_price: float
    competitor_prices: tuple
    unit_cost: float
    demand_level: float
    inventory: int
    day_of_week: int
    day: int
    season: int
    horizon: int
    price_min: float
    price_max: float
    ref_price: float

    @property
    def competitor_mean(self) -> float:
        c = self.competitor_prices
        return float(sum(c) / len(c)) if c else self.own_price


class MarketEnv(_GYM_BASE):
    """A competitive e-commerce pricing environment."""

    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None):
        super().__init__()
        self.cfg = config or EnvConfig()
        self.demand = DemandModel(self.cfg)

        if spaces is not None:
            self.action_space = spaces.Discrete(len(ACTIONS))
            low = np.array(
                [0.0] * (2 + self.cfg.n_competitors) + [0.0, 0.0, 0.0],
                dtype=np.float32,
            )
            high = np.array(
                [np.inf] * (2 + self.cfg.n_competitors) + [np.inf, 6.0, 3.0],
                dtype=np.float32,
            )
            self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self._rng = np.random.default_rng()
        self._reset_state()

    def _reset_state(self) -> None:
        self.price = self.cfg.init_price
        self.competitor_prices = list(self.cfg.competitor_init[: self.cfg.n_competitors])
        while len(self.competitor_prices) < self.cfg.n_competitors:
            self.competitor_prices.append(self.cfg.ref_price)
        self.inventory = int(self.cfg.init_inventory)
        self.day = 0
        self.last_units = 0.0

    def _season(self, day: int) -> int:
        phase = ((self.cfg.start_day_of_year + day) % 365) / 365.0
        return int(min(3, phase * 4))

    def _make_state(self) -> MarketState:
        return MarketState(
            own_price=float(self.price),
            competitor_prices=tuple(round(c, 4) for c in self.competitor_prices),
            unit_cost=float(self.cfg.unit_cost),
            demand_level=float(self.last_units),
            inventory=int(self.inventory),
            day_of_week=self.day % 7,
            day=int(self.day),
            season=self._season(self.day),
            horizon=int(self.cfg.horizon),
            price_min=float(self.cfg.price_min),
            price_max=float(self.cfg.price_max),
            ref_price=float(self.cfg.ref_price),
        )

    def _make_obs(self) -> np.ndarray:
        comps = list(self.competitor_prices)
        comps += [self.cfg.ref_price] * (self.cfg.n_competitors - len(comps))
        vec = [self.price, *comps, self.last_units, self.inventory,
               float(self.day % 7), float(self._season(self.day))]
        return np.asarray(vec, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            try:
                super().reset(seed=seed)
            except TypeError:
                pass
        self._reset_state()
        obs = self._make_obs()
        info = {"state": self._make_state()}
        return obs, info

    def step(self, action: int):
        if not 0 <= int(action) < len(ACTIONS):
            raise ValueError(f"invalid action {action!r}; expected 0..{len(ACTIONS) - 1}")

        _, mult = ACTIONS[int(action)]
        self.price = float(
            np.clip(self.price * mult, self.cfg.price_min, self.cfg.price_max)
        )

        demanded = self.demand.sample_units(
            self.price, self.competitor_prices, self.day, self._rng
        )
        units = float(min(demanded, self.inventory))

        reward = (self.price - self.cfg.unit_cost) * units

        comp_units = sum(
            self.demand.expected_units(cp, self.competitor_prices, self.day)
            for cp in self.competitor_prices
        )
        total = units + comp_units
        market_share = float(units / total) if total > 0 else 0.0

        self.inventory -= int(round(units))
        self.last_units = units
        self.day += 1
        self._step_competitors()

        truncated = self.day >= self.cfg.horizon
        terminated = self.cfg.allow_stockout_termination and self.inventory <= 0
        if self.inventory < 0:
            self.inventory = 0

        obs = self._make_obs()
        info = {
            "state": self._make_state(),
            "units": units,
            "demanded": demanded,
            "price": self.price,
            "market_share": market_share,
            "revenue": self.price * units,
            "gross_profit": reward,
        }
        return obs, float(reward), bool(terminated), bool(truncated), info

    def _step_competitors(self) -> None:
        new = []
        for cp in self.competitor_prices:
            shock = self._rng.normal(0.0, self.cfg.competitor_drift) * cp
            revert = self.cfg.competitor_reversion * (self.cfg.ref_price - cp)
            cp2 = cp + shock + revert
            new.append(float(np.clip(cp2, self.cfg.price_min, self.cfg.price_max)))
        self.competitor_prices = new

    def render(self):  # pragma: no cover - no visual rendering
        return None
