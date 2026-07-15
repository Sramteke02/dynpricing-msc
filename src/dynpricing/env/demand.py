"""The demand model: the economic heart of the simulation.

A price-sensitive function grounded in constant-elasticity demand, modulated by
competitor prices and seasonal/calendar effects and perturbed by multiplicative
noise. The model is fully transparent and owned by the environment; it is
**never** shared with the learning agents. Only the oracle may use it directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from dynpricing.env.config import EnvConfig


@dataclass
class DemandModel:
    """Computes expected and realised units sold given market conditions."""

    cfg: EnvConfig

    # -- calendar effects ---------------------------------------------------
    def seasonal_factor(self, day: int) -> float:
        """Annual seasonal multiplier (sinusoidal), centred on 1.0."""
        phase = (self.cfg.start_day_of_year + day) / 365.0 * 2.0 * math.pi
        return 1.0 + self.cfg.seasonal_amplitude * math.sin(phase)

    def calendar_factor(self, day: int) -> float:
        """Combined weekday/weekend and holiday multiplier, centred on 1.0."""
        dow = day % 7
        factor = 1.0
        if dow >= 5:  # Saturday/Sunday
            factor += self.cfg.weekend_uplift
        if day in set(self.cfg.holiday_days):
            factor += self.cfg.holiday_uplift
        return factor

    def competitor_factor(self, price: float, competitor_prices) -> float:
        """Demand multiplier from relative price position vs competitors.

        Cheaper than the field => >1 (gain share); pricier => <1.
        """
        comp = np.asarray(competitor_prices, dtype=float)
        if comp.size == 0:
            return 1.0
        ratio = float(np.mean(comp)) / max(price, 1e-9)
        # constant-cross-elasticity style response, clipped to stay sane
        return float(np.clip(ratio ** self.cfg.cross_elasticity, 0.25, 4.0))

    # -- core demand --------------------------------------------------------
    def expected_units(
        self,
        price: float,
        competitor_prices,
        day: int,
    ) -> float:
        """Expected (noise-free) units sold for the period."""
        price = float(np.clip(price, self.cfg.price_min, self.cfg.price_max))
        own = (price / self.cfg.ref_price) ** (-self.cfg.elasticity)
        units = (
            self.cfg.base_demand
            * own
            * self.competitor_factor(price, competitor_prices)
            * self.seasonal_factor(day)
            * self.calendar_factor(day)
        )
        return float(max(units, 0.0))

    def sample_units(
        self,
        price: float,
        competitor_prices,
        day: int,
        rng: np.random.Generator,
    ) -> float:
        """Realised units sold, with multiplicative log-normal noise."""
        mean = self.expected_units(price, competitor_prices, day)
        if mean <= 0 or self.cfg.noise_cv <= 0:
            return mean
        # log-normal multiplier with unit mean and CV = noise_cv
        sigma = math.sqrt(math.log(1.0 + self.cfg.noise_cv ** 2))
        mu = -0.5 * sigma ** 2
        noise = float(rng.lognormal(mean=mu, sigma=sigma))
        return float(max(mean * noise, 0.0))

    # -- helpers for the oracle / optimum ----------------------------------
    def expected_profit(
        self,
        price: float,
        competitor_prices,
        day: int,
        unit_cost: float | None = None,
    ) -> float:
        cost = self.cfg.unit_cost if unit_cost is None else unit_cost
        units = self.expected_units(price, competitor_prices, day)
        return (price - cost) * units

    def optimal_price(
        self,
        competitor_prices,
        day: int,
        grid: int = 200,
    ) -> float:
        """Profit-maximising price on a dense grid (the theoretical optimum)."""
        prices = np.linspace(self.cfg.price_min, self.cfg.price_max, grid)
        profits = [self.expected_profit(p, competitor_prices, day) for p in prices]
        return float(prices[int(np.argmax(profits))])
