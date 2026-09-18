"""The demand model: the economic heart of the simulation.

A **linear (differentiated-Bertrand) demand** function: expected units fall
linearly in own price, rise linearly in the mean competitor price, and are
modulated by multiplicative seasonal/calendar factors, then clamped at zero.
Realised units add multiplicative log-normal noise. The model is fully
transparent and owned by the environment; it is **never** shared with the
learning agents. Only the oracle may use it directly.

    q(p, t) = max(0, a0 * S(t) * C(t) - b * p + d * cbar(t))

with a(t) = a0 * S(t) * C(t) + d * cbar(t) the (time-varying) demand
intercept, so q(p, t) = max(0, a(t) - b * p).

Why linear rather than isoelastic. Under constant-elasticity demand
q = K(t) * p^(-eta) the profit-maximising price p* = c * eta / (eta - 1) is a
CONSTANT, independent of season, calendar, competitors and inventory. Linear
demand instead gives p*(t) = (a(t)/b + c)/2, which moves with the intercept
a(t) — so seasonality, calendar and competitor effects genuinely shift the
optimal price over the horizon.
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

    def __post_init__(self) -> None:
        """Guarantee an interior optimum: a(t) > b*c for every day in the
        horizon (evaluated at the neutral competitor level cbar = ref_price).
        Below this threshold the profit-maximising price collapses to the
        marginal cost / lower bound and the model degenerates."""
        cfg = self.cfg
        threshold = cfg.b * cfg.unit_cost
        worst_day, worst_a = 0, math.inf
        for t in range(cfg.horizon):
            a_t = (
                cfg.a0 * self.seasonal_factor(t) * self.calendar_factor(t)
                + cfg.d * cfg.ref_price
            )
            if a_t < worst_a:
                worst_a, worst_day = a_t, t
        if not worst_a > threshold:
            raise ValueError(
                "linear demand requires an interior optimum a(t) > b*unit_cost "
                f"for all t; violated at day {worst_day}: a(t)={worst_a:.4f} "
                f"<= b*c={threshold:.4f} (a0={cfg.a0}, b={cfg.b}, d={cfg.d}, "
                f"unit_cost={cfg.unit_cost}, ref_price={cfg.ref_price})."
            )

    def seasonal_factor(self, day: int) -> float:
        """Annual seasonal multiplier (sinusoidal), centred on 1.0."""
        phase = (self.cfg.start_day_of_year + day) / 365.0 * 2.0 * math.pi
        return 1.0 + self.cfg.seasonal_amplitude * math.sin(phase)

    def calendar_factor(self, day: int) -> float:
        """Combined weekday/weekend and holiday multiplier, centred on 1.0."""
        dow = day % 7
        factor = 1.0
        if dow >= 5:
            factor += self.cfg.weekend_uplift
        if day in set(self.cfg.holiday_days):
            factor += self.cfg.holiday_uplift
        return factor

    def intercept(self, competitor_prices, day: int) -> float:
        """The time-varying linear intercept a(t) = a0*S(t)*C(t) + d*cbar(t)."""
        comp = np.asarray(competitor_prices, dtype=float)
        cbar = float(comp.mean()) if comp.size else self.cfg.ref_price
        return (
            self.cfg.a0 * self.seasonal_factor(day) * self.calendar_factor(day)
            + self.cfg.d * cbar
        )

    def expected_units(
        self,
        price: float,
        competitor_prices,
        day: int,
    ) -> float:
        """Expected (noise-free) units sold for the period.

        q(p, t) = max(0, a(t) - b * p), with a(t) the linear intercept above.
        """
        price = float(np.clip(price, self.cfg.price_min, self.cfg.price_max))
        a = self.intercept(competitor_prices, day)
        units = a - self.cfg.b * price
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
        sigma = math.sqrt(math.log(1.0 + self.cfg.noise_cv ** 2))
        mu = -0.5 * sigma ** 2
        noise = float(rng.lognormal(mean=mu, sigma=sigma))
        return float(max(mean * noise, 0.0))

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
        """Profit-maximising price on a dense grid (the theoretical optimum).

        For linear demand the closed form is p*(t) = (a(t)/b + c)/2 clamped to
        the band; the grid search agrees to grid resolution and stays correct
        if the optimum falls outside the band.
        """
        prices = np.linspace(self.cfg.price_min, self.cfg.price_max, grid)
        profits = [self.expected_profit(p, competitor_prices, day) for p in prices]
        return float(prices[int(np.argmax(profits))])
