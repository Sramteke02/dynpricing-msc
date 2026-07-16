"""Environment configuration.

An :class:`EnvConfig` fully parameterises the simulation. These parameters are
produced offline by the calibration layer (see ``dynpricing.calibration``) and
then frozen for every agent, so the comparison is fair and reproducible.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class EnvConfig:
    """Parameters of the competitive e-commerce market.

    Attributes are grouped by role. Defaults are documented, economically
    plausible values used as the synthetic fallback when no calibration data is
    available.
    """

    # --- Pricing structure -------------------------------------------------
    ref_price: float = 10.0          # reference/anchor price (currency units)
    unit_cost: float = 4.0           # marginal cost per unit
    price_min: float = 5.0           # lowest allowed price
    price_max: float = 20.0          # highest allowed price
    init_price: float = 10.0         # opening price at reset

    # --- Demand model (linear, differentiated Bertrand) --------------------
    base_demand: float = 120.0       # q_ref: expected units/day at ref price, neutral season
    a0: float = 288.0                # demand intercept (units at p=0, cbar=0, neutral season)
    b: float = 24.0                  # own-price sensitivity: dq/dp = -b
    d: float = 7.2                   # cross-price sensitivity: dq/dcbar = +d
    noise_cv: float = 0.10           # coefficient of variation of multiplicative demand noise

    # --- Competition -------------------------------------------------------
    n_competitors: int = 2
    competitor_init: tuple = (10.0, 10.5)
    competitor_drift: float = 0.02   # std of competitor random-walk (relative)
    competitor_reversion: float = 0.10  # pull back toward ref_price each step

    # --- Seasonality / calendar -------------------------------------------
    horizon: int = 90               # selling horizon (days) per episode
    seasonal_amplitude: float = 0.25  # annual sinusoid amplitude on demand
    weekend_uplift: float = 0.15      # extra demand on weekends
    holiday_uplift: float = 0.40      # extra demand on modelled holidays
    holiday_days: tuple = ()          # day-of-episode indices treated as holidays
    start_day_of_year: int = 0        # phase of the annual seasonal cycle

    # --- Inventory ---------------------------------------------------------
    init_inventory: int = 12000
    allow_stockout_termination: bool = True

    # --- MDP ---------------------------------------------------------------
    discount: float = 0.99

    # --- Provenance --------------------------------------------------------
    calibration_notes: str = "synthetic defaults"
    calibration_sources: tuple = ()

    def __post_init__(self) -> None:
        if self.price_min <= 0 or self.price_max <= self.price_min:
            raise ValueError("require 0 < price_min < price_max")
        if not (self.price_min <= self.init_price <= self.price_max):
            raise ValueError("init_price must lie within [price_min, price_max]")
        if self.b <= 0:
            raise ValueError("own-price sensitivity b must be positive")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")

    # --- Serialisation -----------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=list))

    @classmethod
    def from_dict(cls, data: dict) -> "EnvConfig":
        # tuples survive JSON as lists; coerce the fields we declared as tuples
        tuple_fields = {
            "competitor_init",
            "holiday_days",
            "calibration_sources",
        }
        clean = {}
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        for k, v in data.items():
            if k not in known:
                continue
            clean[k] = tuple(v) if k in tuple_fields and isinstance(v, list) else v
        return cls(**clean)

    @classmethod
    def load(cls, path: str | Path) -> "EnvConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))
