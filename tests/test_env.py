"""Unit tests for the simulation core (E1)."""

import numpy as np
import pytest

from dynpricing.env.config import EnvConfig
from dynpricing.env.demand import DemandModel
from dynpricing.env.market_env import MarketEnv, ACTIONS


def test_demand_falls_as_price_rises():
    cfg = EnvConfig()
    dm = DemandModel(cfg)
    lo = dm.expected_units(cfg.price_min, cfg.competitor_init, day=0)
    hi = dm.expected_units(cfg.price_max, cfg.competitor_init, day=0)
    assert lo > hi > 0  # downward-sloping demand


def test_demand_magnitude_plausible():
    cfg = EnvConfig()
    dm = DemandModel(cfg)
    at_ref = dm.expected_units(cfg.ref_price, cfg.competitor_init, day=0)
    # at the reference price, demand should be on the order of base_demand
    assert 0.3 * cfg.base_demand < at_ref < 3 * cfg.base_demand


def test_reward_is_margin_times_units():
    cfg = EnvConfig(noise_cv=0.0)  # deterministic
    env = MarketEnv(cfg)
    env.reset(seed=0)
    _, reward, _, _, info = env.step(0)  # hold
    expected = (info["price"] - cfg.unit_cost) * info["units"]
    assert reward == pytest.approx(expected)
    assert info["revenue"] == pytest.approx(info["price"] * info["units"])


def test_inventory_depletes_and_terminates():
    cfg = EnvConfig(init_inventory=200, horizon=1000, noise_cv=0.0)
    env = MarketEnv(cfg)
    env.reset(seed=1)
    inv_prev = cfg.init_inventory
    terminated = False
    for _ in range(1000):
        _, _, terminated, truncated, info = env.step(2)  # lower price -> more sales
        assert info["state"].inventory <= inv_prev
        inv_prev = info["state"].inventory
        if terminated or truncated:
            break
    assert terminated  # should stock out before the horizon


def test_obs_shape_and_state_wellformed():
    cfg = EnvConfig()
    env = MarketEnv(cfg)
    obs, info = env.reset(seed=2)
    assert obs.shape == env.observation_space.shape
    state = info["state"]
    assert state.inventory == cfg.init_inventory
    assert len(state.competitor_prices) == cfg.n_competitors
    assert state.price_min <= state.own_price <= state.price_max


def test_action_space_and_bounds():
    cfg = EnvConfig()
    env = MarketEnv(cfg)
    env.reset(seed=0)
    for a in range(len(ACTIONS)):
        _, _, _, _, info = env.step(a)
        assert cfg.price_min <= info["price"] <= cfg.price_max
    with pytest.raises(ValueError):
        env.step(999)


def test_reset_is_reproducible():
    cfg = EnvConfig()
    e1, e2 = MarketEnv(cfg), MarketEnv(cfg)
    e1.reset(seed=7)
    e2.reset(seed=7)
    for a in [0, 1, 2, 3, 0, 1]:
        _, r1, _, _, _ = e1.step(a)
        _, r2, _, _, _ = e2.step(a)
        assert r1 == pytest.approx(r2)


def test_optimal_price_within_band():
    cfg = EnvConfig()
    dm = DemandModel(cfg)
    p = dm.optimal_price(cfg.competitor_init, day=0)
    assert cfg.price_min <= p <= cfg.price_max
    # optimum should be above marginal cost
    assert p > cfg.unit_cost
