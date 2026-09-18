"""Unit tests for the simulation core (E1)."""

import numpy as np
import pytest

from dynpricing.env.config import EnvConfig
from dynpricing.env.demand import DemandModel
from dynpricing.env.market_env import MarketEnv, ACTIONS


def _intercept(cfg, dm, day, cbar):
    """a(t) = a0*S(t)*C(t) + d*cbar, the linear-demand intercept."""
    return cfg.a0 * dm.seasonal_factor(day) * dm.calendar_factor(day) + cfg.d * cbar


def _numeric_pstar(cfg, dm, comps, day, grid=40001):
    """Profit-maximising price by dense-grid search over expected profit."""
    ps = np.linspace(cfg.price_min, cfg.price_max, grid)
    profits = np.array([dm.expected_profit(p, comps, day) for p in ps])
    return float(ps[int(np.argmax(profits))]), float(ps[1] - ps[0])


def _seasonal_days(dm):
    """Days of the annual seasonal peak, trough, and a mid (S~=1) point."""
    S = np.array([dm.seasonal_factor(t) for t in range(365)])
    return int(np.argmax(S)), int(np.argmin(S)), int(np.argmin(np.abs(S - 1.0)))


def test_demand_falls_as_price_rises():
    cfg = EnvConfig()
    dm = DemandModel(cfg)
    lo = dm.expected_units(cfg.price_min, cfg.competitor_init, day=0)
    hi = dm.expected_units(cfg.price_max, cfg.competitor_init, day=0)
    assert lo > hi >= 0


def test_demand_magnitude_plausible():
    cfg = EnvConfig()
    dm = DemandModel(cfg)
    at_ref = dm.expected_units(cfg.ref_price, cfg.competitor_init, day=0)
    assert 0.3 * cfg.base_demand < at_ref < 3 * cfg.base_demand


def test_demand_strictly_decreasing_across_band():
    """(1) q(p) strictly decreasing in p wherever demand is positive, and
    never increasing anywhere in the band."""
    cfg = EnvConfig(weekend_uplift=0.0)
    dm = DemandModel(cfg)
    comps = [cfg.ref_price]
    for day in _seasonal_days(dm):
        ps = np.linspace(cfg.price_min, cfg.price_max, 400)
        q = np.array([dm.expected_units(p, comps, day) for p in ps])
        diffs = np.diff(q)
        assert np.all(diffs <= 1e-9)
        positive = q[:-1] > 1e-9
        assert positive.any()
        assert np.all(diffs[positive] < 0)


def test_demand_zero_at_choke_and_never_negative():
    """(2) q == 0 exactly at the choke price a(t)/b, and never negative
    anywhere in the band (the max(0, .) clamp)."""
    cfg = EnvConfig(weekend_uplift=0.0)
    dm = DemandModel(cfg)
    comps = [cfg.ref_price]
    cbar = float(np.mean(comps))
    for day in _seasonal_days(dm):
        a = _intercept(cfg, dm, day, cbar)
        choke = a / cfg.b
        assert cfg.price_min <= choke <= cfg.price_max
        assert dm.expected_units(choke, comps, day) == pytest.approx(0.0, abs=1e-9)
        assert dm.expected_units(choke * 0.99, comps, day) > 0.0
        for p in np.linspace(cfg.price_min, cfg.price_max, 200):
            assert dm.expected_units(p, comps, day) >= 0.0


def test_numeric_argmax_matches_closed_form_pstar():
    """(3) dense-grid argmax of expected profit == closed form
    p*(t) = (a(t)/b + c)/2 at seasonal peak, trough, and mid."""
    cfg = EnvConfig(weekend_uplift=0.0)
    dm = DemandModel(cfg)
    comps = [cfg.ref_price]
    cbar = float(np.mean(comps))
    for day in _seasonal_days(dm):
        p_numeric, step = _numeric_pstar(cfg, dm, comps, day)
        a = _intercept(cfg, dm, day, cbar)
        p_closed = (a / cfg.b + cfg.unit_cost) / 2.0
        assert cfg.price_min < p_closed < cfg.price_max
        assert abs(p_numeric - p_closed) <= 2 * step


def test_pstar_differs_between_seasonal_peak_and_trough():
    """(4) REGRESSION: p*(peak) != p*(trough). This is the test that would
    have caught the original isoelastic defect (constant p*), and it MUST
    fail against the old q = K(t)*p^-eta demand."""
    cfg = EnvConfig(weekend_uplift=0.0)
    dm = DemandModel(cfg)
    comps = [cfg.ref_price]
    peak, trough, _ = _seasonal_days(dm)
    p_peak, _ = _numeric_pstar(cfg, dm, comps, peak)
    p_trough, _ = _numeric_pstar(cfg, dm, comps, trough)
    assert abs(p_peak - p_trough) > 0.1


def test_interior_condition_a_gt_bc_over_horizon():
    """(5) interior condition a(t) > b*c holds for every day in the horizon."""
    cfg = EnvConfig()
    dm = DemandModel(cfg)
    cbar = float(np.mean(cfg.competitor_init))
    for t in range(cfg.horizon):
        assert _intercept(cfg, dm, t, cbar) > cfg.b * cfg.unit_cost


def test_reward_is_margin_times_units():
    cfg = EnvConfig(noise_cv=0.0)
    env = MarketEnv(cfg)
    env.reset(seed=0)
    _, reward, _, _, info = env.step(0)
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
        _, _, terminated, truncated, info = env.step(2)
        assert info["state"].inventory <= inv_prev
        inv_prev = info["state"].inventory
        if terminated or truncated:
            break
    assert terminated


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
    assert p > cfg.unit_cost
