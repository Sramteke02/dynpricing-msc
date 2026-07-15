"""Tests for the exact backward-induction oracle (optional rigour)."""

import numpy as np
import pytest

from dynpricing.env.config import EnvConfig
from dynpricing.env.demand import DemandModel
from dynpricing.env.market_env import MarketEnv
from dynpricing.agents.oracle_dp import BackwardInductionOracle
from dynpricing.agents.baselines import OracleAgent, FixedPriceAgent
from dynpricing.eval.harness import run_episode


@pytest.fixture
def cfg():
    # small problem so the DP solves quickly in tests
    return EnvConfig(horizon=30, init_inventory=4000)


def test_dp_solves_and_has_finite_value(cfg):
    dp = BackwardInductionOracle(cfg, inv_buckets=120)
    assert np.isfinite(dp.optimal_value)
    assert dp.optimal_value > 0
    assert dp.n_prices > 1


def test_dp_oracle_not_beaten_by_fixed(cfg):
    dp = BackwardInductionOracle(cfg, inv_buckets=150)
    seeds = range(8)
    dp_p = np.mean([run_episode(dp, MarketEnv(cfg), seed=s).gross_profit for s in seeds])
    fx_p = np.mean([run_episode(FixedPriceAgent(), MarketEnv(cfg), seed=s).gross_profit
                    for s in seeds])
    assert dp_p >= fx_p - 1e-6


def test_fluid_and_dp_oracles_agree(cfg):
    """The fluid heuristic should match the exact ceiling within a few percent."""
    dp = BackwardInductionOracle(cfg, inv_buckets=200)
    seeds = range(12)
    dp_p = np.mean([run_episode(dp, MarketEnv(cfg), seed=s).gross_profit for s in seeds])
    fl_p = np.mean([run_episode(OracleAgent(DemandModel(cfg)), MarketEnv(cfg), seed=s).gross_profit
                    for s in seeds])
    rel_gap = abs(fl_p - dp_p) / dp_p
    assert rel_gap < 0.05  # agree within 5%
