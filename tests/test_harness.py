"""Integration tests for the evaluation harness and calibration (E2, E5)."""

import json

import pytest

from dynpricing.env.config import EnvConfig
from dynpricing.eval.harness import run_experiment, default_scenarios
from dynpricing.eval.metrics import aggregate, pricing_stability
from dynpricing.calibration import calibrate


@pytest.fixture
def cfg():
    return EnvConfig(horizon=30, init_inventory=20000)


def test_run_experiment_records_all_agents(cfg):
    agents = ["fixed", "cost_plus", "random", "oracle"]
    results = run_experiment(cfg, agents, seeds=range(3))
    assert len(results) == len(agents) * 3
    for m in results:
        assert m.n_steps > 0
        assert m.revenue >= 0
    rows = [m.to_row() for m in results]
    agg = aggregate(rows)
    assert {r["agent"] for r in agg} == set(agents)


def test_scenarios_build(cfg):
    scens = default_scenarios(cfg)
    names = {s.name for s in scens}
    assert {"baseline", "high_competition", "scarce_inventory", "strong_seasonality"} <= names


def test_pricing_stability_metric():
    assert pricing_stability([10, 10, 10]) == pytest.approx(0.0)
    assert pricing_stability([10, 11, 10, 12]) > 0
    assert pricing_stability([5]) == 0.0


def test_calibration_synthetic_fallback(tmp_path):
    result = calibrate(tmp_path)
    assert result.sources_used == []
    assert isinstance(result.config, EnvConfig)
    out = tmp_path / "cfg.json"
    result.config.save(out)
    loaded = EnvConfig.load(out)
    assert loaded.ref_price == result.config.ref_price


def test_calibration_from_synthetic_uci(tmp_path):
    import pandas as pd
    import numpy as np

    rng = np.random.default_rng(0)
    price = rng.uniform(2, 20, size=5000)
    qty = np.maximum(1, (300 * price ** -1.5 * rng.lognormal(0, 0.2, size=5000))).round()
    df = pd.DataFrame({"Price": price, "Quantity": qty})
    df.to_csv(tmp_path / "online_retail_II.csv", index=False)

    result = calibrate(tmp_path)
    assert "UCI Online Retail II" in result.sources_used
    cfg = result.config
    implied = cfg.b * cfg.ref_price / cfg.base_demand
    assert implied == pytest.approx(2.0, rel=1e-3)


def test_sanity_report_passes_on_defaults():
    from dynpricing.calibration import sanity_report

    report, ok = sanity_report(EnvConfig())
    assert ok
    assert "demand falls as price rises          : YES" in report
