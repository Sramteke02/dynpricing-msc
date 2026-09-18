"""Tests for statistical rigour (Task 3): bootstrap CIs and paired comparisons."""

import numpy as np

from dynpricing.eval.stats import (
    bootstrap_mean_ci,
    summarize,
    paired_comparison,
    resolve_agent_name,
)


def test_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(0)
    vals = rng.normal(100, 10, size=200)
    mean, lo, hi, sem = bootstrap_mean_ci(vals, seed=0)
    assert lo < mean < hi
    assert hi - lo < 10
    assert sem > 0


def test_bootstrap_ci_reproducible():
    vals = [1, 2, 3, 4, 5, 6, 7, 8]
    a = bootstrap_mean_ci(vals, seed=42)
    b = bootstrap_mean_ci(vals, seed=42)
    assert a == b


def _rows(agent, scenario, values):
    return [{"agent": agent, "scenario": scenario, "seed": i,
             "revenue": v, "gross_profit": v,
             "market_share": 0.3, "pricing_stability": 0.1}
            for i, v in enumerate(values)]


def test_summarize_has_ci_fields():
    rows = _rows("a", "baseline", [10, 12, 11, 13, 9])
    summ = summarize(rows)
    assert len(summ) == 1
    r = summ[0]
    for key in ("gross_profit_mean", "gross_profit_ci_low", "gross_profit_ci_high"):
        assert key in r
    assert r["n_seeds"] == 5
    assert r["gross_profit_ci_low"] <= r["gross_profit_mean"] <= r["gross_profit_ci_high"]


def test_paired_comparison_detects_clear_difference():
    rows = _rows("a", "baseline", [100, 110, 90, 105, 95, 100, 102, 98])
    rows += _rows("b", "baseline", [150, 160, 140, 155, 145, 150, 152, 148])
    res = paired_comparison(rows, "a", "b", "gross_profit", "baseline")
    assert res is not None
    assert res.significant
    assert res.winner == "b"
    assert res.ci_high < 0


def test_paired_comparison_ns_when_identical_noise():
    rng = np.random.default_rng(1)
    base = rng.normal(100, 5, size=12)
    rows = _rows("a", "baseline", base)
    rows += _rows("b", "baseline", base + rng.normal(0, 5, size=12))
    res = paired_comparison(rows, "a", "b", "gross_profit", "baseline")
    assert res is not None
    assert not res.significant


def test_resolve_agent_name_prefix():
    rows = _rows("llm:gpt-4o-mini-2024-07-18", "baseline", [1, 2])
    assert resolve_agent_name(rows, "llm") == "llm:gpt-4o-mini-2024-07-18"
    assert resolve_agent_name(rows, "nope") is None
