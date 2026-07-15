"""Statistical summaries for the evaluation (Task 3: rigour).

Two ideas make the results chapter defensible:

1. **Confidence intervals, not point numbers.** Every per-agent metric is
   reported as a mean with a bootstrap 95% confidence interval over seeds.
2. **Paired comparisons.** All agents are evaluated on the *same seed set*, so
   agent differences are tested *paired by seed* (each seed is a matched block).
   The GBM-vs-LLM gap is reported as a paired mean difference with a CI; if the
   CI excludes zero the difference is significant at the 5% level.

The bootstrap is dependency-free (no SciPy) and uses a fixed RNG seed so the
intervals are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

METRICS = ("revenue", "gross_profit", "market_share", "pricing_stability")


def bootstrap_mean_ci(
    values, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0
):
    """Bootstrap mean and (1-alpha) percentile CI for a 1-D sample."""
    v = np.asarray(values, dtype=float)
    n = v.size
    mean = float(np.mean(v)) if n else 0.0
    if n < 2:
        return mean, mean, mean, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = v[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    sem = float(np.std(v, ddof=1) / np.sqrt(n))
    return mean, lo, hi, sem


def _index(rows):
    """Group rows -> {(agent, scenario): {seed: row}}."""
    out: dict = {}
    for r in rows:
        out.setdefault((r["agent"], r["scenario"]), {})[r["seed"]] = r
    return out


def summarize(rows, metrics=METRICS, n_boot: int = 10000, seed: int = 0) -> list[dict]:
    """Per (agent, scenario) mean + bootstrap 95% CI for each metric."""
    grouped = _index(rows)
    out = []
    for (agent, scenario), by_seed in grouped.items():
        entry = {"agent": agent, "scenario": scenario, "n_seeds": len(by_seed)}
        for m in metrics:
            vals = [r[m] for r in by_seed.values()]
            mean, lo, hi, sem = bootstrap_mean_ci(vals, n_boot=n_boot, seed=seed)
            entry[f"{m}_mean"] = mean
            entry[f"{m}_ci_low"] = lo
            entry[f"{m}_ci_high"] = hi
            entry[f"{m}_sem"] = sem
        out.append(entry)
    return out


@dataclass
class PairedResult:
    agent_a: str
    agent_b: str
    scenario: str
    metric: str
    n_pairs: int
    mean_a: float
    mean_b: float
    mean_diff: float          # a - b
    ci_low: float
    ci_high: float
    significant: bool         # 95% CI excludes 0
    winner: str

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def paired_comparison(
    rows,
    agent_a: str,
    agent_b: str,
    metric: str = "gross_profit",
    scenario: str = "baseline",
    n_boot: int = 10000,
    seed: int = 0,
) -> PairedResult | None:
    """Paired (by seed) bootstrap comparison of two agents on one metric.

    Returns ``None`` if the agents do not share at least two common seeds.
    """
    grouped = _index(rows)
    a = grouped.get((agent_a, scenario), {})
    b = grouped.get((agent_b, scenario), {})
    common = sorted(set(a) & set(b))
    if len(common) < 2:
        return None

    da = np.array([a[s][metric] for s in common], dtype=float)
    db = np.array([b[s][metric] for s in common], dtype=float)
    diff = da - db  # matched per seed

    mean_diff, lo, hi, _ = bootstrap_mean_ci(diff, n_boot=n_boot, seed=seed)
    significant = (lo > 0) or (hi < 0)
    winner = (agent_a if mean_diff > 0 else agent_b) if significant else "tie"
    return PairedResult(
        agent_a=agent_a, agent_b=agent_b, scenario=scenario, metric=metric,
        n_pairs=len(common), mean_a=float(da.mean()), mean_b=float(db.mean()),
        mean_diff=mean_diff, ci_low=lo, ci_high=hi,
        significant=significant, winner=winner,
    )


def resolve_agent_name(rows, prefix: str, scenario: str | None = None) -> str | None:
    """Find the full agent name matching ``prefix`` (e.g. 'llm' -> 'llm:gpt-...')."""
    names = {r["agent"] for r in rows
             if scenario is None or r["scenario"] == scenario}
    if prefix in names:
        return prefix
    matches = sorted(n for n in names if n.split(":")[0] == prefix)
    return matches[0] if matches else None
