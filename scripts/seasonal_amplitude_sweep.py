"""Seasonal-amplitude sweep: how much is dynamic pricing actually worth?

Sweeps ``seasonal_amplitude`` over {0.0, 0.1, 0.25, 0.4, 0.6} on the calibrated
config, baseline scenario otherwise unchanged, and evaluates fixed / gbm /
gbm_uniform / oracle over 30 shared seeds at each amplitude.

A = 0.0 is the control: the seasonal term is switched off, so the optimal price
stops moving *seasonally* and a fixed price should come close to the oracle.
(It is not a perfectly flat optimum — the weekday/weekend factor and competitor
drift still shift p*(t); the script quantifies exactly how much.)

No agent code is touched: agents are built through the existing registry and
driven through the existing harness.

Oracle caching. The oracle is deterministic given (config, seed): its price rule
is a grid search over a closed-form demand model and the only randomness is the
environment noise, which the seed fixes. Its episodes are cached to disk keyed by
a hash of the *full* config plus the seed. Note this buys nothing *within* one
sweep — seasonal_amplitude is part of the config, so every amplitude is a
genuinely different oracle problem and must be solved on its own — but it makes
re-runs of this script cheap. ``--verify-cache`` recomputes a couple of cached
episodes and checks they still match.

Run: ``python scripts/seasonal_amplitude_sweep.py [--seeds 30] [--verify-cache]``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dynpricing.env.config import EnvConfig
from dynpricing.env.demand import DemandModel
from dynpricing.env.market_env import MarketEnv
from dynpricing.agents.registry import build_agent
from dynpricing.eval.harness import Scenario, run_episode
from dynpricing.eval.stats import bootstrap_mean_ci, paired_comparison, summarize

CONFIG = "configs/calibrated.json"
AMPLITUDES = (0.0, 0.1, 0.25, 0.4, 0.6)
AGENTS = ("fixed", "gbm", "gbm_uniform", "oracle")
OUT_DIR = Path("results/seasonal_sweep")
CACHE_PATH = OUT_DIR / "oracle_cache.json"


# -- config helpers --------------------------------------------------------
def with_amplitude(base: EnvConfig, amplitude: float) -> EnvConfig:
    d = base.to_dict()
    d["seasonal_amplitude"] = float(amplitude)
    return EnvConfig.from_dict(d)


def config_key(cfg: EnvConfig) -> str:
    blob = json.dumps(cfg.to_dict(), sort_keys=True, default=list)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def optimal_price_dispersion(cfg: EnvConfig) -> tuple[float, float, float]:
    """(min, max, std) of the closed-form p*(t) over the horizon.

    Competitors are held at ref_price so this isolates the calendar/seasonal
    movement in the optimum — i.e. how much room a dynamic pricer even has.
    """
    demand = DemandModel(cfg)
    comps = [cfg.ref_price] * cfg.n_competitors
    stars = []
    for t in range(cfg.horizon):
        a = demand.intercept(comps, t)
        stars.append(float(np.clip((a / cfg.b + cfg.unit_cost) / 2.0,
                                   cfg.price_min, cfg.price_max)))
    arr = np.asarray(stars)
    return float(arr.min()), float(arr.max()), float(arr.std())


# -- episode running (with an oracle cache) --------------------------------
def load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


def run_agent(name: str, cfg: EnvConfig, seed: int, scenario: str,
              cache: dict | None) -> dict:
    """One (agent, seed) episode -> a metrics row. Oracle rows may come from cache."""
    key = f"{name}:{config_key(cfg)}:{seed}"
    if cache is not None and key in cache:
        row = dict(cache[key])
        row["scenario"] = scenario
        row["cached"] = True
        return row

    agent = build_agent(name, cfg, seed=seed)
    if getattr(agent, "requires_training", False):
        agent.train(make_env=lambda: MarketEnv(cfg), seed=seed)
    metrics = run_episode(agent, MarketEnv(cfg), seed=seed, scenario=scenario)
    row = metrics.to_row()
    row["cached"] = False
    if cache is not None:
        cache[key] = {k: v for k, v in row.items() if k != "cached"}
    return row


# -- reporting -------------------------------------------------------------
def bar(value: float, lo: float, hi: float, width: int = 46) -> str:
    frac = 0.0 if hi <= lo else (value - lo) / (hi - lo)
    return "#" * max(1, int(round(frac * width)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--amplitudes", default=None,
                    help="comma-separated override, e.g. '0.0,0.25'")
    ap.add_argument("--verify-cache", action="store_true",
                    help="recompute 2 cached oracle episodes and check they match")
    args = ap.parse_args()

    amplitudes = (tuple(float(a) for a in args.amplitudes.split(","))
                  if args.amplitudes else AMPLITUDES)
    seeds = list(range(args.seeds))
    base = EnvConfig.load(CONFIG)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = load_cache()
    n_cache_start = len(cache)

    print(f"Seasonal-amplitude sweep | config={CONFIG} | {len(AGENTS)} agents "
          f"x {len(amplitudes)} amplitudes x {len(seeds)} seeds")
    print(f"base seasonal_amplitude={base.seasonal_amplitude}, "
          f"weekend_uplift={base.weekend_uplift}, band=[{base.price_min:.3f}, "
          f"{base.price_max:.3f}]\n")

    print("How much room does a dynamic pricer have? "
          "closed-form p*(t), competitors held at ref_price:")
    print(f"  {'amplitude':>10}{'min p*':>10}{'max p*':>10}{'range':>9}{'std':>8}")
    for a in amplitudes:
        lo, hi, sd = optimal_price_dispersion(with_amplitude(base, a))
        print(f"  {a:>10.2f}{lo:>10.3f}{hi:>10.3f}{hi - lo:>9.3f}{sd:>8.3f}")

    if args.verify_cache and cache:
        print("\nverifying oracle cache ...")
        checked = 0
        for a in amplitudes[:1]:
            cfg = with_amplitude(base, a)
            for seed in seeds[:2]:
                key = f"oracle:{config_key(cfg)}:{seed}"
                if key not in cache:
                    continue
                fresh = run_agent("oracle", cfg, seed, "verify", cache=None)
                delta = abs(fresh["gross_profit"] - cache[key]["gross_profit"])
                print(f"  A={a} seed={seed}: |Δ| = {delta:.6f}  "
                      f"{'OK' if delta < 1e-6 else 'MISMATCH'}")
                checked += 1
        if not checked:
            print("  (nothing cached yet for these amplitudes)")

    # -- the sweep ---------------------------------------------------------
    rows: list[dict] = []
    for a in amplitudes:
        cfg = with_amplitude(base, a)
        scenario = f"A={a:.2f}"
        for name in AGENTS:
            use_cache = cache if name == "oracle" else None
            for seed in seeds:
                rows.append(run_agent(name, cfg, seed, scenario, use_cache))
            print(f"  {scenario} | {name:<12} done")

    CACHE_PATH.write_text(json.dumps(cache, indent=2))
    n_hits = sum(1 for r in rows if r.get("cached"))
    print(f"\noracle cache: {n_cache_start} entries on entry, {len(cache)} on exit, "
          f"{n_hits} episode(s) served from cache this run")

    clean = [{k: v for k, v in r.items() if k != "cached"} for r in rows]
    import pandas as pd

    pd.DataFrame(clean).to_csv(OUT_DIR / "metrics.csv", index=False)
    summ = summarize(clean)
    (OUT_DIR / "metrics_aggregated.json").write_text(json.dumps(summ, indent=2))

    # -- per-amplitude ladders --------------------------------------------
    pct: dict[str, dict[float, float]] = {n: {} for n in AGENTS}
    profits: dict[str, dict[float, float]] = {n: {} for n in AGENTS}
    for a in amplitudes:
        scenario = f"A={a:.2f}"
        print(f"\n[{scenario}]  (n={len(seeds)} seeds)")
        ceiling = float(np.mean([r["gross_profit"] for r in clean
                                 if r["agent"] == "oracle" and r["scenario"] == scenario]))
        for name in AGENTS:
            vals = [r["gross_profit"] for r in clean
                    if r["agent"] == name and r["scenario"] == scenario]
            mean, lo, hi, _ = bootstrap_mean_ci(vals)
            share = 100 * mean / ceiling
            pct[name][a] = share
            profits[name][a] = mean
            print(f"  {name:<13} profit={mean:10.1f}  "
                  f"[{lo:9.1f}, {hi:9.1f}]  ({share:6.2f}% of oracle)")

    # -- the curve ---------------------------------------------------------
    print("\n" + "=" * 78)
    print("CURVE — % of oracle vs seasonal amplitude")
    print("=" * 78)
    print(f"\n{'amplitude':>10}{'fixed':>10}{'gbm':>10}{'gbm_uniform':>14}{'oracle':>9}")
    for a in amplitudes:
        print(f"{a:>10.2f}{pct['fixed'][a]:>9.2f}%{pct['gbm'][a]:>9.2f}%"
              f"{pct['gbm_uniform'][a]:>13.2f}%{100.0:>8.1f}%")

    lo_axis = min(min(pct["fixed"].values()), min(pct["gbm_uniform"].values())) - 1
    print(f"\n(bars scaled to [{lo_axis:.1f}%, 100%])")
    for a in amplitudes:
        print(f"\n  A={a:.2f}")
        for name in ("fixed", "gbm_uniform", "oracle"):
            v = pct[name][a]
            print(f"    {name:<12}{v:6.2f}% |{bar(v, lo_axis, 100.0)}")

    # -- the three questions ----------------------------------------------
    print("\n" + "=" * 78)
    print("ANSWERS")
    print("=" * 78)

    gu = [pct["gbm_uniform"][a] for a in amplitudes]
    print(f"\n1. gbm_uniform across amplitudes: min={min(gu):.2f}% "
          f"max={max(gu):.2f}%  (spread {max(gu) - min(gu):.2f} pts)")
    print("   within the 95-99% band at every amplitude: "
          f"{'YES' if all(95.0 <= v <= 99.5 for v in gu) else 'NO'}")
    trend = gu[-1] - gu[0]
    print(f"   change from A={amplitudes[0]:.2f} to A={amplitudes[-1]:.2f}: "
          f"{trend:+.2f} pts")

    print("\n2. where does dynamic pricing earn the most "
          "(fixed furthest below gbm_uniform)?")
    gaps = {}
    for a in amplitudes:
        scenario = f"A={a:.2f}"
        res = paired_comparison(clean, "gbm_uniform", "fixed", scenario=scenario)
        gap_pts = pct["gbm_uniform"][a] - pct["fixed"][a]
        gaps[a] = (gap_pts, res)
        sig = "significant" if res.significant else "n.s. (CI spans 0)"
        print(f"   A={a:.2f}: gbm_uniform - fixed = {gap_pts:+6.2f} pts of oracle "
              f"| paired Δ={res.mean_diff:+9.1f} "
              f"CI=[{res.ci_low:+.1f}, {res.ci_high:+.1f}] {sig}")
    best = max(gaps, key=lambda k: gaps[k][0])
    print(f"   -> largest at A={best:.2f} ({gaps[best][0]:+.2f} pts, "
          f"{gaps[best][1].mean_diff:+,.1f} profit)")

    print("\n3. control at A=0.00 (no seasonality -> p* barely moves):")
    if 0.0 in gaps:
        lo0, hi0, sd0 = optimal_price_dispersion(with_amplitude(base, 0.0))
        res0 = gaps[0.0][1]
        print(f"   p*(t) range at A=0: [{lo0:.3f}, {hi0:.3f}] "
              f"(std {sd0:.3f}) — residual movement is the weekend factor, "
              "not seasonality")
        print(f"   fixed       = {pct['fixed'][0.0]:.2f}% of oracle")
        print(f"   gbm_uniform = {pct['gbm_uniform'][0.0]:.2f}% of oracle")
        print(f"   gbm_uniform - fixed: paired Δ={res0.mean_diff:+.1f} "
              f"CI=[{res0.ci_low:+.1f}, {res0.ci_high:+.1f}] "
              f"{'significant' if res0.significant else 'n.s.'}")

    out = {
        "amplitudes": list(amplitudes),
        "n_seeds": len(seeds),
        "pct_of_oracle": {n: {str(a): pct[n][a] for a in amplitudes} for n in AGENTS},
        "mean_profit": {n: {str(a): profits[n][a] for a in amplitudes} for n in AGENTS},
        "gbm_uniform_minus_fixed": {
            str(a): {"pts_of_oracle": gaps[a][0], **gaps[a][1].to_dict()}
            for a in amplitudes
        },
        "optimal_price_dispersion": {
            str(a): dict(zip(("min", "max", "std"),
                             optimal_price_dispersion(with_amplitude(base, a))))
            for a in amplitudes
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(out, indent=2))
    print(f"\n[ok] wrote {OUT_DIR}/metrics.csv, metrics_aggregated.json, summary.json")


if __name__ == "__main__":
    main()
