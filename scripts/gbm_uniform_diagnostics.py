"""Diagnostics for ``gbm_uniform`` — the causal test of the extrapolation claim.

Prints, on the calibrated config:

1. the price range each agent's *training data* covers (original vs uniform);
2. whether the train-range clamp ever binds during evaluation;
3. the baseline seed-0 price path of ``gbm`` (the documented failing seed) next
   to ``gbm_uniform``'s;
4. a 2x2 ablation over 30 baseline seeds that attributes any recovery to
   exploration coverage, to the clamp, or to both.

Run: ``python scripts/gbm_uniform_diagnostics.py``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dynpricing.env.config import EnvConfig
from dynpricing.env.demand import DemandModel
from dynpricing.env.market_env import MarketEnv
from dynpricing.agents.gbm_uniform_agent import UniformExplorationGBMAgent
from dynpricing.agents.registry import build_agent
from dynpricing.eval.harness import Scenario, run_episode
from dynpricing.eval.stats import bootstrap_mean_ci

CONFIG = "configs/calibrated.json"
N_SEEDS = 30
FAILING_SEED = 0


def rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def train_agent(cfg, *, uniform: bool, clamp: bool, seed: int, verbose: bool = False):
    agent = UniformExplorationGBMAgent(
        seed=seed, uniform_exploration=uniform, clamp_to_train_max=clamp,
        verbose=verbose,
    )
    agent.train(make_env=lambda: MarketEnv(cfg), seed=seed)
    return agent


def choke_price(cfg) -> float:
    """Price at which true demand hits zero at the neutral competitor level."""
    demand = DemandModel(cfg)
    return demand.intercept([cfg.ref_price] * cfg.n_competitors, 0) / cfg.b


def main() -> None:
    cfg = EnvConfig.load(CONFIG)
    scen = Scenario("baseline", cfg)
    choke = choke_price(cfg)

    rule("1. TRAINING COVERAGE  (seed 0, 40 exploration episodes)")
    print(f"legal band [{cfg.price_min:.3f}, {cfg.price_max:.3f}]   "
          f"true choke price (day 0) ~ {choke:.3f}\n")

    original = train_agent(cfg, uniform=False, clamp=False, seed=0)
    print("ORIGINAL gbm exploration (random moves over {hold,+5%,-5%,-10%}):")
    print(f"  training prices: [{original.train_price_min:.3f}, "
          f"{original.train_price_max:.3f}]")

    uniform = train_agent(cfg, uniform=True, clamp=True, seed=0)
    print("\nNEW gbm_uniform exploration (uniform over the band):")
    print(uniform.coverage_report())

    rule("2. DOES THE OPTIMISER CLAMP EVER BIND?")
    env = scen.make_env()
    m_uniform = run_episode(uniform, env, seed=FAILING_SEED, scenario="baseline")
    print(uniform.clamp_report())
    print("\nInterpretation: with training data spanning the whole band, "
          "train_max == price_max,\nso no reachable price can exceed it — the "
          "clamp is inert. The coverage change,\nnot the clamp, is doing the work "
          "(see the ablation in section 4).")

    rule(f"3. PRICE PATH ON THE FAILING SEED ({FAILING_SEED}, baseline)")
    gbm = build_agent("gbm", cfg, seed=FAILING_SEED)
    gbm.train(make_env=lambda: MarketEnv(cfg), seed=FAILING_SEED)
    m_gbm = run_episode(gbm, scen.make_env(), seed=FAILING_SEED, scenario="baseline")

    for label, m in (("gbm (original)", m_gbm), ("gbm_uniform", m_uniform)):
        p = np.asarray(m.prices)
        at_max = int(np.sum(p >= cfg.price_max - 1e-6))
        above_choke = int(np.sum(p > choke))
        print(f"\n{label}:")
        print(f"  first 20 steps : {[round(float(x), 3) for x in p[:20]]}")
        print(f"  last 10 steps  : {[round(float(x), 3) for x in p[-10:]]}")
        print(f"  min / max      : {p.min():.3f} / {p.max():.3f}")
        print(f"  steps at price_max ({cfg.price_max:.3f}) : {at_max}/{len(p)}")
        print(f"  steps above the choke ({choke:.3f})   : {above_choke}/{len(p)}")
        print(f"  gross profit   : {m.gross_profit:,.1f}")

    rule(f"4. ABLATION (2x2) — which change recovers performance? "
         f"(baseline, {N_SEEDS} seeds)")
    seeds = list(range(N_SEEDS))

    cells: dict[tuple[bool, bool], list[float]] = {}
    binds: dict[tuple[bool, bool], tuple[int, int]] = {}
    for uniform_ in (False, True):
        for s in seeds:
            agent = train_agent(cfg, uniform=uniform_, clamp=False, seed=s)
            for clamp_ in (False, True):
                agent.clamp_to_train_max = clamp_
                agent.n_clamp_binds = agent.n_decisions = 0
                agent.n_candidates_excluded = 0
                profit = run_episode(agent, scen.make_env(), seed=s).gross_profit
                cells.setdefault((uniform_, clamp_), []).append(profit)
                b, d = binds.get((uniform_, clamp_), (0, 0))
                binds[(uniform_, clamp_)] = (b + agent.n_clamp_binds,
                                             d + agent.n_decisions)

    oracle_profits = [
        run_episode(build_agent("oracle", cfg, seed=s), scen.make_env(),
                    seed=s).gross_profit
        for s in seeds
    ]
    oracle = float(np.mean(oracle_profits))

    original = [
        run_episode(build_agent("gbm", cfg, seed=s).train(
            make_env=lambda: MarketEnv(cfg), seed=s),
            scen.make_env(), seed=s).gross_profit
        for s in seeds
    ]
    drift_noclamp = cells[(False, False)]
    max_dev = max(abs(a - b) for a, b in zip(original, drift_noclamp))
    print(f"\nfaithfulness check — (drift, no clamp) vs the original gbm agent:"
          f" max per-seed |Δ| = {max_dev:,.4f}"
          f"  ({'IDENTICAL' if max_dev < 1e-6 else 'DIFFERS'})")

    def paired(a: list[float], b: list[float]) -> tuple[float, float, float, bool]:
        diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
        mean, lo, hi, _ = bootstrap_mean_ci(diff)
        return mean, lo, hi, (lo > 0 or hi < 0)

    labels = {
        (False, False): "drift-only (ORIGINAL gbm)",
        (False, True): "drift + clamp",
        (True, False): "uniform-only",
        (True, True): "uniform + clamp (gbm_uniform)",
    }
    full = cells[(True, True)]

    print(f"\n{'variant':<32}{'profit':>11}{'% oracle':>10}{'clamp binds':>13}"
          f"{'paired Δ vs original':>24}{'95% CI':>26}")
    for key in ((False, False), (False, True), (True, False), (True, True)):
        vals = cells[key]
        b, d = binds[key]
        bind_pct = f"{100 * b / d:.2f}%" if key[1] else "—"
        mean = float(np.mean(vals))
        if key == (False, False):
            delta, ci = "—", "—"
        else:
            md, lo, hi, sig = paired(vals, cells[(False, False)])
            delta = f"{md:+,.1f}"
            ci = f"[{lo:+,.1f}, {hi:+,.1f}]{'' if sig else ' n.s.'}"
        print(f"{labels[key]:<32}{mean:>11,.1f}{100 * mean / oracle:>9.1f}%"
              f"{bind_pct:>13}{delta:>24}{ci:>26}")
    print(f"{'oracle (ceiling)':<32}{oracle:>11,.1f}{100.0:>9.1f}%")

    print("\nThe decisive contrast — uniform-only vs the full agent "
          "(same fitted model, clamp is the ONLY difference):")
    md, lo, hi, sig = paired(cells[(True, False)], full)
    n_ident = sum(1 for a, b in zip(cells[(True, False)], full) if abs(a - b) < 1e-9)
    print(f"  paired Δ = {md:+,.4f}   95% CI = [{lo:+,.4f}, {hi:+,.4f}]   "
          f"{'SIGNIFICANT' if sig else 'NOT distinguishable from zero'}")
    print(f"  seeds with byte-identical profit: {n_ident}/{len(full)}")

    out = {
        "seeds": seeds,
        "oracle": oracle_profits,
        "original_gbm": original,
        **{f"uniform={k[0]}_clamp={k[1]}": v for k, v in cells.items()},
    }
    path = Path("results/gbm_uniform/ablation_per_seed.json")
    path.write_text(json.dumps(out, indent=2))
    print(f"\n[ok] wrote per-seed profits -> {path}")


if __name__ == "__main__":
    main()
