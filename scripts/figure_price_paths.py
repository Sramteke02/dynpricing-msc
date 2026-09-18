"""Dissertation figure: price paths under strong seasonality (RQ2).

Draws, for a single seed over the 365-day season, on one axes:

1. the optimal price ``p*(t)`` from the backward-induction oracle;
2. the corrected gradient-boosting agent (``gbm_uniform``);
3. the LLM under prompt v5;
4. the LLM under prompt v4;
5. the fixed price, as a flat reference line.

The two LLM paths are read from the committed RQ2 episode files — no API call
is ever made here.  The ``oracle``, ``gbm_uniform`` and ``fixed`` paths are not
committed per seed (``results/*/price_paths.json`` is hardcoded to baseline
seed 0, see ``cli._collect_price_paths``), so they are *replayed* from the
calibrated config.  The replay is gated: each agent must reproduce the gross
profit committed in ``results/gbm_uniform/metrics.csv`` to within 1e-6, or the
script aborts without writing a figure.  That keeps the curves on the page
belonging to the same episodes as the reported numbers.

Output is greyscale only — series are told apart by line style, so the figure
survives black-and-white printing.  Writes a 9 x 4.5 in, 200 dpi PNG and prints
each path's standard deviation and correlation with ``p*(t)`` for the caption.

Run: ``python scripts/figure_price_paths.py [--seed 7]``
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (after the Agg backend is set)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dynpricing.env.config import EnvConfig  # noqa: E402
from dynpricing.env.market_env import MarketEnv  # noqa: E402
from dynpricing.agents.registry import build_agent  # noqa: E402
from dynpricing.eval.harness import default_scenarios, run_episode  # noqa: E402

SCENARIO = "strong_seasonality"
REPLAYED = ("oracle", "gbm_uniform", "fixed")
SEEDS_WITH_BOTH_PROMPTS = (7, 8, 9)
PROFIT_TOL = 1e-6

CONFIG = ROOT / "configs" / "calibrated.json"
METRICS = ROOT / "results" / "gbm_uniform" / "metrics.csv"
OUT = ROOT / "results" / "figures" / "price_paths_strong_seasonality.png"


def committed_profits(seed: int) -> dict[str, float]:
    """Gross profit per agent for this seed/scenario, as committed."""
    with METRICS.open() as fh:
        rows = [
            r for r in csv.DictReader(fh)
            if r["scenario"] == SCENARIO and int(r["seed"]) == seed
        ]
    if not rows:
        raise SystemExit(f"no committed metrics for {SCENARIO} seed {seed} in {METRICS}")
    return {r["agent"]: float(r["gross_profit"]) for r in rows}


def replay(seed: int) -> dict[str, np.ndarray]:
    """Re-simulate the non-LLM agents, refusing to return unverified paths."""
    cfg = EnvConfig.load(CONFIG)
    scenario = {s.name: s for s in default_scenarios(cfg)}[SCENARIO]
    want = committed_profits(seed)

    paths: dict[str, np.ndarray] = {}
    for name in REPLAYED:
        agent = build_agent(name, scenario.config, seed=seed)
        if getattr(agent, "requires_training", False):
            agent.train(make_env=lambda: MarketEnv(cfg), seed=seed)
        m = run_episode(agent, scenario.make_env(), seed=seed, scenario=SCENARIO)

        got, expected = m.gross_profit, want[name]
        if abs(got - expected) > PROFIT_TOL:
            raise SystemExit(
                f"ABORT: {name} replayed gross profit {got!r} != committed "
                f"{expected!r}. The config or the simulator has changed since "
                f"the results were committed; the figure would not match the "
                f"reported numbers."
            )
        print(f"  [ok] {name:12s} reproduced committed gross profit {expected:.6f}")
        paths[name] = np.asarray(m.prices, dtype=float)
    return paths


def llm_path(version: str, seed: int) -> np.ndarray:
    f = ROOT / "results" / f"rq2_llm_{version}" / f"seed{seed}_{SCENARIO}.json"
    return np.asarray(json.loads(f.read_text())["prices"], dtype=float)


def charged(prices: np.ndarray) -> np.ndarray:
    """The 365 prices actually charged.

    ``run_episode`` records the initial price before any action, then one price
    per step, so the raw path has 366 entries and element 0 is not a decision.
    """
    if prices.size != 366:
        raise SystemExit(f"expected a 366-point path, got {prices.size}")
    return prices[1:]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=7, choices=SEEDS_WITH_BOTH_PROMPTS,
                    help="seed to plot (default: 7)")
    args = ap.parse_args()
    seed = args.seed

    print(f"Replaying non-LLM agents for {SCENARIO}, seed {seed} (no API calls):")
    rp = replay(seed)

    series = [
        ("Optimal price $p^*(t)$ (oracle)", charged(rp["oracle"]),
         "-", "0.00", 1.7),
        ("Gradient boosting (gbm_uniform)", charged(rp["gbm_uniform"]),
         (0, (6, 2)), "0.42", 1.4),
        ("LLM, prompt v5", charged(llm_path("v5", seed)),
         (0, (5, 1.6, 1, 1.6)), "0.12", 1.4),
        ("LLM, prompt v4", charged(llm_path("v4", seed)),
         (0, (1.4, 1.8)), "0.30", 1.7),
        ("Fixed price", charged(rp["fixed"]),
         "-", "0.68", 1.1),
    ]

    day = np.arange(1, 366)
    star = series[0][1]

    print(f"\nCaption statistics, {SCENARIO} seed {seed} (365 charged prices):")
    print(f"  {'series':34s}{'sd':>9s}{'corr with p*':>14s}")
    for label, y, *_ in series:
        flat = np.ptp(y) == 0.0
        sd = 0.0 if flat else float(np.std(y, ddof=1))
        corr = "undefined" if flat else f"{np.corrcoef(y, star)[0, 1]:.4f}"
        print(f"  {label:34s}{sd:9.4f}{corr:>14s}")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for label, y, style, grey, lw in series:
        ax.plot(day, y, linestyle=style, color=grey, linewidth=lw, label=label,
                solid_capstyle="round", dash_capstyle="round")

    ax.set_xlabel("Day of season")
    ax.set_ylabel("Price")
    ax.set_title(f"Price paths under strong seasonality (seed {seed})", pad=10)
    ax.set_xlim(1, 365)
    ax.margins(y=0.06)
    ax.grid(True, color="0.90", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("0.55")
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors="0.35", labelcolor="0.20", length=3, width=0.8)

    leg = ax.legend(loc="lower left", frameon=True, framealpha=1.0, fontsize=8.5,
                    borderpad=0.6, labelspacing=0.5, handlelength=3.4, ncol=2,
                    columnspacing=1.6)
    leg.get_frame().set_edgecolor("0.80")
    leg.get_frame().set_linewidth(0.6)

    fig.subplots_adjust(left=0.075, right=0.985, top=0.905, bottom=0.175)
    fig.text(0.075, 0.035,
             f"Single episode, seed {seed}, strong-seasonality scenario; all five "
             f"agents face the identical demand and competitor realisation.",
             fontsize=8, color="0.35", ha="left")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, facecolor="white")
    print(f"\n[ok] wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
