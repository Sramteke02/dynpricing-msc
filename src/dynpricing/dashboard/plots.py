"""Results dashboard (E6).

Reproduces the wireframe from the proposal: a bar chart of total profit per
agent (oracle shown as a dashed ceiling), a line chart of price over a season,
and a metrics table. Figures are rendered with Matplotlib.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt  # noqa: E402

# canonical display order matching the proposal's ladder
_ORDER = ["fixed", "cost_plus", "competitor_match", "random", "gbm", "llm", "oracle"]
_HIGHLIGHT = {"gbm", "llm"}


def _order_key(agent: str) -> int:
    base = agent.split(":")[0]
    return _ORDER.index(base) if base in _ORDER else len(_ORDER)


def build_dashboard(
    results_csv: str | Path,
    out_path: str | Path,
    *,
    scenario: str | None = None,
    price_paths: dict | None = None,
) -> Path:
    """Build the three-panel dashboard PNG from a metrics CSV."""
    import pandas as pd

    df = pd.read_csv(results_csv)
    if scenario:
        df = df[df["scenario"] == scenario]
    if df.empty:
        raise ValueError("no rows to plot (check the scenario filter)")

    # Bootstrap means + 95% CIs (Task 3), and the paired GBM-vs-LLM result.
    from dynpricing.eval.stats import summarize, paired_comparison, resolve_agent_name

    rows = df.to_dict("records")
    scen_name = scenario or df["scenario"].iloc[0]
    summ = [s for s in summarize(rows) if s["scenario"] == scen_name]
    summ.sort(key=lambda s: _order_key(s["agent"]))
    agg = pd.DataFrame(summ)
    n_seeds = int(agg["n_seeds"].iloc[0]) if not agg.empty else 0

    gbm = resolve_agent_name(rows, "gbm", scen_name)
    llm = resolve_agent_name(rows, "llm", scen_name)
    paired = (paired_comparison(rows, gbm, llm, "gross_profit", scen_name)
              if gbm and llm else None)

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1.2], hspace=0.62, wspace=0.25)
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_line = fig.add_subplot(gs[0, 1])
    ax_table = fig.add_subplot(gs[1, :])

    title = "Dynamic Pricing Results Dashboard"
    if scenario:
        title += f"  —  scenario: {scenario}"
    fig.suptitle(title, fontsize=15, fontweight="bold")

    # --- bar chart: total gross profit per agent (with 95% CI error bars) --
    oracle_profit = None
    oracle_rows = agg[agg["agent"].str.startswith("oracle")]
    if not oracle_rows.empty:
        oracle_profit = float(oracle_rows["gross_profit_mean"].iloc[0])

    bars = agg[~agg["agent"].str.startswith("oracle")]
    colors = ["#d62728" if a.split(":")[0] in _HIGHLIGHT else "#9aa0a6"
              for a in bars["agent"]]
    means = bars["gross_profit_mean"].to_numpy()
    err_low = means - bars["gross_profit_ci_low"].to_numpy()
    err_high = bars["gross_profit_ci_high"].to_numpy() - means
    labels = [a.split(":")[0] for a in bars["agent"]]
    ax_bar.bar(labels, means, color=colors,
               yerr=[err_low, err_high], capsize=4, ecolor="#333333")
    if oracle_profit is not None:
        ax_bar.axhline(oracle_profit, ls="--", color="#1f1f1f", lw=1.5,
                       label="oracle (ceiling)")
        ax_bar.legend(loc="upper left", fontsize=8)
    ax_bar.set_title(f"Total gross profit by agent (mean ± 95% CI, n={n_seeds})")
    ax_bar.set_ylabel("mean gross profit")
    ax_bar.tick_params(axis="x", rotation=45, labelsize=8)

    # --- line chart: price over a season ----------------------------------
    ax_line.set_title("Price over a season")
    ax_line.set_xlabel("day")
    ax_line.set_ylabel("price")
    if price_paths:
        for agent in sorted(price_paths, key=_order_key):
            path = price_paths[agent]
            ls = "--" if agent.split(":")[0] == "oracle" else "-"
            lw = 2.0 if agent.split(":")[0] in _HIGHLIGHT else 1.2
            ax_line.plot(path, label=agent.split(":")[0], linestyle=ls, lw=lw)
        ax_line.legend(fontsize=7, ncol=2)
    else:
        ax_line.text(0.5, 0.5, "price paths not provided\n(run with --save-paths)",
                     ha="center", va="center", transform=ax_line.transAxes,
                     color="grey")

    # --- metrics table (means with 95% CIs) -------------------------------
    ax_table.axis("off")
    cell_text = []
    for _, r in agg.iterrows():
        gp_ci = f"{r['gross_profit_mean']:.0f}  [{r['gross_profit_ci_low']:.0f}, {r['gross_profit_ci_high']:.0f}]"
        rev_ci = f"{r['revenue_mean']:.0f}  [{r['revenue_ci_low']:.0f}, {r['revenue_ci_high']:.0f}]"
        cell_text.append([
            r["agent"].split(":")[0],
            rev_ci,
            gp_ci,
            f"{r['market_share_mean'] * 100:.1f}%",
            f"{r['pricing_stability_mean']:.3f}",
        ])
    table = ax_table.table(
        cellText=cell_text,
        colLabels=["Agent", "Revenue [95% CI]", "Gross profit [95% CI]",
                   "Market share", "Stability"],
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.4)
    title_txt = f"Metrics by agent (mean across {n_seeds} seeds, bootstrap 95% CI)"
    if paired is not None:
        if paired.significant:
            verdict = (f"GBM vs LLM: Δ={paired.mean_diff:+.0f} "
                       f"[{paired.ci_low:+.0f}, {paired.ci_high:+.0f}] — "
                       f"SIGNIFICANT ({paired.winner.split(':')[0]} wins)")
        else:
            verdict = (f"GBM vs LLM: Δ={paired.mean_diff:+.0f} "
                       f"[{paired.ci_low:+.0f}, {paired.ci_high:+.0f}] — "
                       f"not significant (CI spans 0)")
        title_txt += f"\n{verdict}"
    ax_table.set_title(title_txt, pad=12, fontsize=10)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_price_paths(price_paths: dict, out_path: str | Path,
                     title: str = "Price over a season") -> Path:
    """Standalone price-over-time chart."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for agent in sorted(price_paths, key=_order_key):
        ax.plot(price_paths[agent], label=agent.split(":")[0])
    ax.set_title(title)
    ax.set_xlabel("day")
    ax.set_ylabel("price")
    ax.legend(fontsize=8)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path
