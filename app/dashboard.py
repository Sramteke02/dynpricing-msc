"""Interactive Streamlit dashboard — runs agents live through the existing harness.

Launch::

    streamlit run app/dashboard.py

This is a *live* tool: it builds agents from the existing registry and drives
them through the existing harness. It imports the project; it never modifies the
agents, the environment, or the harness. Every number shown is computed from the
run you just triggered — nothing is hardcoded.

The static, committed-results dashboard at ``results/dashboard/index.html`` is a
separate artefact and is untouched by this file.

All compute lives in plain functions below; the Streamlit UI is inside
``main()``, so the module can be imported and tested without starting a server.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dynpricing.env.config import EnvConfig
from dynpricing.env.market_env import MarketEnv
from dynpricing.agents.registry import build_agent
from dynpricing.eval.harness import Scenario, default_scenarios, run_episode
from dynpricing.eval.stats import bootstrap_mean_ci

CONFIG_PATH = ROOT / "configs" / "calibrated.json"
ORACLE_CACHE = ROOT / "results" / "seasonal_sweep" / "oracle_cache.json"

ALL_AGENTS = ["fixed", "cost_plus", "competitor_match", "random",
              "gbm", "gbm_uniform", "oracle", "llm"]
DEFAULT_AGENTS = [a for a in ALL_AGENTS if a != "llm"]

#: colour follows the entity, never its rank — a fixed slot per agent, so
#: changing the selection never repaints the survivors. Validated categorical
#: order from the data-viz palette.
SERIES_HUES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
AGENT_COLOR = {a: SERIES_HUES[i] for i, a in enumerate(ALL_AGENTS)}

METRIC_COLS = ["revenue", "gross_profit", "market_share", "pricing_stability"]


# -- config / scenarios ----------------------------------------------------
def load_config() -> EnvConfig:
    return EnvConfig.load(str(CONFIG_PATH))


def scenarios_for(cfg: EnvConfig) -> dict[str, Scenario]:
    """The harness's own scenario definitions, not a re-implementation."""
    return {s.name: s for s in default_scenarios(cfg)}


def config_key(cfg: EnvConfig) -> str:
    """Fingerprint used by the oracle cache (same recipe as the sweep script)."""
    blob = json.dumps(cfg.to_dict(), sort_keys=True, default=list).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


# -- oracle cache ----------------------------------------------------------
def load_oracle_cache() -> dict:
    if ORACLE_CACHE.exists():
        try:
            return json.loads(ORACLE_CACHE.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def oracle_from_cache(cfg: EnvConfig, seeds: list[int]) -> tuple[list[dict], list[int]]:
    """Return (cached rows, seeds still missing) for this exact config."""
    cache = load_oracle_cache()
    key = config_key(cfg)
    rows, missing = [], []
    for seed in seeds:
        entry = cache.get(f"oracle:{key}:{seed}")
        if entry is None:
            missing.append(seed)
        else:
            row = dict(entry)
            row["agent"] = "oracle"
            rows.append(row)
    return rows, missing


# -- running ---------------------------------------------------------------
def llm_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def run_agent_seeds(agent_name: str, cfg: EnvConfig, scenario: Scenario,
                    seeds: list[int], on_step=None) -> tuple[list[dict], dict]:
    """Drive one agent over the seeds. Returns (metric rows, {seed: price path}).

    Mirrors ``harness.evaluate_agent``: training agents are trained on the base
    config and evaluated on the scenario config.
    """
    rows, paths = [], {}
    for i, seed in enumerate(seeds):
        agent = build_agent(agent_name, scenario.config, seed=seed)
        if getattr(agent, "requires_training", False):
            agent.train(make_env=lambda: MarketEnv(cfg), seed=seed)
        metrics = run_episode(agent, scenario.make_env(), seed=seed,
                              scenario=scenario.name)
        row = metrics.to_row()
        rows.append(row)
        paths[seed] = metrics.prices
        if on_step:
            on_step(agent_name, i + 1, len(seeds))
    return rows, paths


def summarise(rows: list[dict]) -> pd.DataFrame:
    """Per-agent means with bootstrap 95% CIs, plus % of oracle."""
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    out = []
    for agent, grp in df.groupby("agent"):
        entry = {"agent": agent, "n_seeds": len(grp)}
        for col in METRIC_COLS:
            mean, lo, hi, _ = bootstrap_mean_ci(grp[col].to_numpy())
            entry[col] = mean
            if col == "gross_profit":
                entry["ci_low"], entry["ci_high"] = lo, hi
        out.append(entry)
    res = pd.DataFrame(out)
    oracle = res.loc[res.agent == "oracle", "gross_profit"]
    ceiling = float(oracle.iloc[0]) if len(oracle) else None
    res["pct_of_oracle"] = (100 * res.gross_profit / ceiling) if ceiling else np.nan
    if ceiling:
        res["ci_low_pct"] = 100 * res.ci_low / ceiling
        res["ci_high_pct"] = 100 * res.ci_high / ceiling
    order = {a: i for i, a in enumerate(ALL_AGENTS)}
    return res.sort_values("agent", key=lambda s: s.map(order)).reset_index(drop=True)


# -- charts ----------------------------------------------------------------
def ladder_chart(summary: pd.DataFrame):
    import altair as alt

    data = summary.dropna(subset=["pct_of_oracle"])
    base = alt.Chart(data)
    bars = base.mark_bar(size=20, cornerRadiusEnd=4, color=SERIES_HUES[0]).encode(
        y=alt.Y("agent:N", sort=list(data.agent), title=None),
        x=alt.X("pct_of_oracle:Q", title="% of oracle",
                scale=alt.Scale(domain=[0, 105])),
        tooltip=[alt.Tooltip("agent:N", title="agent"),
                 alt.Tooltip("pct_of_oracle:Q", title="% of oracle", format=".2f"),
                 alt.Tooltip("gross_profit:Q", title="gross profit", format=",.1f"),
                 alt.Tooltip("ci_low:Q", title="CI low", format=",.1f"),
                 alt.Tooltip("ci_high:Q", title="CI high", format=",.1f")],
    )
    err = base.mark_rule(strokeWidth=1.5, color="#52514e").encode(
        y=alt.Y("agent:N", sort=list(data.agent)),
        x="ci_low_pct:Q", x2="ci_high_pct:Q",
    )
    labels = base.mark_text(align="left", dx=6, fontSize=11).encode(
        y=alt.Y("agent:N", sort=list(data.agent)),
        x="pct_of_oracle:Q",
        text=alt.Text("pct_of_oracle:Q", format=".1f"),
    )
    ceiling = alt.Chart(pd.DataFrame({"x": [100.0]})).mark_rule(
        color="#c3c2b7", strokeWidth=1).encode(x="x:Q")
    return (ceiling + bars + err + labels).properties(height=max(160, 34 * len(data)))


def paths_chart(paths: dict[str, list[float]]):
    import altair as alt

    frames = []
    for agent, prices in paths.items():
        frames.append(pd.DataFrame({"day": range(len(prices)), "price": prices,
                                    "agent": agent}))
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    agents = [a for a in ALL_AGENTS if a in paths]
    hover = alt.selection_point(fields=["day"], nearest=True, on="pointermove",
                                empty=False)
    line = alt.Chart(df).mark_line(strokeWidth=2).encode(
        x=alt.X("day:Q", title="day of episode"),
        y=alt.Y("price:Q", title="price"),
        color=alt.Color("agent:N", title="agent",
                        scale=alt.Scale(domain=agents,
                                        range=[AGENT_COLOR[a] for a in agents])),
    )
    points = line.mark_point(size=44, filled=True).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip("agent:N"), alt.Tooltip("day:Q"),
                 alt.Tooltip("price:Q", format=".3f")],
    ).add_params(hover)
    return (line + points).properties(height=340)


# -- UI --------------------------------------------------------------------
def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Dynamic Pricing — live dashboard",
                       layout="wide")
    st.title("Dynamic Pricing — live agent runner")
    st.caption(
        "Runs the selected agents **live** through the existing harness and "
        "reports what that run produced. Nothing here is hardcoded. The static "
        "dashboard built from committed results lives at "
        "`results/dashboard/index.html`."
    )

    if not CONFIG_PATH.exists():
        st.error(f"config not found: {CONFIG_PATH.relative_to(ROOT)} — run "
                 "`dynpricing calibrate` first.")
        return
    cfg = load_config()
    scenarios = scenarios_for(cfg)

    with st.sidebar:
        st.header("Run settings")
        agents = st.multiselect("Agents", ALL_AGENTS, default=DEFAULT_AGENTS)
        scenario_name = st.selectbox("Scenario", list(scenarios))
        n_seeds = st.slider("Seeds", min_value=1, max_value=30, value=5,
                            help="More seeds tighten the CIs; the GBM agents "
                                 "retrain per seed, so this drives the runtime.")
        run = st.button("Run", type="primary", width="stretch")
        st.divider()
        st.caption(
            f"config `{CONFIG_PATH.name}` · fingerprint `{config_key(cfg)}` · "
            f"horizon {cfg.horizon} · band [{cfg.price_min:.3f}, "
            f"{cfg.price_max:.3f}] · inventory {cfg.init_inventory:,}")
        if not llm_available():
            st.caption("`OPENAI_API_KEY` is not set — the llm agent will be "
                       "skipped, not faked.")

    if run:
        st.session_state["result"] = execute_run(
            st, cfg, scenarios[scenario_name], agents, list(range(n_seeds)))

    result = st.session_state.get("result")
    if not result:
        st.info("Pick agents, a scenario and a seed count in the sidebar, then "
                "press **Run**.")
        return
    render_result(st, result)


def execute_run(st, cfg: EnvConfig, scenario: Scenario, agents: list[str],
                seeds: list[int]) -> dict:
    """Do the work, driving a progress indicator. Returns everything to render."""
    rows: list[dict] = []
    paths: dict[str, list[float]] = {}
    notes: list[str] = []
    skipped: list[tuple[str, str]] = []
    started = time.time()

    live = [a for a in agents if a != "llm"]
    if "llm" in agents:
        if llm_available():
            live.append("llm")
        else:
            skipped.append(("llm", "pending — set OPENAI_API_KEY"))

    progress = st.progress(0.0, text="starting…")
    total = max(1, len(live))
    done = 0

    for agent_name in live:
        # the oracle is deterministic given (config, seed): reuse the committed
        # cache when the config matches exactly, and say which happened.
        if agent_name == "oracle":
            cached, missing = oracle_from_cache(scenario.config, seeds)
            rows.extend(cached)
            if cached and not missing:
                notes.append(f"oracle: all {len(cached)} seed(s) loaded from "
                             f"`results/seasonal_sweep/oracle_cache.json` "
                             f"(config fingerprint `{config_key(scenario.config)}` "
                             f"matches) — not recomputed.")
            elif cached:
                notes.append(f"oracle: {len(cached)} seed(s) from cache, "
                             f"{len(missing)} computed live.")
            else:
                notes.append(f"oracle: computed live — no cache entry for this "
                             f"config (fingerprint `{config_key(scenario.config)}`).")
            if missing:
                progress.progress(done / total, text=f"running oracle "
                                  f"({len(missing)} uncached seed(s))…")
                fresh, fresh_paths = run_agent_seeds("oracle", cfg, scenario, missing)
                rows.extend(fresh)
                paths.update({"oracle": fresh_paths[seeds[0]]}
                             if seeds and seeds[0] in fresh_paths else {})
            if seeds and "oracle" not in paths:
                # price paths are not cached; one live episode for the path only
                _, p = run_agent_seeds("oracle", cfg, scenario, [seeds[0]])
                paths["oracle"] = p[seeds[0]]
                notes.append("oracle: seed-0 price path computed live "
                             "(the cache stores metrics, not paths).")
            done += 1
            progress.progress(done / total, text="oracle done")
            continue

        def on_step(name, i, n):
            progress.progress(min(1.0, (done + i / n) / total),
                              text=f"running {name} — seed {i}/{n}")

        try:
            agent_rows, agent_paths = run_agent_seeds(
                agent_name, cfg, scenario, seeds, on_step=on_step)
        except Exception as exc:  # a broken agent must not take the page down
            skipped.append((agent_name, f"failed: {exc}"))
            done += 1
            continue
        rows.extend(agent_rows)
        if seeds and seeds[0] in agent_paths:
            paths[agent_name] = agent_paths[seeds[0]]
        done += 1
        progress.progress(done / total, text=f"{agent_name} done")

    progress.empty()
    return {
        "rows": rows, "paths": paths, "notes": notes, "skipped": skipped,
        "scenario": scenario.name, "seeds": seeds,
        "elapsed": time.time() - started,
        "summary": summarise(rows),
    }


def render_result(st, result: dict) -> None:
    summary: pd.DataFrame = result["summary"]
    st.subheader(f"{result['scenario']} · {len(result['seeds'])} seed(s) · "
                 f"{result['elapsed']:.1f}s")

    for agent_name, reason in result["skipped"]:
        st.warning(f"**{agent_name}** — {reason} (skipped, not estimated)")
    for note in result["notes"]:
        st.caption(note)

    if summary.empty:
        st.info("No agents produced results.")
        return
    if "oracle" not in set(summary.agent):
        st.warning("The oracle was not selected, so there is no ceiling to "
                   "normalise against — the % of oracle chart is hidden.")
    else:
        st.markdown("#### % of oracle (bars) with bootstrap 95% CI")
        st.altair_chart(ladder_chart(summary), width="stretch")

    if result["paths"]:
        seed0 = result["seeds"][0]
        st.markdown(f"#### Price path, seed {seed0}")
        st.caption("The GBM ratchet shows up here: the original `gbm` climbs "
                   "past the demand choke to `price_max` and stays there, while "
                   "`gbm_uniform` settles near the true optimum.")
        chart = paths_chart(result["paths"])
        if chart is not None:
            st.altair_chart(chart, width="stretch")

    st.markdown("#### Metrics")
    table = summary.rename(columns={
        "revenue": "revenue", "gross_profit": "gross profit",
        "market_share": "market share", "pricing_stability": "stability",
        "pct_of_oracle": "% of oracle"})
    cols = ["agent", "n_seeds", "revenue", "gross profit", "ci_low", "ci_high",
            "market share", "stability", "% of oracle"]
    st.dataframe(
        table[[c for c in cols if c in table.columns]],
        hide_index=True, width="stretch",
        column_config={
            "revenue": st.column_config.NumberColumn(format="%.1f"),
            "gross profit": st.column_config.NumberColumn(format="%.1f"),
            "ci_low": st.column_config.NumberColumn("CI low", format="%.1f"),
            "ci_high": st.column_config.NumberColumn("CI high", format="%.1f"),
            "market share": st.column_config.NumberColumn(format="%.3f"),
            "stability": st.column_config.NumberColumn(format="%.3f"),
            "% of oracle": st.column_config.NumberColumn(format="%.2f"),
        })
    st.download_button("Download this run as CSV",
                       pd.DataFrame(result["rows"]).to_csv(index=False),
                       file_name=f"live_run_{result['scenario']}.csv",
                       mime="text/csv")


if __name__ == "__main__":
    main()
