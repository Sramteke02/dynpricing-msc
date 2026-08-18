"""Seller-facing price explainer — what to charge, and why.

    streamlit run app/seller_dashboard.py

Set the market situation in the sidebar and press "Find the best price". The
dashboard shows the profit-maximising price for that situation, an AI
assistant's reasoning in plain words, what the best automated agent does, and
how close each gets.

It imports the project's agents, environment and demand model and calls them —
it modifies nothing. The Mistral key is read from ``.env`` (or the environment)
exactly as the other tools do; with no key the AI panel says so and every other
panel still works.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dynpricing.env.config import EnvConfig
from dynpricing.env.market_env import ACTIONS, MarketEnv, MarketState
from dynpricing.agents.registry import build_agent
from dynpricing.agents.llm_agent import LLMAgent

CONFIG_PATH = ROOT / "configs" / "calibrated.json"
ENV_FILE = ROOT / ".env"
#: how many selling periods each agent gets to adjust its price. One ±5% move
#: cannot cross the band, so a single-step comparison would measure the action
#: set rather than the agent.
DEFAULT_PERIODS = 5


# -- key loading (same source as the CLI tools) ----------------------------
def load_env_key() -> bool:
    """Populate MISTRAL_API_KEY from .env if it is not already set."""
    if os.environ.get("MISTRAL_API_KEY"):
        return True
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            os.environ.setdefault(name.strip(), value.strip().strip("'\""))
    return bool(os.environ.get("MISTRAL_API_KEY"))


# -- demand maths ----------------------------------------------------------
# Same closed form as DemandModel.expected_units / optimal_price, with the
# seasonal factor supplied by the slider instead of read off the calendar, so
# the seller can dial demand from trough to peak directly.
def intercept(cfg: EnvConfig, season_factor: float, calendar: float,
              competitor_mean: float) -> float:
    """a = a0·S·C + d·competitor_mean — the demand curve's height."""
    return cfg.a0 * season_factor * calendar + cfg.d * competitor_mean


def units_at(cfg: EnvConfig, a: float, price: float) -> float:
    return max(0.0, a - cfg.b * price)


def profit_at(cfg: EnvConfig, a: float, price: float) -> float:
    return (price - cfg.unit_cost) * units_at(cfg, a, price)


def optimal_price(cfg: EnvConfig, a: float) -> float:
    """p* = (a/b + c)/2, clipped to the legal band."""
    return float(np.clip((a / cfg.b + cfg.unit_cost) / 2.0,
                         cfg.price_min, cfg.price_max))


def profit_curve(cfg: EnvConfig, a: float, n: int = 240) -> pd.DataFrame:
    prices = np.linspace(cfg.price_min, cfg.price_max, n)
    return pd.DataFrame({
        "price": prices,
        "profit": [profit_at(cfg, a, p) for p in prices],
    })


def season_index(season_factor: float, rising: bool) -> int:
    """Quarter of the seasonal cycle implied by level and direction."""
    if season_factor >= 1.0:
        return 0 if rising else 1
    return 3 if rising else 2


# -- running the agents ----------------------------------------------------
def make_state(cfg: EnvConfig, price: float, competitor_mean: float,
               inventory: int, day: int, season: int, last_units: float) -> MarketState:
    return MarketState(
        own_price=float(price),
        competitor_prices=(round(competitor_mean, 4), round(competitor_mean, 4)),
        unit_cost=float(cfg.unit_cost),
        demand_level=float(last_units),
        inventory=int(inventory),
        day_of_week=int(day % 7),
        day=int(day),
        season=int(season),
        horizon=int(cfg.horizon),
        price_min=float(cfg.price_min),
        price_max=float(cfg.price_max),
        ref_price=float(cfg.ref_price),
    )


def step_price(price: float, action: int, cfg: EnvConfig) -> float:
    _, mult = ACTIONS[int(action)]
    return float(np.clip(price * mult, cfg.price_min, cfg.price_max))


def settle(agent, cfg: EnvConfig, start: float, a: float, competitor_mean: float,
           inventory: int, day: int, season: int, periods: int,
           on_step=None) -> tuple[list[float], list[str]]:
    """Let one agent adjust its price for `periods` periods in a FIXED market.

    Returns (price path including the start, reasoning per period).
    """
    price = float(start)
    path, notes = [price], []
    for i in range(periods):
        last_units = units_at(cfg, a, price)
        state = make_state(cfg, price, competitor_mean, inventory, day + i,
                           season, last_units)
        action = agent.act(state)
        price = step_price(price, action, cfg)
        path.append(price)
        note = ""
        log = getattr(agent, "log", None)
        if log:
            note = log[-1].reasoning
        notes.append(note)
        if on_step:
            on_step(i + 1, periods, price)
    return path, notes


# -- plain-language reasons ------------------------------------------------
def demand_words(season_factor: float) -> tuple[str, str]:
    if season_factor >= 1.35:
        return "very high", "at the seasonal peak"
    if season_factor >= 1.1:
        return "high", "above the yearly average"
    if season_factor > 0.9:
        return "average", "around the yearly average"
    if season_factor > 0.65:
        return "low", "below the yearly average"
    return "very low", "near the seasonal trough"


def why_this_price(cfg: EnvConfig, a: float, p_star: float, season_factor: float,
                   competitor_mean: float) -> str:
    level, where = demand_words(season_factor)
    units = units_at(cfg, a, p_star)
    profit = profit_at(cfg, a, p_star)
    versus = ("above" if p_star > competitor_mean * 1.02 else
              "below" if p_star < competitor_mean * 0.98 else "level with")
    direction = ("Because demand is %s, customers will keep buying at a higher "
                 "price before volume drops away, so the profit-maximising "
                 "price is higher." % level) if season_factor >= 1.0 else (
                 "Because demand is %s, pushing the price up loses more sales "
                 "than it gains in margin, so the profit-maximising price is "
                 "lower." % level)
    return (
        f"Demand this period is **{level}** ({where}). {direction} "
        f"The best price is **{p_star:.2f}** — that is **{versus}** the "
        f"competitor average of {competitor_mean:.2f}.\n\n"
        f"At {p_star:.2f} you would sell about **{units:.0f} units** at a margin "
        f"of {p_star - cfg.unit_cost:.2f} each, earning roughly "
        f"**{profit:,.0f} per period**. Charging more or less than this earns "
        f"less: the curve below peaks exactly at {p_star:.2f}."
    )


# -- UI --------------------------------------------------------------------
def main() -> None:
    import altair as alt
    import streamlit as st

    st.set_page_config(page_title="What price should I charge?", layout="wide")
    cfg = EnvConfig.load(str(CONFIG_PATH))
    have_key = load_env_key()

    st.title("What price should I charge?")
    st.caption(
        "Set your situation on the left and press **Find the best price**. "
        "Every number is computed live for the situation you set."
    )

    with st.sidebar:
        st.header("Your situation")
        demand_pct = st.slider(
            "Demand this period", 0, 100, 50,
            help="0 = quietest time of year (trough). 100 = busiest (peak).")
        competitor_mean = st.slider(
            "What competitors are charging (average)",
            float(cfg.price_min), float(cfg.price_max), float(cfg.ref_price), 0.05)
        inventory = st.slider("Stock left (units)", 0, 60000, 24000, 1000)
        day = st.slider("Day of the selling season", 0, 364, 120,
                        help="How far through the year you are. Affects how "
                             "long you have left to sell your stock.")
        periods = st.slider("Periods each method may adjust over", 1, 12,
                            DEFAULT_PERIODS,
                            help="A price move is limited to ±5% per period, so "
                                 "each method needs a few periods to reach its "
                                 "preferred price.")
        go = st.button("Find the best price", type="primary", width="stretch")
        st.divider()
        if have_key:
            st.caption("AI assistant: connected (Mistral).")
        else:
            st.caption("AI assistant: **set MISTRAL_API_KEY** to enable.")

    # demand level -> seasonal factor, using the strong-seasonality amplitude so
    # the slider spans a market that really does move
    amplitude = 0.5
    season_factor = (1 - amplitude) + (demand_pct / 100.0) * (2 * amplitude)
    calendar = 1.0 + (cfg.weekend_uplift if (day % 7) >= 5 else 0.0)
    rising = (day % 365) < 91 or (day % 365) >= 274
    season = season_index(season_factor, rising)

    a = intercept(cfg, season_factor, calendar, competitor_mean)
    p_star = optimal_price(cfg, a)
    best_profit = profit_at(cfg, a, p_star)

    if not go:
        st.info("Set your situation in the sidebar, then press "
                "**Find the best price**.")
        return
    if best_profit <= 0:
        st.error("At this demand level no price earns a profit — every price "
                 "is at or below your unit cost. Try a higher demand level.")
        return

    # ---------------- panel 1: the recommendation ----------------
    st.subheader("1. Recommended price")
    left, right = st.columns([1, 1.4])
    with left:
        st.metric("Best price for this situation", f"{p_star:.2f}")
        st.metric("You would earn, per period", f"{best_profit:,.0f}")
        st.caption(f"Your cost is {cfg.unit_cost:.2f} per unit. "
                   f"Competitors average {competitor_mean:.2f}.")
    with right:
        st.markdown(why_this_price(cfg, a, p_star, season_factor, competitor_mean))

    curve = profit_curve(cfg, a)
    peak = pd.DataFrame({"price": [p_star], "profit": [best_profit]})
    line = alt.Chart(curve).mark_line(strokeWidth=2, color="#2a78d6").encode(
        x=alt.X("price:Q", title="price you charge"),
        y=alt.Y("profit:Q", title="profit per period"),
        tooltip=[alt.Tooltip("price:Q", format=".2f"),
                 alt.Tooltip("profit:Q", format=",.0f")])
    mark = alt.Chart(peak).mark_point(size=140, filled=True, color="#2a78d6").encode(
        x="price:Q", y="profit:Q")
    label = alt.Chart(peak).mark_text(dy=-14, fontSize=13, color="#0b0b0b").encode(
        x="price:Q", y="profit:Q", text=alt.Text("price:Q", format=".2f"))
    st.altair_chart(line + mark + label, width="stretch")
    st.caption("The peak of this curve is the best price. Left of it you are "
               "leaving margin on the table; right of it you lose more sales "
               "than the extra margin is worth.")

    results: list[dict] = []
    start_price = float(competitor_mean)   # you currently match the market

    # ---------------- panel 2: the LLM ----------------
    st.subheader("2. An AI assistant's reasoning")
    if not have_key:
        st.warning("**Set MISTRAL_API_KEY** to enable this panel. Create a "
                   "`.env` file containing `MISTRAL_API_KEY=...` (it is "
                   "git-ignored), then reload. The other panels work without it.")
    else:
        prog = st.progress(0.0, text="asking the assistant…")
        try:
            llm = LLMAgent(mode="api", template_name="default_v5",
                           strict_llm=False, min_call_interval=8.0,
                           max_retries=3, backoff_base=2.0, cache=False)
            t0 = time.time()
            path, notes = settle(
                llm, cfg, start_price, a, competitor_mean, inventory, day, season,
                periods,
                on_step=lambda i, n, p: prog.progress(
                    i / n, text=f"period {i} of {n} — price now {p:.2f}"))
            prog.empty()
            final = path[-1]
            fell_back = [d for d in llm.log if d.used_fallback]
            if fell_back:
                st.error("The assistant could not be reached for "
                         f"{len(fell_back)} of {periods} periods. Reason: "
                         f"{fell_back[0].fallback_reason}")
            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("Assistant's price", f"{final:.2f}",
                          delta=f"{final - start_price:+.2f} from {start_price:.2f}")
                st.caption(f"{time.time() - t0:.0f}s · {len(llm.log)} live calls")
            with c2:
                st.markdown("**Its reasoning, in its own words:**")
                for i, note in enumerate(notes, 1):
                    if note:
                        st.markdown(f"*Period {i} → {path[i]:.2f}:* {note}")
            results.append({"method": "AI assistant", "price": final})
        except Exception as exc:                      # never take the page down
            prog.empty()
            st.error(f"The assistant could not be reached: {exc}")

    # ---------------- panel 3: gbm_uniform ----------------
    st.subheader("3. Best automated method (gradient boosting)")
    with st.spinner("running the automated method…"):
        gbm = st.session_state.get("_gbm")
        if gbm is None:
            gbm = build_agent("gbm_uniform", cfg, seed=0)
            gbm.train(make_env=lambda: MarketEnv(cfg), seed=0)
            st.session_state["_gbm"] = gbm
        gpath, _ = settle(gbm, cfg, start_price, a, competitor_mean, inventory,
                          day, season, periods)
    g1, g2 = st.columns([1, 2])
    with g1:
        st.metric("Automated price", f"{gpath[-1]:.2f}",
                  delta=f"{gpath[-1] - start_price:+.2f} from {start_price:.2f}")
    with g2:
        st.markdown(
            "Across our 30-seed tests this method priced closest to optimal "
            "(99% of the best possible), but it **gives no explanation** — it "
            "optimises numerically against a demand curve it learned from data, "
            "and reports only a number.\n\n"
            "It learned that curve under *average* market conditions, so in an "
            "unusually busy or quiet period it is working outside what it has "
            "seen before."
        )
    results.append({"method": "Automated (gradient boosting)", "price": gpath[-1]})

    # ---------------- panel 4: how close did each get ----------------
    st.subheader("4. How close did each get?")
    rows = [{"method": "Best possible", "price": p_star, "pct": 100.0}]
    for r in results:
        rows.append({"method": r["method"], "price": r["price"],
                     "pct": 100 * profit_at(cfg, a, r["price"]) / best_profit})
    df = pd.DataFrame(rows)
    bars = alt.Chart(df).mark_bar(size=26, cornerRadiusEnd=4,
                                  color="#2a78d6").encode(
        y=alt.Y("method:N", sort=list(df.method), title=None),
        x=alt.X("pct:Q", title="% of the best possible profit",
                scale=alt.Scale(domain=[0, 105])),
        tooltip=[alt.Tooltip("method:N"), alt.Tooltip("price:Q", format=".2f"),
                 alt.Tooltip("pct:Q", format=".1f")])
    text = alt.Chart(df).mark_text(align="left", dx=6).encode(
        y=alt.Y("method:N", sort=list(df.method)), x="pct:Q",
        text=alt.Text("pct:Q", format=".1f"))
    st.altair_chart(bars + text, width="stretch")
    st.dataframe(
        df.rename(columns={"method": "Method", "price": "Price",
                           "pct": "% of best possible profit"}),
        hide_index=True, width="stretch",
        column_config={"Price": st.column_config.NumberColumn(format="%.2f"),
                       "% of best possible profit":
                           st.column_config.NumberColumn(format="%.1f")})
    # the honest line is computed from what actually happened, not asserted
    scored = [r for r in rows if r["method"] != "Best possible"]
    choke = a / cfg.b          # above this price nothing sells at all
    verdict = ""
    if scored:
        scored.sort(key=lambda r: -r["pct"])
        dead = [r for r in scored if r["pct"] < 0.5]
        if len(dead) == len(scored):
            verdict = (
                f"⚠️ **Neither method found a workable price here.** At this "
                f"demand level nobody buys above **{choke:.2f}**, and both "
                f"priced above it — so both would sell nothing and earn nothing, "
                f"while {p_star:.2f} would have earned {best_profit:,.0f}. ")
        elif dead:
            verdict = (f"⚠️ **{dead[0]['method']}** priced above {choke:.2f}, "
                       "where nothing sells at this demand level, so it earns "
                       "nothing. ")
        elif len(scored) >= 2 and abs(scored[0]["price"] - scored[1]["price"]) < 0.01:
            verdict = ("In this situation both methods landed on the same price, "
                       "so neither is closer. ")
        elif len(scored) >= 2:
            verdict = (f"In this situation **{scored[0]['method']}** got closer "
                       f"({scored[0]['pct']:.1f}% vs {scored[1]['pct']:.1f}%). ")
    gap = max((100 - r["pct"] for r in scored), default=0.0)
    reach = start_price * (1.05 ** periods)
    st.caption(
        verdict +
        "The automated method usually prices closest to optimal; the AI "
        "assistant explains its reasoning in plain words but may under-adjust "
        "when demand is extreme — try the demand slider at 0 or 100 and "
        f"compare. Each method was given **{periods} periods** to adjust and a "
        f"move is capped at ±5% per period, so from {start_price:.2f} the "
        f"highest reachable price is {min(reach, cfg.price_max):.2f}"
        + (f" — below the best price of {p_star:.2f}, which limits how close "
           "anything can get here. Increase the periods slider to give them room."
           if reach < p_star - 0.01 else ".")
        + (f" The best of them left {gap:.1f}% of the achievable profit on the "
           "table." if gap > 0.5 else "")
    )


if __name__ == "__main__":
    main()
