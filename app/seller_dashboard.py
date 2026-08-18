"""Seller-facing price explainer — what to charge, and why.

    streamlit run app/seller_dashboard.py

Set the market situation in the sidebar and press "Find the best price". The
dashboard shows the profit-maximising price for that situation, an LLM's
reasoning in plain words, what the gbm_uniform agent does, and how close each
gets. All seller-facing copy is deliberately jargon-free; the technical names
appear only in the small print at the foot of the page.

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
#: how many selling periods each method gets to adjust its price. One ±5% move
#: cannot cross the band, so a single-step comparison would measure the action
#: set rather than the method. 8 gives enough room to reach the optimum across
#: the whole demand range, so the dashboard opens on a working example.
DEFAULT_ADJUST_CHANCES = 8

#: plain-language demand levels -> the same 0-100 scale the maths already used.
DEMAND_LEVELS = {"Very quiet": 0, "Quiet": 25, "Normal": 50,
                 "Busy": 75, "Very busy": 100}


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
        return "very busy", "one of the busiest times of the year"
    if season_factor >= 1.1:
        return "busy", "busier than usual"
    if season_factor > 0.9:
        return "normal", "about as busy as usual"
    if season_factor > 0.65:
        return "quiet", "quieter than usual"
    return "very quiet", "one of the quietest times of the year"


def why_this_price(cfg: EnvConfig, a: float, p_star: float, season_factor: float,
                   competitor_mean: float) -> str:
    """Two short sentences. The price and the profit are already on screen as
    metrics, so this gives the reason and the consequence, not a restatement."""
    level, _ = demand_words(season_factor)
    units = units_at(cfg, a, p_star)
    versus = ("above" if p_star > competitor_mean * 1.02 else
              "below" if p_star < competitor_mean * 0.98 else "level with")
    if season_factor > 1.05:
        reason = (f"Trade is **{level}**, so shoppers will pay more before "
                  "sales start to drop away.")
    elif season_factor < 0.95:
        reason = (f"Trade is **{level}**, so a higher price would lose you "
                  "more in sales than it gains you per item.")
    else:
        reason = ("Trade is **about average**, so the best price sits in the "
                  "middle — enough profit per item, without losing customers.")
    return (f"{reason}\n\nThat is **{versus}** the {competitor_mean:.2f} other "
            f"sellers charge, and would sell you about **{units:.0f} units** "
            "each period. Any other price earns less.")


def summarise_moves(path: list[float]) -> str:
    """One plain sentence describing what a method did, from its price path.

    Written from the prices themselves, so it carries none of the model's own
    jargon (pacing, periods remaining, margin).
    """
    start, final = path[0], path[-1]
    ups = sum(1 for i in range(1, len(path)) if path[i] > path[i - 1] + 1e-9)
    downs = sum(1 for i in range(1, len(path)) if path[i] < path[i - 1] - 1e-9)
    peak, dip = max(path), min(path)

    if ups + downs == 0:
        return f"It left the price alone, staying at **{final:.2f}**."
    if abs(final - start) < 0.02:
        moved = "tried the price both ways but came back to where it started"
    elif final > start:
        moved = "raised the price"
        if downs and dip < start - 0.02:
            moved = "dropped the price at first, then raised it"
        elif downs:
            moved = "raised the price, easing back once or twice on the way"
    else:
        moved = "lowered the price"
        if ups and peak > start + 0.02:
            moved = "pushed the price up at first, then brought it back down"
        elif ups:
            moved = "lowered the price, nudging it up once or twice on the way"
    return (f"It {moved}, settling at **{final:.2f}** "
            f"after {ups + downs} change{'s' if ups + downs != 1 else ''}.")


# -- UI --------------------------------------------------------------------
def main() -> None:
    import altair as alt
    import streamlit as st

    st.set_page_config(page_title="What price should I charge?", layout="wide")
    cfg = EnvConfig.load(str(CONFIG_PATH))
    have_key = load_env_key()

    st.title("What price should I charge?")
    st.markdown("#### Set your situation on the left, then see the best price "
                "to charge and why.")

    with st.sidebar:
        st.header("Your situation")
        demand_label = st.select_slider(
            "How busy is the market right now?",
            options=list(DEMAND_LEVELS), value="Normal",
            help="Quiet times of year sell less at any price. Busy times sell "
                 "more, so you can charge more.")
        demand_pct = DEMAND_LEVELS[demand_label]

        competitor_mean = st.slider(
            "What are other sellers charging?",
            float(cfg.price_min), float(cfg.price_max), 2.20, 0.05,
            help="What similar sellers charge on average.")

        inventory = st.slider(
            "How much stock do you have left to sell?",
            0, 60000, 24000, 1000, format="%d units")

        day = st.slider(
            "How far into the season are you?", 0, 364, 120,
            format="Day %d of 365",
            help="Affects how long you have left to sell your stock.")
        st.caption(f"About {max(1, round(day / 30.4)):.0f} months in, "
                   f"{365 - day} days left to sell.")

        periods = st.slider(
            "How many chances to adjust the price before we compare?",
            1, 12, DEFAULT_ADJUST_CHANCES,
            help="Prices move in small steps (up to 5% at a time), so more "
                 "chances means more room to reach the best price. With too "
                 "few, a method may simply run out of room.")

        go = st.button("Find the best price", type="primary", width="stretch")
        st.divider()
        if have_key:
            st.caption("AI explanation: connected and ready.")
        else:
            st.caption("AI explanation: **set MISTRAL_API_KEY** to enable.")

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
        st.info("Set your situation on the left, then press "
                "**Find the best price**.")
        return
    if best_profit <= 0:
        st.error("Trade is so quiet that no price makes a profit here — every "
                 "price shoppers would accept is below what the stock costs "
                 "you. Try a busier setting.")
        return

    # ---------------- panel 1: the recommendation ----------------
    st.header("Your recommended price")
    left, right = st.columns([1, 1.4])
    with left:
        st.metric("Charge this", f"{p_star:.2f}")
        st.metric("You would earn each period", f"{best_profit:,.0f}")
        st.caption(f"Each unit costs you {cfg.unit_cost:.2f} to buy. "
                   f"Other sellers are charging about {competitor_mean:.2f}.")
    with right:
        st.markdown(why_this_price(cfg, a, p_star, season_factor, competitor_mean))

    curve = profit_curve(cfg, a)
    peak = pd.DataFrame({"price": [p_star], "profit": [best_profit]})
    line = alt.Chart(curve).mark_line(strokeWidth=2, color="#2a78d6").encode(
        x=alt.X("price:Q", title="the price you charge"),
        y=alt.Y("profit:Q", title="profit you make each period"),
        tooltip=[alt.Tooltip("price:Q", format=".2f"),
                 alt.Tooltip("profit:Q", format=",.0f")])
    mark = alt.Chart(peak).mark_point(size=140, filled=True, color="#2a78d6").encode(
        x="price:Q", y="profit:Q")
    label = alt.Chart(peak).mark_text(dy=-14, fontSize=13, color="#0b0b0b").encode(
        x="price:Q", y="profit:Q", text=alt.Text("price:Q", format=".2f"))
    st.altair_chart(line + mark + label, width="stretch")
    st.caption("The high point of the curve is the best price.")

    results: list[dict] = []
    start_price = float(competitor_mean)   # you currently match the market

    # ---------------- panel 2: the LLM ----------------
    st.subheader("Why this price? (an AI explains in plain words)")
    if not have_key:
        st.warning("**Set MISTRAL_API_KEY** to enable this panel. Create a "
                   "`.env` file containing `MISTRAL_API_KEY=...` (it is "
                   "git-ignored), then reload. Everything else works without it.")
    else:
        prog = st.progress(0.0, text="asking the AI…")
        try:
            llm = LLMAgent(mode="api", template_name="default_v5",
                           strict_llm=False, min_call_interval=8.0,
                           max_retries=3, backoff_base=2.0, cache=False)
            t0 = time.time()
            path, notes = settle(
                llm, cfg, start_price, a, competitor_mean, inventory, day, season,
                periods,
                on_step=lambda i, n, p: prog.progress(
                    i / n, text=f"change {i} of {n} — price now {p:.2f}"))
            prog.empty()
            final = path[-1]
            fell_back = [d for d in llm.log if d.used_fallback]
            if fell_back:
                st.error("The AI could not be reached for "
                         f"{len(fell_back)} of {periods} changes. Reason: "
                         f"{fell_back[0].fallback_reason}")
            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("The AI would charge", f"{final:.2f}",
                          delta=f"{final - start_price:+.2f} vs {start_price:.2f} today")
            with c2:
                st.markdown(summarise_moves(path))
                st.caption(f"Took {time.time() - t0:.0f}s to think it through.")
            with st.expander("See the AI's full reasoning"):
                for i, note in enumerate(notes, 1):
                    if note:
                        st.markdown(f"*Change {i} → {path[i]:.2f}:* {note}")
            results.append({"method": "AI suggestion", "price": final})
        except Exception as exc:                      # never take the page down
            prog.empty()
            st.error(f"The AI could not be reached: {exc}")

    # ---------------- panel 3: gbm_uniform ----------------
    st.subheader("What a trained pricing model suggests")
    with st.spinner("checking the trained model…"):
        gbm = st.session_state.get("_gbm")
        if gbm is None:
            gbm = build_agent("gbm_uniform", cfg, seed=0)
            gbm.train(make_env=lambda: MarketEnv(cfg), seed=0)
            st.session_state["_gbm"] = gbm
        gpath, _ = settle(gbm, cfg, start_price, a, competitor_mean, inventory,
                          day, season, periods)
    g1, g2 = st.columns([1, 2])
    with g1:
        st.metric("The model would charge", f"{gpath[-1]:.2f}",
                  delta=f"{gpath[-1] - start_price:+.2f} vs {start_price:.2f} today")
    with g2:
        st.markdown(summarise_moves(gpath))
        st.caption("Usually the closest to the best price in our testing — but "
                   "it **cannot tell you why**. It only gives you a number.")
    results.append({"method": "Trained pricing model", "price": gpath[-1]})

    # ---------------- panel 4: how close did each get ----------------
    st.subheader("How close did each get?")
    st.markdown("**Higher = closer to the most profit possible.**")
    rows = [{"method": "Best possible", "price": p_star, "pct": 100.0}]
    for r in results:
        rows.append({"method": r["method"], "price": r["price"],
                     "pct": 100 * profit_at(cfg, a, r["price"]) / best_profit})
    df = pd.DataFrame(rows)
    bars = alt.Chart(df).mark_bar(size=26, cornerRadiusEnd=4,
                                  color="#2a78d6").encode(
        y=alt.Y("method:N", sort=list(df.method), title=None),
        x=alt.X("pct:Q", title="share of the most profit possible (%)",
                scale=alt.Scale(domain=[0, 105])),
        tooltip=[alt.Tooltip("method:N"), alt.Tooltip("price:Q", format=".2f"),
                 alt.Tooltip("pct:Q", format=".1f")])
    text = alt.Chart(df).mark_text(align="left", dx=6).encode(
        y=alt.Y("method:N", sort=list(df.method)), x="pct:Q",
        text=alt.Text("pct:Q", format=".1f"))
    st.altair_chart(bars + text, width="stretch")
    st.dataframe(
        df.rename(columns={"method": "Method", "price": "Price it charges",
                           "pct": "Share of the most profit possible (%)"}),
        hide_index=True, width="stretch",
        column_config={
            "Price it charges": st.column_config.NumberColumn(format="%.2f"),
            "Share of the most profit possible (%)":
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
    # the honest verdict stays in the main text; the mechanics move out of the way
    st.markdown(verdict.strip() or
                "The trained model is usually closest to the best price; the AI "
                "explains its thinking but tends to move too little when the "
                "market is very quiet or very busy.")
    if gap > 0.5:
        st.caption(f"Even the best of them left {gap:.1f}% of the possible "
                   "profit on the table.")

    with st.expander("Why can't they always reach the best price?"):
        st.markdown(
            f"A price can only move about 5% at a time, and each method was "
            f"given **{periods} chances** to adjust. Starting from "
            f"{start_price:.2f}, the highest it could reach is "
            f"**{min(reach, cfg.price_max):.2f}**"
            + (f" — less than the best price of {p_star:.2f}, so nothing can "
               "get all the way there in this setting. Give them more chances "
               "to adjust and they get closer."
               if reach < p_star - 0.01 else
               ", so there was room to reach the best price. Any shortfall is "
               "the method's own judgement, not a lack of room.")
            + "\n\nTry the busiest and quietest settings to see where each one "
              "struggles."
        )
        st.caption(
            "The trained model is a gradient-boosting predict-then-optimise "
            "agent (`gbm_uniform`); the AI explanation comes from a large "
            "language model. Both are described in the project README."
        )


if __name__ == "__main__":
    main()
