"""Seller-facing price explainer — what to charge, and why.

    streamlit run app/seller_dashboard.py

Set the market situation in the sidebar and press "Suggest a price". The
dashboard shows ONE suggested price for that situation -- the LLM agent's --
with its reasoning in plain words and a profit-at-each-price chart.

This is the seller view: it deliberately carries no oracle, no comparison
against other agents and no "best possible" figure. Those belong to the
research tooling (`dynpricing run`, `scripts/`, `results/dashboard/index.html`).
The honest framing here is a *suggestion with reasoning*, not an optimum.

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
from dynpricing.env.market_env import ACTIONS, MarketState
from dynpricing.agents.llm_agent import LLMAgent

CONFIG_PATH = ROOT / "configs" / "calibrated.json"
ENV_FILE = ROOT / ".env"
#: how many small steps the agent may take to reach its price. A move is capped
#: at ±5%, so one step cannot cross the band; 8 gives it room across the whole
#: demand range. Fixed rather than exposed -- a seller should not have to reason
#: about the action set.
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
    st.markdown("#### Set your situation on the left, then get a suggested "
                "price and the reasoning behind it.")

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

        go = st.button("Suggest a price", type="primary", width="stretch")
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

    periods = DEFAULT_ADJUST_CHANCES   # prices move in small steps; fixed here

    if not go:
        st.info("Set your situation on the left, then press "
                "**Suggest a price**.")
        return
    if best_profit <= 0:
        st.error("Trade is so quiet that no price makes a profit here — every "
                 "price shoppers would accept is below what the stock costs "
                 "you. Try a busier setting.")
        return

    # ---------------- the suggested price ----------------
    if not have_key:
        st.warning("**Set MISTRAL_API_KEY** to get a suggested price. Create a "
                   "`.env` file containing `MISTRAL_API_KEY=...` (it is "
                   "git-ignored), then reload.")
        _profit_chart(st, alt, cfg, a, None)
        return

    start_price = float(competitor_mean)   # you currently match the market
    st.info("Thinking about your situation… this takes about a minute.",
            icon="⏳")
    prog = st.progress(0.0, text="Thinking…")
    try:
        llm = LLMAgent(mode="api", template_name="default_v5",
                       strict_llm=False, min_call_interval=8.0,
                       max_retries=3, backoff_base=2.0, cache=False)
        t0 = time.time()
        path, notes = settle(
            llm, cfg, start_price, a, competitor_mean, inventory, day, season,
            periods,
            on_step=lambda i, n, pr: prog.progress(
                i / n, text=f"Thinking… step {i} of {n}"))
        prog.empty()
    except Exception as exc:                       # never take the page down
        prog.empty()
        st.error(f"The AI could not be reached: {exc}")
        _profit_chart(st, alt, cfg, a, None)
        return

    final = path[-1]
    level, where = demand_words(season_factor)

    st.header("Suggested price")
    c1, c2 = st.columns([1, 1.6])
    with c1:
        st.metric("Suggested price", f"{final:.2f}",
                  delta=f"{final - start_price:+.2f} vs {start_price:.2f} now")
    with c2:
        st.markdown(f"**Why:** {summarise_moves(path)}")
        st.caption(f"Trade is {level} — {where}. Other sellers are around "
                   f"{competitor_mean:.2f}.")

    fell_back = [d for d in llm.log if d.used_fallback]
    if fell_back:
        st.warning(f"The AI could not be reached for {len(fell_back)} of "
                   f"{periods} steps, so part of this was worked out without "
                   f"it. Reason: {fell_back[0].fallback_reason}")

    with st.expander("See the AI's full reasoning"):
        for i, note in enumerate(notes, 1):
            if note:
                st.markdown(f"*Step {i} → {path[i]:.2f}:* {note}")
        st.caption(f"Took {time.time() - t0:.0f}s. The price moves in small "
                   "steps, up to 5% at a time.")

    _profit_chart(st, alt, cfg, a, final)

    st.caption("This is a suggested price with reasoning, not a guaranteed "
               "optimum.")


def _profit_chart(st, alt, cfg, a, marked: float | None) -> None:
    """Profit at each price, with the suggested price marked if we have one."""
    curve = profit_curve(cfg, a)
    chart = alt.Chart(curve).mark_line(strokeWidth=2, color="#2a78d6").encode(
        x=alt.X("price:Q", title="the price you charge"),
        y=alt.Y("profit:Q", title="profit you make each period"),
        tooltip=[alt.Tooltip("price:Q", format=".2f"),
                 alt.Tooltip("profit:Q", format=",.0f")])
    if marked is not None:
        here = pd.DataFrame({"price": [marked],
                             "profit": [profit_at(cfg, a, marked)]})
        chart = (chart
                 + alt.Chart(here).mark_point(size=150, filled=True,
                                              color="#eb6834").encode(
                     x="price:Q", y="profit:Q")
                 + alt.Chart(here).mark_text(dy=-14, fontSize=13).encode(
                     x="price:Q", y="profit:Q",
                     text=alt.Text("price:Q", format=".2f")))
    st.altair_chart(chart, width="stretch")
    st.caption("What you would earn at each price. The dot is the suggested "
               "price." if marked is not None else
               "What you would earn at each price.")


if __name__ == "__main__":
    main()
