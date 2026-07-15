"""LLM-based pricing agent (D1-D3).

The agent renders the market state as a natural-language prompt, asks an LLM to
reason and return a *structured* JSON decision, parses it (with a robust
per-call fallback), and logs everything for qualitative analysis and cost/
reliability auditing.

Two explicit modes (no silent substitution):

* ``mode="api"`` (default) -- the real OpenAI path. Requires ``OPENAI_API_KEY``.
  If the key (or the ``openai`` package) is missing the agent **fails loudly**
  at construction with a clear message. This is the mode that tests RQ2.
* ``mode="heuristic"`` -- an explicit, clearly-logged rule-of-thumb used as a
  baseline / offline fallback. Every decision is flagged ``used_fallback=True``.

For reproducibility the default model is a **pinned dated snapshot** and the
temperature defaults to **0**. Every call logs the model id, template name, the
full rendered prompt, the raw response, parsed decision, latency and token
usage (see :class:`LLMDecision` and :meth:`dump_log`).

Supporting the desirable requirements:
* **D1** structured decision + reasoning via the OpenAI Chat Completions API.
* **D2** full per-call logging and a prompt template that is a constructor
  argument (an experimental variable).
* **D3** the model id is a constructor argument, so a smaller and a larger model
  can be compared under identical conditions.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from dynpricing.agents.base import Agent, action_to_reach_price
from dynpricing.env.market_env import ACTIONS, MarketState

#: Pinned, dated model snapshot for reproducibility. Override via the ``model``
#: argument (e.g. a larger model) to support the D3 capability comparison.
DEFAULT_MODEL = "gpt-4o-mini-2024-07-18"

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert revenue-management pricing analyst for an online "
    "retailer. You set the price of a single product each period to maximise "
    "total gross profit over the whole selling horizon, balancing margin, "
    "competitor prices, demand, seasonality AND inventory pacing. You respond "
    "ONLY with a single compact JSON object and nothing else."
)

#: The default user-prompt template. It deliberately exposes remaining
#: inventory and periods-left and invites explicit pacing reasoning, which is
#: essential for RQ3. Treat the template (and its name) as an experimental
#: variable per D2.
DEFAULT_USER_TEMPLATE = """You are pricing one product for the next selling period in a competitive market.

Current market state:
- Our current price: {own_price:.2f}  (allowed band {price_min:.2f} to {price_max:.2f})
- Unit cost: {unit_cost:.2f}
- Competitor prices: {competitor_prices}
- Units sold last period: {demand_level:.0f}
- Remaining inventory: {inventory} units
- Periods elapsed: {day} of {horizon}; PERIODS REMAINING: {steps_left}
- Even sell-through to clear stock: about {sell_rate:.1f} units per remaining period
- Day-of-week index (0=Mon..6=Sun): {day_of_week}; season index (0-3): {season}

Think about margin vs. volume AND inventory pacing: with {steps_left} periods and
{inventory} units left, pricing too low dumps stock early and wastes margin, while
pricing too high leaves stock unsold at the end of the horizon.

Choose exactly one action:
{action_menu}

Respond with ONLY this JSON object:
{{"action": <integer 0-{max_action}>, "reasoning": "<one or two sentences; mention pacing if relevant>"}}
"""

TEMPLATES = {"default_v2": DEFAULT_USER_TEMPLATE}


@dataclass
class LLMDecision:
    """A fully-auditable record of a single pricing decision."""

    day: int
    mode: str                 # "api" or "heuristic"
    model: str
    template_name: str
    action: int
    reasoning: str
    used_fallback: bool
    system_prompt: str = ""
    prompt: str = ""          # full rendered user prompt
    raw_response: str = ""
    fallback_reason: str = ""
    latency_s: float = 0.0
    usage: dict | None = None  # token usage, if the API returned it

    def to_record(self) -> dict:
        return asdict(self)


class LLMAgent(Agent):
    name = "llm"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        user_template: str | None = None,
        template_name: str = "default_v2",
        api_key: str | None = None,
        max_tokens: int = 300,
        mode: str = "api",
        force_fallback: bool = False,
        request_seed: int | None = 0,
        cache: bool = True,
    ):
        if force_fallback:
            mode = "heuristic"
        if mode not in ("api", "heuristic"):
            raise ValueError(f"mode must be 'api' or 'heuristic', got {mode!r}")

        self.mode = mode
        self.model = model
        self.name = f"llm:{model}"
        self.temperature = float(temperature)
        self.system_prompt = system_prompt
        self.template_name = template_name
        self.user_template = user_template or TEMPLATES.get(template_name, DEFAULT_USER_TEMPLATE)
        self.max_tokens = int(max_tokens)
        self.request_seed = request_seed
        self._use_cache = bool(cache)

        self.log: list[LLMDecision] = []
        self._cache: dict[str, LLMDecision] = {}
        self._client = None

        if self.mode == "api":
            self._client = self._init_client_strict(api_key)

    # -- client setup -------------------------------------------------------
    def _init_client_strict(self, api_key: str | None):
        """Build the OpenAI client or fail loudly with an actionable message."""
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "LLMAgent(mode='api') requires an OpenAI API key but "
                "OPENAI_API_KEY is not set.\n"
                "  - Set it:   export OPENAI_API_KEY=sk-...\n"
                "  - Or run the explicit offline baseline instead: "
                "LLMAgent(mode='heuristic')  (CLI: --llm-mode heuristic).\n"
                "The heuristic mode is a documented rule-of-thumb, NOT the LLM, "
                "and every decision it makes is logged as used_fallback=True."
            )
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "LLMAgent(mode='api') requires the 'openai' package "
                f"(pip install openai). Import failed: {exc}"
            ) from exc
        return OpenAI(api_key=key)

    @property
    def using_llm(self) -> bool:
        return self.mode == "api" and self._client is not None

    # -- prompt rendering ---------------------------------------------------
    def _action_menu(self) -> str:
        lines = []
        for idx, (label, mult) in enumerate(ACTIONS):
            pct = (mult - 1.0) * 100
            desc = ("hold the price" if abs(pct) < 1e-9
                    else f"{'raise' if pct > 0 else 'lower'} price by {abs(pct):.0f}%")
            lines.append(f"  {idx}: {desc}")
        return "\n".join(lines)

    def render_prompt(self, state: MarketState) -> str:
        steps_left = max(0, state.horizon - state.day)
        sell_rate = state.inventory / max(1, steps_left)
        return self.user_template.format(
            own_price=state.own_price,
            unit_cost=state.unit_cost,
            competitor_prices=", ".join(f"{c:.2f}" for c in state.competitor_prices),
            demand_level=state.demand_level,
            inventory=state.inventory,
            day=state.day,
            horizon=state.horizon,
            steps_left=steps_left,
            sell_rate=sell_rate,
            day_of_week=state.day_of_week,
            season=state.season,
            price_min=state.price_min,
            price_max=state.price_max,
            action_menu=self._action_menu(),
            max_action=len(ACTIONS) - 1,
        )

    # -- decision -----------------------------------------------------------
    def _cache_key(self, state: MarketState) -> str:
        return json.dumps([
            round(state.own_price, 2),
            [round(c, 2) for c in state.competitor_prices],
            state.day_of_week, state.season,
            int(state.inventory > 0),
            max(0, state.horizon - state.day),
        ])

    def act(self, state: MarketState) -> int:
        if self._use_cache:
            key = self._cache_key(state)
            if key in self._cache:
                cached = self._cache[key]
                # log a copy so repeated states are still represented in the log
                replay = LLMDecision(**{**cached.to_record(), "day": state.day})
                self.log.append(replay)
                return replay.action

        decision = self._decide(state)
        self.log.append(decision)
        if self._use_cache:
            self._cache[self._cache_key(state)] = decision
        return decision.action

    def _decide(self, state: MarketState) -> LLMDecision:
        if self.mode == "heuristic":
            return self._heuristic(state, reason="explicit heuristic mode")

        prompt = self.render_prompt(state)
        t0 = time.perf_counter()
        try:
            raw, usage = self._call_llm(prompt)
            latency = time.perf_counter() - t0
            action, reasoning = self._parse(raw)
            if action is None:
                d = self._heuristic(state, reason="unparseable LLM output")
                d.raw_response, d.prompt, d.latency_s, d.usage = raw, prompt, latency, usage
                return d
            return LLMDecision(
                day=state.day, mode="api", model=self.model,
                template_name=self.template_name, action=action, reasoning=reasoning,
                used_fallback=False, system_prompt=self.system_prompt, prompt=prompt,
                raw_response=raw, latency_s=latency, usage=usage,
            )
        except Exception as exc:
            d = self._heuristic(state, reason=f"LLM call error: {exc}")
            d.prompt, d.latency_s = prompt, time.perf_counter() - t0
            return d

    def _call_llm(self, prompt: str) -> tuple[str, dict | None]:
        kwargs = dict(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        if self.request_seed is not None:
            kwargs["seed"] = self.request_seed
        resp = self._client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content or ""
        usage = None
        if getattr(resp, "usage", None) is not None:
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            }
        return content, usage

    @staticmethod
    def _parse(content: str) -> tuple[int | None, str]:
        for candidate in (content, _extract_json(content)):
            if not candidate:
                continue
            try:
                obj = json.loads(candidate)
                action = int(obj["action"])
                if 0 <= action < len(ACTIONS):
                    return action, str(obj.get("reasoning", ""))
            except Exception:
                continue
        m = re.search(r"action\D+(\d+)", content)
        if m:
            a = int(m.group(1))
            if 0 <= a < len(ACTIONS):
                return a, content.strip()[:200]
        return None, content.strip()[:200]

    # -- transparent heuristic (explicit, logged) --------------------------
    def _heuristic(self, state: MarketState, reason: str) -> LLMDecision:
        """A simple, explainable margin/volume/pacing rule. Always logged."""
        floor = state.unit_cost * 1.3
        target = max(floor, 0.5 * state.competitor_mean + 0.5 * state.own_price)
        steps_left = max(1, state.horizon - state.day)
        # lean higher when inventory is scarce relative to remaining demand
        if state.inventory < 0.1 * max(state.demand_level, 1) * steps_left:
            target *= 1.05
        action = action_to_reach_price(state, target)
        return LLMDecision(
            day=state.day, mode="heuristic", model="(none)",
            template_name="(heuristic)", action=action,
            reasoning=f"[heuristic: {reason}] target ~ {target:.2f}",
            used_fallback=True, fallback_reason=reason,
        )

    # -- analysis / audit helpers ------------------------------------------
    def reasoning_log(self) -> list[dict]:
        return [
            {"day": d.day, "mode": d.mode, "action": d.action,
             "reasoning": d.reasoning, "used_fallback": d.used_fallback}
            for d in self.log
        ]

    def dump_log(self, path) -> None:
        """Write the full per-call audit log as JSON lines."""
        from pathlib import Path

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as fh:
            for d in self.log:
                fh.write(json.dumps(d.to_record()) + "\n")

    def usage_summary(self) -> dict:
        """Aggregate call counts, fallback rate and token usage for costing."""
        calls = [d for d in self.log if d.mode == "api"]
        fallbacks = [d for d in self.log if d.used_fallback]
        total_tokens = sum((d.usage or {}).get("total_tokens", 0) for d in self.log)
        prompt_tokens = sum((d.usage or {}).get("prompt_tokens", 0) for d in self.log)
        completion_tokens = sum((d.usage or {}).get("completion_tokens", 0) for d in self.log)
        return {
            "decisions": len(self.log),
            "api_calls": len([d for d in calls if not d.used_fallback]),
            "fallbacks": len(fallbacks),
            "fallback_rate": (len(fallbacks) / len(self.log)) if self.log else 0.0,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }


def _extract_json(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return None
