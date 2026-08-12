"""LLM-based pricing agent (D1-D3).

The agent renders the market state as a natural-language prompt, asks an LLM to
reason and return a *structured* JSON decision, parses it (with a robust
per-call fallback), and logs everything for qualitative analysis and cost/
reliability auditing.

Two explicit modes (no silent substitution):

* ``mode="api"`` (default) -- the real API path. Requires the *provider's* API
  key (``MISTRAL_API_KEY`` by default, ``OPENAI_API_KEY`` for ``provider=
  "openai"``). If the key (or the client package) is missing the agent **fails
  loudly** at construction with a clear message. This is the mode that tests RQ2.
* ``mode="heuristic"`` -- an explicit, clearly-logged rule-of-thumb used as a
  baseline / offline fallback. Every decision is flagged ``used_fallback=True``.

Providers
---------
The provider is swappable via the ``provider`` argument; see :data:`PROVIDERS`.
Mistral's La Plateforme exposes an **OpenAI-compatible** Chat Completions
endpoint at ``https://api.mistral.ai/v1``, so both providers share one request
path -- only the base URL, the key's environment variable, the model id and a
couple of parameter quirks differ. That keeps the D3 model comparison honest:
the same code, prompt and parser drive every model.

For reproducibility the default model is a **pinned dated snapshot** and the
temperature defaults to **0**. Every call logs the provider, model id, template
name, the full rendered prompt, the raw response, parsed decision, latency and
token usage (see :class:`LLMDecision` and :meth:`dump_log`).

Supporting the desirable requirements:
* **D1** structured decision + reasoning via the Chat Completions API.
* **D2** full per-call logging and a prompt template that is a constructor
  argument (an experimental variable).
* **D3** provider and model id are constructor arguments, so models can be
  compared under identical conditions.

No key is ever read from, or written to, source: keys come from the environment
(or a secrets file the environment is populated from) only.
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

#: Provider registry. Mistral's La Plateforme is OpenAI-compatible, so both
#: entries drive the same ``chat.completions.create`` call; only these fields
#: differ. Add a provider by adding a row -- no other code changes.
#:
#: ``supports_seed``: OpenAI accepts a ``seed`` request parameter for
#: best-effort determinism. It is not part of Mistral's chat-completions
#: schema (Mistral's own SDK calls the equivalent ``random_seed``), so we do
#: not send it there rather than risk a 422 on every call. Determinism on the
#: Mistral path therefore rests on ``temperature=0`` plus the response cache.
PROVIDERS: dict[str, dict] = {
    "mistral": {
        "env_var": "MISTRAL_API_KEY",
        "base_url": "https://api.mistral.ai/v1",
        # Mistral Large, Dec-2025 snapshot — pinned, matching this project's
        # reproducibility convention (the `-latest` alias would silently move
        # under us and break comparability across runs).
        #
        # The id is what GET /v1/models actually serves. The docs' model table
        # renders it as "mistral-large-3-25-12"; that string is rejected by the
        # API. Verified against the live model list, not the docs.
        #
        # Reasoning line: `magistral-small-latest` is served and is the only
        # Magistral left (the medium tier is retired). It is a *small* model, so
        # it is not the default; set provider model explicitly to try it for a
        # chain-of-thought comparison.
        "default_model": "mistral-large-2512",
        "supports_seed": False,
        "docs": "https://docs.mistral.ai/getting-started/models/models_overview/",
    },
    "openai": {
        "env_var": "OPENAI_API_KEY",
        "base_url": None,          # the SDK's own default
        "default_model": "gpt-4o-mini-2024-07-18",
        "supports_seed": True,
        "docs": "https://platform.openai.com/docs/models",
    },
}

#: Default provider. Mistral is the provider this project has access to.
DEFAULT_PROVIDER = "mistral"

#: Pinned, dated model snapshot for reproducibility. Override via the ``model``
#: argument (e.g. a larger model) to support the D3 capability comparison.
DEFAULT_MODEL = PROVIDERS[DEFAULT_PROVIDER]["default_model"]

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


#: HTTP statuses worth retrying: rate limiting and transient server faults.
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class LLMFallbackError(RuntimeError):
    """A decision fell back to the heuristic while in ``strict_llm`` mode.

    Raised so a *contaminated* episode can never be silently reported as an LLM
    result. Rate limiting is handled by the retry loop before this point, so
    reaching here means the failure survived every retry (or the model returned
    output that could not be parsed).
    """


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
    provider: str = ""        # "mistral" / "openai"; "" for the heuristic
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
        model: str | None = None,
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
        provider: str = DEFAULT_PROVIDER,
        client=None,
        max_retries: int = 5,
        backoff_base: float = 1.0,
        backoff_cap: float = 30.0,
        min_call_interval: float = 1.0,
        strict_llm: bool = True,
        sleep=time.sleep,
    ):
        """``provider`` selects the row in :data:`PROVIDERS`; ``model`` defaults
        to that provider's pinned snapshot.

        ``client`` injects a pre-built chat-completions client. It exists so the
        request/parse/fallback pipeline can be exercised **offline, with no key
        and no cost** (see ``tests/test_llm_provider.py``). It is dependency
        injection for tests, not a substitute result: whatever the injected
        client returns is parsed and logged exactly like a real response.

        Rate limiting and result integrity:

        * ``min_call_interval`` paces requests *proactively* — at least this
          many seconds between call starts, so the limit is approached rather
          than hit. 8 back-to-back calls were enough to trip Mistral's limit.
        * ``max_retries`` / ``backoff_base`` / ``backoff_cap`` retry a 429 or a
          transient 5xx with exponential backoff (1s, 2s, 4s … capped),
          honouring a ``Retry-After`` header when the server sends one. Rate
          limits therefore do not become heuristic fallbacks.
        * ``strict_llm`` (default **True**) raises :class:`LLMFallbackError` if
          a decision still falls back after all retries. An episode that mixes
          heuristic and LLM decisions is not a valid RQ2 observation, so it
          fails loudly instead of being quietly reported. Set it False for
          exploratory runs; :meth:`usage_summary` then reports
          ``contaminated`` and the fallback count.
        """
        if force_fallback:
            mode = "heuristic"
        if mode not in ("api", "heuristic"):
            raise ValueError(f"mode must be 'api' or 'heuristic', got {mode!r}")
        if provider not in PROVIDERS:
            raise ValueError(
                f"unknown provider {provider!r}; choose from {sorted(PROVIDERS)}")

        self.provider = provider
        self.provider_config = PROVIDERS[provider]
        self.mode = mode
        self.model = model or self.provider_config["default_model"]
        self.name = f"llm:{self.model}"
        self.temperature = float(temperature)
        self.system_prompt = system_prompt
        self.template_name = template_name
        self.user_template = user_template or TEMPLATES.get(template_name, DEFAULT_USER_TEMPLATE)
        self.max_tokens = int(max_tokens)
        self.request_seed = request_seed
        self._use_cache = bool(cache)

        self.max_retries = int(max_retries)
        self.backoff_base = float(backoff_base)
        self.backoff_cap = float(backoff_cap)
        self.min_call_interval = float(min_call_interval)
        self.strict_llm = bool(strict_llm)
        self._sleep = sleep
        self._last_call_started: float | None = None
        self.n_retries = 0
        self.n_throttle_waits = 0

        self.log: list[LLMDecision] = []
        self._cache: dict[str, LLMDecision] = {}
        self._client = None

        if client is not None:
            self._client = client
        elif self.mode == "api":
            self._client = self._init_client_strict(api_key)

    # -- client setup -------------------------------------------------------
    def _init_client_strict(self, api_key: str | None):
        """Build the provider's client or fail loudly with an actionable message.

        The key is read from the provider's environment variable and is never
        logged, echoed, or persisted.
        """
        env_var = self.provider_config["env_var"]
        base_url = self.provider_config["base_url"]
        key = api_key or os.environ.get(env_var)
        if not key:
            raise RuntimeError(
                f"LLMAgent(mode='api', provider={self.provider!r}) requires an "
                f"API key but {env_var} is not set.\n"
                f"  - Set it:   export {env_var}=...\n"
                "  - Or run the explicit offline baseline instead: "
                "LLMAgent(mode='heuristic')  (CLI: --llm-mode heuristic).\n"
                "The heuristic mode is a documented rule-of-thumb, NOT the LLM, "
                "and every decision it makes is logged as used_fallback=True."
            )
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "LLMAgent(mode='api') uses the 'openai' package as the HTTP "
                f"client (pip install openai). For provider={self.provider!r} it "
                f"talks to {base_url or 'the OpenAI default endpoint'}, which is "
                f"OpenAI-compatible. Import failed: {exc}"
            ) from exc
        kwargs = {"api_key": key}
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAI(**kwargs)

    @property
    def using_llm(self) -> bool:
        return self.mode == "api" and self._client is not None

    @property
    def fallback_count(self) -> int:
        """Decisions that fell back to the heuristic (0 in a clean api run)."""
        return sum(1 for d in self.log if d.used_fallback)

    @property
    def contaminated(self) -> bool:
        """True if an api-mode episode contains any heuristic decision.

        A contaminated episode is not a valid LLM observation: its price path
        is part model, part rule-of-thumb. Check this before recording any RQ2
        result. With ``strict_llm=True`` it can never become True, because the
        first fallback raises instead.
        """
        return self.mode == "api" and self.fallback_count > 0

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
            else:
                return LLMDecision(
                    day=state.day, mode="api", model=self.model,
                    template_name=self.template_name, action=action,
                    reasoning=reasoning, used_fallback=False, provider=self.provider,
                    system_prompt=self.system_prompt, prompt=prompt,
                    raw_response=raw, latency_s=latency, usage=usage,
                )
        except LLMFallbackError:
            raise
        except Exception as exc:
            d = self._heuristic(state, reason=f"LLM call error: {exc}")
            d.prompt, d.latency_s = prompt, time.perf_counter() - t0

        # Single exit for every api-mode fallback. Retries are already spent by
        # here, so this is a genuine contamination of the episode.
        if self.strict_llm:
            raise LLMFallbackError(
                f"LLM decision fell back to the heuristic at day {state.day} "
                f"after {self.max_retries} retries: {d.fallback_reason}\n"
                "This episode would mix heuristic and LLM decisions, which is "
                "not a valid LLM result. Re-run it, or pass strict_llm=False to "
                "accept a contaminated episode (usage_summary() then reports "
                "contaminated=True and the fallback count)."
            )
        return d

    # -- rate limiting ------------------------------------------------------
    @staticmethod
    def _status_of(exc: Exception) -> int | None:
        """HTTP status behind an SDK exception, if there is one."""
        for attr in ("status_code", "code"):
            value = getattr(exc, attr, None)
            if isinstance(value, int):
                return value
        resp = getattr(exc, "response", None)
        status = getattr(resp, "status_code", None)
        if isinstance(status, int):
            return status
        m = re.search(r"\b(4\d{2}|5\d{2})\b", str(exc))
        return int(m.group(1)) if m else None

    def _is_retryable(self, exc: Exception) -> bool:
        status = self._status_of(exc)
        if status in RETRYABLE_STATUS:
            return True
        if status is not None:            # a definite 4xx we cannot fix by waiting
            return False
        # no status at all: connection reset, timeout, DNS blip
        text = str(exc).lower()
        return any(t in text for t in
                   ("timeout", "timed out", "connection", "temporarily", "rate limit"))

    @staticmethod
    def _retry_after(exc: Exception) -> float | None:
        """Server-specified wait, when the response carries Retry-After."""
        resp = getattr(exc, "response", None)
        headers = getattr(resp, "headers", None)
        if not headers:
            return None
        for key in ("retry-after", "Retry-After", "x-ratelimit-reset"):
            try:
                raw = headers.get(key)
            except Exception:
                raw = None
            if raw:
                try:
                    return max(0.0, float(raw))
                except (TypeError, ValueError):
                    continue
        return None

    def _throttle(self) -> None:
        """Proactively space out calls so the limit is approached, not hit."""
        if self.min_call_interval <= 0 or self._last_call_started is None:
            return
        wait = self.min_call_interval - (time.perf_counter() - self._last_call_started)
        if wait > 0:
            self.n_throttle_waits += 1
            self._sleep(wait)

    def _call_llm(self, prompt: str) -> tuple[str, dict | None]:
        """One decision's worth of API traffic, with pacing and retries."""
        for attempt in range(self.max_retries + 1):
            self._throttle()
            self._last_call_started = time.perf_counter()
            try:
                return self._request(prompt)
            except Exception as exc:
                if attempt >= self.max_retries or not self._is_retryable(exc):
                    raise
                delay = self._retry_after(exc)
                if delay is None:
                    delay = min(self.backoff_cap,
                                self.backoff_base * (2 ** attempt))
                self.n_retries += 1
                self._sleep(delay)
        raise RuntimeError("unreachable: retry loop exhausted")  # pragma: no cover

    def _request(self, prompt: str) -> tuple[str, dict | None]:
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
        # `seed` is OpenAI-only; Mistral's schema has no such field and would
        # reject it, so it is omitted rather than sent hopefully.
        if self.request_seed is not None and self.provider_config["supports_seed"]:
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
            "provider": self.provider,
            "model": self.model,
            "decisions": len(self.log),
            "api_calls": len([d for d in calls if not d.used_fallback]),
            "fallbacks": len(fallbacks),
            "fallback_rate": (len(fallbacks) / len(self.log)) if self.log else 0.0,
            "contaminated": self.contaminated,
            "retries": self.n_retries,
            "throttle_waits": self.n_throttle_waits,
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
