"""Offline provider tests for the LLM agent — no API key, no network, no cost.

These exercise the full request -> parse -> fallback pipeline against a stub
chat-completions client that records the kwargs it was called with and returns
canned responses. Nothing here contacts Mistral or OpenAI; a network call would
fail loudly rather than silently pass, because the stub is the only client.
"""

from __future__ import annotations

import json

import pytest

from dynpricing.env.config import EnvConfig
from dynpricing.env.market_env import MarketEnv, ACTIONS
from dynpricing.agents.llm_agent import (
    DEFAULT_MODEL, DEFAULT_PROVIDER, PROVIDERS, LLMAgent, LLMFallbackError,
)
from dynpricing.eval.harness import run_episode


# -- the offline stub -------------------------------------------------------
class _Usage:
    prompt_tokens = 321
    completion_tokens = 42
    total_tokens = 363


class _Message:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Message(content)


class _Response:
    def __init__(self, content):
        self.choices = [_Choice(content)]
        self.usage = _Usage()


class StubCompletions:
    """Stands in for ``client.chat.completions``; records calls, returns canned text."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        if isinstance(reply, Exception):
            raise reply
        return _Response(reply)


class StubClient:
    def __init__(self, replies):
        self.chat = type("_Chat", (), {})()
        self.chat.completions = StubCompletions(replies)


class HTTPError(Exception):
    """Mimics an SDK status error (429/5xx) without importing the SDK."""

    def __init__(self, status, message="boom", retry_after=None):
        super().__init__(f"Error code: {status} - {message}")
        self.status_code = status
        if retry_after is not None:
            self.response = type("_R", (), {"headers": {"retry-after": retry_after}})()


def agent_kwargs(**over):
    """Defaults that keep offline tests instant: no pacing, no real sleeping."""
    base = dict(mode="api", cache=False, min_call_interval=0.0,
                backoff_base=0.0, sleep=lambda _s: None)
    base.update(over)
    return base


@pytest.fixture
def cfg():
    return EnvConfig(horizon=6, init_inventory=20000)


def _state(cfg):
    env = MarketEnv(cfg)
    _, info = env.reset(seed=0)
    return info["state"]


# -- provider wiring --------------------------------------------------------
def test_mistral_is_the_default_provider_and_pins_a_dated_model():
    assert DEFAULT_PROVIDER == "mistral"
    assert DEFAULT_MODEL == PROVIDERS["mistral"]["default_model"]
    assert PROVIDERS["mistral"]["env_var"] == "MISTRAL_API_KEY"
    assert PROVIDERS["mistral"]["base_url"] == "https://api.mistral.ai/v1"


def test_missing_key_names_the_right_env_var_per_provider(monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="MISTRAL_API_KEY"):
        LLMAgent(mode="api", provider="mistral")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        LLMAgent(mode="api", provider="openai")


def test_unknown_provider_rejected():
    with pytest.raises(ValueError, match="unknown provider"):
        LLMAgent(mode="heuristic", provider="codestral-by-mistake")


def test_seed_sent_for_openai_but_not_mistral(cfg, monkeypatch):
    """`seed` is OpenAI-only; sending it to Mistral would 422 every call."""
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    reply = json.dumps({"action": 0, "reasoning": "hold"})

    mistral = LLMAgent(provider="mistral", client=StubClient([reply]),
                       **agent_kwargs())
    mistral.act(_state(cfg))
    sent = mistral._client.chat.completions.calls[0]
    assert "seed" not in sent
    assert sent["model"] == PROVIDERS["mistral"]["default_model"]
    assert sent["response_format"] == {"type": "json_object"}
    assert sent["temperature"] == 0.0

    openai = LLMAgent(provider="openai", client=StubClient([reply]),
                      **agent_kwargs())
    openai.act(_state(cfg))
    assert openai._client.chat.completions.calls[0]["seed"] == 0


# -- structured output ------------------------------------------------------
def test_structured_json_is_parsed_and_logged_without_fallback(cfg, monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    reply = json.dumps({"action": 2, "reasoning": "undercut; stock is ample"})
    agent = LLMAgent(client=StubClient([reply]), **agent_kwargs())

    action = agent.act(_state(cfg))

    assert action == 2
    entry = agent.log[-1]
    assert entry.used_fallback is False
    assert entry.provider == "mistral"
    assert entry.model == DEFAULT_MODEL
    assert entry.reasoning == "undercut; stock is ample"
    assert entry.usage["total_tokens"] == 363
    assert "PERIODS REMAINING" in entry.prompt      # the prompt really was rendered


def test_reasoning_model_prose_around_json_still_parses(cfg, monkeypatch):
    """A chain-of-thought model may wrap its JSON in prose; that must still parse."""
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    reply = ("Let me think. Inventory is ample and rivals are cheaper, so I should "
             'undercut.\n{"action": 3, "reasoning": "cut 10% to move stock"}\nDone.')
    agent = LLMAgent(client=StubClient([reply]), **agent_kwargs())

    assert agent.act(_state(cfg)) == 3
    assert agent.log[-1].used_fallback is False


# -- malformed-response fallback -------------------------------------------
@pytest.mark.parametrize("bad", [
    "I think we should lower the price a bit.",     # no JSON at all
    '{"action": 99, "reasoning": "out of range"}',  # out-of-range action
    '{"reasoning": "no action key"}',               # missing key
    "",                                             # empty response
])
def test_malformed_response_falls_back_and_says_why(cfg, monkeypatch, bad):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    agent = LLMAgent(client=StubClient([bad]), **agent_kwargs(strict_llm=False))

    action = agent.act(_state(cfg))

    entry = agent.log[-1]
    assert 0 <= action < len(ACTIONS)          # still a legal action
    assert entry.used_fallback is True
    assert entry.fallback_reason == "unparseable LLM output"
    assert entry.raw_response == bad           # the raw text is kept for audit


def test_non_retryable_error_falls_back_and_records_it(cfg, monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    agent = LLMAgent(client=StubClient([HTTPError(401, "Invalid API Key")]),
                     **agent_kwargs(strict_llm=False))

    action = agent.act(_state(cfg))

    entry = agent.log[-1]
    assert 0 <= action < len(ACTIONS)
    assert entry.used_fallback is True
    assert "401" in entry.fallback_reason


# -- prompt template --------------------------------------------------------
def test_default_template_states_no_infeasible_unit_quota(cfg):
    """v2 asked for inventory/steps_left units per period — unreachable at any
    price — and the model cut to unit cost chasing it. v3 must not restate it."""
    from dynpricing.agents.llm_agent import DEFAULT_TEMPLATE_NAME, TEMPLATES

    agent = LLMAgent(mode="heuristic")
    prompt = agent.render_prompt(_state(cfg))

    assert agent.template_name == DEFAULT_TEMPLATE_NAME == "default_v4"
    assert "units per remaining period" not in prompt
    assert "Even sell-through" not in prompt
    # v4 keeps v3's fix and adds the explicit comparison
    assert "THE COMPETITOR AVERAGE" in prompt
    # the pacing *facts* RQ3 needs are still exposed
    assert "Remaining inventory" in prompt
    assert "PERIODS REMAINING" in prompt
    assert "pacing" in prompt.lower()
    # and v2 stays available so the pair can be compared
    assert "units per remaining period" in TEMPLATES["default_v2"]


def test_v4_states_the_competitor_comparison_explicitly():
    """v3 left the comparison to the model, which read 1.97 as 'lower' than 1.00."""
    from dynpricing.env.market_env import MarketState

    # the exact state mis-read on baseline seed 1, day 250
    state = MarketState(own_price=1.00, competitor_prices=(2.01, 1.97),
                        unit_cost=0.84, demand_level=54, inventory=29446,
                        day_of_week=5, day=250, season=2, horizon=365,
                        price_min=0.84, price_max=5.972, ref_price=2.1)

    v3 = LLMAgent(mode="heuristic", template_name="default_v3").render_prompt(state)
    v4 = LLMAgent(mode="heuristic", template_name="default_v4").render_prompt(state)

    assert "BELOW" not in v3                      # v3 leaves it to be inferred
    assert "Competitor average: 1.99" in v4
    assert "-50% (BELOW) THE COMPETITOR AVERAGE" in v4
    # v4 is otherwise v3: same length bar the one added line
    assert len(v4.splitlines()) == len(v3.splitlines()) + 1


@pytest.mark.parametrize("own,comps,word", [
    (1.00, (2.01, 1.97), "BELOW"),
    (3.00, (2.01, 1.97), "ABOVE"),
    (1.99, (2.01, 1.97), "LEVEL WITH"),
])
def test_v4_gap_direction(own, comps, word):
    from dynpricing.env.market_env import MarketState

    state = MarketState(own_price=own, competitor_prices=comps, unit_cost=0.84,
                        demand_level=50, inventory=1000, day_of_week=0, day=1,
                        season=0, horizon=365, price_min=0.84, price_max=5.972,
                        ref_price=2.1)
    prompt = LLMAgent(mode="heuristic",
                      template_name="default_v4").render_prompt(state)
    assert f"({word}) THE COMPETITOR AVERAGE" in prompt


def test_v5_states_the_season_without_leaking_the_optimum(cfg):
    """v5 = v4 + one line naming the seasonal demand state and its direction."""
    from dynpricing.agents.llm_agent import SEASON_STATES

    state = _state(cfg)
    v4 = LLMAgent(mode="heuristic", template_name="default_v4").render_prompt(state)
    v5 = LLMAgent(mode="heuristic", template_name="default_v5").render_prompt(state)

    assert len(v5.splitlines()) == len(v4.splitlines()) + 1
    assert "SEASONAL DEMAND is currently" in v5
    assert SEASON_STATES[int(state.season)] in v5
    # it may state the demand state, never the answer
    for banned in ("optimal price", "p*", "should charge", "set price to",
                   "unit_cost", "a0", "elasticity"):
        assert banned not in v5.lower()


def test_v5_season_wording_matches_the_true_seasonal_factor():
    """Each quarter's wording must agree with the environment's own S(t)."""
    from dynpricing.env.demand import DemandModel
    from dynpricing.agents.llm_agent import SEASON_STATES
    from dynpricing.eval.harness import default_scenarios

    cfg = EnvConfig.load("configs/calibrated.json")
    sc = {s.name: s for s in default_scenarios(cfg)}["strong_seasonality"].config
    demand = DemandModel(sc)
    for quarter, day in {0: 20, 1: 115, 2: 210, 3: 300}.items():
        s_now, s_next = demand.seasonal_factor(day), demand.seasonal_factor(day + 5)
        wording = SEASON_STATES[quarter]
        assert ("ABOVE" in wording) == (s_now > 1.0)
        assert ("RISING" in wording) == (s_next > s_now)


def test_v2_template_still_selectable_for_comparison(cfg):
    agent = LLMAgent(mode="heuristic", template_name="default_v2")
    assert "units per remaining period" in agent.render_prompt(_state(cfg))


# -- rate limiting: retry with backoff --------------------------------------
def test_429_is_retried_with_exponential_backoff_then_succeeds(cfg, monkeypatch):
    """A rate limit must cost a wait, never a heuristic fallback."""
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    slept = []
    good = json.dumps({"action": 1, "reasoning": "ok"})
    client = StubClient([HTTPError(429, "Rate limit exceeded"),
                         HTTPError(429, "Rate limit exceeded"),
                         good])
    agent = LLMAgent(client=client, **agent_kwargs(
        backoff_base=1.0, sleep=slept.append))

    action = agent.act(_state(cfg))

    assert action == 1
    assert agent.log[-1].used_fallback is False        # no contamination
    assert agent.contaminated is False
    assert agent.n_retries == 2
    assert slept == [1.0, 2.0]                          # 1s then 2s
    assert len(client.chat.completions.calls) == 3


def test_backoff_is_capped(cfg, monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    slept = []
    agent = LLMAgent(client=StubClient([HTTPError(429)]),
                     **agent_kwargs(max_retries=6, backoff_base=1.0,
                                    backoff_cap=4.0, strict_llm=False,
                                    sleep=slept.append))
    agent.act(_state(cfg))
    assert slept == [1.0, 2.0, 4.0, 4.0, 4.0, 4.0]      # capped at 4s


def test_retry_after_header_is_honoured(cfg, monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    slept = []
    good = json.dumps({"action": 0, "reasoning": "ok"})
    agent = LLMAgent(
        client=StubClient([HTTPError(429, retry_after="7"), good]),
        **agent_kwargs(backoff_base=1.0, sleep=slept.append))
    agent.act(_state(cfg))
    assert slept == [7.0]              # server's number wins over our backoff


def test_5xx_retried_but_4xx_is_not(cfg, monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    good = json.dumps({"action": 0, "reasoning": "ok"})

    ok = LLMAgent(client=StubClient([HTTPError(503), good]), **agent_kwargs())
    ok.act(_state(cfg))
    assert ok.n_retries == 1

    # 401/400 cannot be fixed by waiting: fail immediately, do not burn retries
    bad = LLMAgent(client=StubClient([HTTPError(401, "Invalid API Key")]),
                   **agent_kwargs(strict_llm=False))
    bad.act(_state(cfg))
    assert bad.n_retries == 0
    assert len(bad._client.chat.completions.calls) == 1


def test_calls_are_paced_proactively(cfg, monkeypatch):
    """Spacing is applied between calls, not only after a 429."""
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    slept = []
    good = json.dumps({"action": 0, "reasoning": "ok"})
    agent = LLMAgent(client=StubClient([good]),
                     **agent_kwargs(min_call_interval=1.5, sleep=slept.append))
    for _ in range(3):
        agent.act(_state(cfg))

    assert agent.n_throttle_waits == 2       # first call is free, then paced
    assert all(0 < s <= 1.5 for s in slept)
    assert agent.n_retries == 0


# -- contamination guard ----------------------------------------------------
def test_strict_mode_raises_rather_than_contaminating_the_episode(cfg, monkeypatch):
    """CRITICAL: a post-retry fallback must never be silently accepted."""
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    agent = LLMAgent(client=StubClient([HTTPError(429)]),
                     **agent_kwargs(max_retries=2))

    with pytest.raises(LLMFallbackError, match="mix heuristic and LLM"):
        agent.act(_state(cfg))
    assert agent.n_retries == 2               # it really did retry first


def test_strict_mode_is_the_default(cfg, monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    agent = LLMAgent(client=StubClient(["not json"]), mode="api", cache=False,
                     min_call_interval=0.0, sleep=lambda _s: None)
    assert agent.strict_llm is True
    with pytest.raises(LLMFallbackError):
        agent.act(_state(cfg))


def test_contamination_is_reported_when_strictness_is_relaxed(cfg, monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    agent = LLMAgent(client=StubClient(["not json"]),
                     **agent_kwargs(strict_llm=False))
    agent.act(_state(cfg))

    assert agent.contaminated is True
    assert agent.fallback_count == 1
    summary = agent.usage_summary()
    assert summary["contaminated"] is True
    assert summary["fallbacks"] == 1


def test_heuristic_mode_is_not_contamination(cfg):
    """Deliberate offline baseline != a contaminated LLM episode."""
    agent = LLMAgent(mode="heuristic")
    agent.act(_state(cfg))
    assert agent.fallback_count == 1
    assert agent.contaminated is False
    assert agent.usage_summary()["contaminated"] is False


# -- end to end through the real harness ------------------------------------
def test_full_episode_through_the_harness_offline(cfg, monkeypatch):
    """The whole pipeline: harness -> agent -> stub provider -> parse -> action."""
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    replies = [json.dumps({"action": i % len(ACTIONS), "reasoning": f"step {i}"})
               for i in range(cfg.horizon)]
    agent = LLMAgent(client=StubClient(replies), **agent_kwargs())

    metrics = run_episode(agent, MarketEnv(cfg), seed=0)

    assert metrics.n_steps == cfg.horizon
    assert len(agent.log) == cfg.horizon
    assert all(not d.used_fallback for d in agent.log)
    summary = agent.usage_summary()
    assert summary["provider"] == "mistral"
    assert summary["model"] == DEFAULT_MODEL
    assert summary["api_calls"] == cfg.horizon
    assert summary["fallback_rate"] == 0.0
    assert summary["total_tokens"] == 363 * cfg.horizon
