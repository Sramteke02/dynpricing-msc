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
    DEFAULT_MODEL, DEFAULT_PROVIDER, PROVIDERS, LLMAgent,
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

    mistral = LLMAgent(mode="api", provider="mistral", client=StubClient([reply]),
                       cache=False)
    mistral.act(_state(cfg))
    sent = mistral._client.chat.completions.calls[0]
    assert "seed" not in sent
    assert sent["model"] == PROVIDERS["mistral"]["default_model"]
    assert sent["response_format"] == {"type": "json_object"}
    assert sent["temperature"] == 0.0

    openai = LLMAgent(mode="api", provider="openai", client=StubClient([reply]),
                      cache=False)
    openai.act(_state(cfg))
    assert openai._client.chat.completions.calls[0]["seed"] == 0


# -- structured output ------------------------------------------------------
def test_structured_json_is_parsed_and_logged_without_fallback(cfg, monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    reply = json.dumps({"action": 2, "reasoning": "undercut; stock is ample"})
    agent = LLMAgent(mode="api", client=StubClient([reply]), cache=False)

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
    agent = LLMAgent(mode="api", client=StubClient([reply]), cache=False)

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
    agent = LLMAgent(mode="api", client=StubClient([bad]), cache=False)

    action = agent.act(_state(cfg))

    entry = agent.log[-1]
    assert 0 <= action < len(ACTIONS)          # still a legal action
    assert entry.used_fallback is True
    assert entry.fallback_reason == "unparseable LLM output"
    assert entry.raw_response == bad           # the raw text is kept for audit


def test_api_error_falls_back_and_records_the_error(cfg, monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    agent = LLMAgent(mode="api", client=StubClient([RuntimeError("429 rate limit")]),
                     cache=False)

    action = agent.act(_state(cfg))

    entry = agent.log[-1]
    assert 0 <= action < len(ACTIONS)
    assert entry.used_fallback is True
    assert "429 rate limit" in entry.fallback_reason


# -- end to end through the real harness ------------------------------------
def test_full_episode_through_the_harness_offline(cfg, monkeypatch):
    """The whole pipeline: harness -> agent -> stub provider -> parse -> action."""
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    replies = [json.dumps({"action": i % len(ACTIONS), "reasoning": f"step {i}"})
               for i in range(cfg.horizon)]
    agent = LLMAgent(mode="api", client=StubClient(replies), cache=False)

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
