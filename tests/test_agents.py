"""Tests for the decision agents (E3, E4, D1)."""

import pytest

from dynpricing.env.config import EnvConfig
from dynpricing.env.market_env import MarketEnv, ACTIONS
from dynpricing.agents.registry import AGENT_NAMES, build_agent
from dynpricing.agents.llm_agent import LLMAgent
from dynpricing.eval.harness import run_episode, Scenario


@pytest.fixture
def cfg():
    return EnvConfig(horizon=40, init_inventory=20000)


@pytest.mark.parametrize("name", AGENT_NAMES)
def test_agent_obeys_interface(name, cfg):
    agent = build_agent(name, cfg, seed=0, force_fallback=True)
    if getattr(agent, "requires_training", False):
        agent.train(make_env=lambda: MarketEnv(cfg), n_episodes=5, seed=0)
    env = MarketEnv(cfg)
    _, info = env.reset(seed=0)
    action = agent.act(info["state"])
    assert isinstance(action, int) and 0 <= action < len(ACTIONS)


def test_fixed_agent_holds_price(cfg):
    agent = build_agent("fixed", cfg)
    env = MarketEnv(cfg)
    m = run_episode(agent, env, seed=0)
    # holding from a constant start => flat price path => zero stability metric
    assert m.pricing_stability == pytest.approx(0.0, abs=1e-9)


def test_oracle_is_best(cfg):
    """The oracle should not be beaten on profit (correctness check)."""
    seeds = range(8)
    scen = Scenario("baseline", cfg)
    from dynpricing.eval.harness import evaluate_agent
    import numpy as np

    def mean_profit(name, **kw):
        ms = evaluate_agent(name, cfg, scen, seeds, agent_kwargs=kw)
        return float(np.mean([m.gross_profit for m in ms]))

    oracle = mean_profit("oracle")
    for name in ["fixed", "cost_plus", "competitor_match", "random"]:
        assert mean_profit(name) <= oracle + 1e-6


@pytest.mark.xfail(
    reason="Tree extrapolation: under linear demand the GBM's downward-biased "
    "exploration (drift -2.66%/step) concentrates training data at low prices; "
    "boosted trees then extrapolate demand FLAT above that range, so the myopic "
    "optimiser ratchets price past the choke (true demand=0) up to price_max. A "
    "real finding, not a flaky assertion. See docs/gbm_extrapolation_diagnosis.md.",
    strict=False,
)
def test_gbm_trains_and_beats_random(cfg):
    import numpy as np
    from dynpricing.eval.harness import evaluate_agent

    scen = Scenario("baseline", cfg)
    seeds = range(5)
    gbm = np.mean([m.gross_profit for m in
                   evaluate_agent("gbm", cfg, scen, seeds,
                                  agent_kwargs={"exploration_episodes": 15})])
    rnd = np.mean([m.gross_profit for m in evaluate_agent("random", cfg, scen, seeds)])
    assert gbm > rnd  # a learned demand model should beat random pricing


def test_gbm_uniform_training_spans_the_price_band(cfg):
    """Change 1: exploration must cover the whole band, not just its lower end."""
    agent = build_agent("gbm_uniform", cfg, seed=0)
    agent.train(make_env=lambda: MarketEnv(cfg), n_episodes=5, seed=0)
    span = cfg.price_max - cfg.price_min
    assert agent.train_price_min <= cfg.price_min + 0.05 * span
    assert agent.train_price_max >= cfg.price_max - 0.05 * span


def test_gbm_uniform_never_prices_above_its_training_range(cfg):
    """Change 2: the clamp must hold even when training under-covers the band.

    Trained with the *original* agent's drifting exploration, the training data
    stops well short of price_max; the optimiser must then never select a price
    above what it saw.
    """
    from dynpricing.agents.gbm_uniform_agent import UniformExplorationGBMAgent
    from dynpricing.env.market_env import ACTIONS

    agent = UniformExplorationGBMAgent(seed=0, uniform_exploration=False)
    agent.train(make_env=lambda: MarketEnv(cfg), n_episodes=5, seed=0)
    assert agent.train_price_max < cfg.price_max  # otherwise the test is vacuous

    env = MarketEnv(cfg)
    m = run_episode(agent, env, seed=0)
    assert max(m.prices[1:]) <= agent.train_price_max + 1e-9
    assert agent.n_clamp_binds > 0  # the clamp actually bound


def test_gbm_uniform_beats_random(cfg):
    """The counterpart to the xfail above.

    Same features, same regressor, same myopic optimiser as `gbm` — only the
    exploration and the clamp differ. If removing extrapolation past the
    training range is what fixes the pathology, this must pass where the `gbm`
    version xfails. See docs/gbm_extrapolation_diagnosis.md.
    """
    import numpy as np
    from dynpricing.eval.harness import evaluate_agent

    scen = Scenario("baseline", cfg)
    seeds = range(5)
    gbm_u = np.mean([m.gross_profit for m in
                     evaluate_agent("gbm_uniform", cfg, scen, seeds,
                                    agent_kwargs={"exploration_episodes": 15})])
    rnd = np.mean([m.gross_profit for m in evaluate_agent("random", cfg, scen, seeds)])
    assert gbm_u > rnd


def test_llm_fallback_runs_without_key(cfg):
    agent = LLMAgent(force_fallback=True)
    assert not agent.using_llm
    env = MarketEnv(cfg)
    m = run_episode(agent, env, seed=0)
    assert m.n_steps > 0
    log = agent.reasoning_log()
    assert log and all(entry["used_fallback"] for entry in log)


def test_llm_parse_structured_output():
    action, reasoning = LLMAgent._parse('{"action": 2, "reasoning": "undercut"}')
    assert action == 2 and reasoning == "undercut"
    # tolerate surrounding prose
    action, _ = LLMAgent._parse('Sure! {"action": 1, "reasoning": "raise"} done')
    assert action == 1
    # invalid -> None so the caller can fall back
    action, _ = LLMAgent._parse("no json here")
    assert action is None
