"""Uniform-exploration GBM agent (``gbm_uniform``) — a controlled variant of ``gbm``.

This is a **sibling** of :class:`~dynpricing.agents.gbm_agent.GradientBoostingAgent`,
deliberately kept in its own module so the original (pathological) agent stays
runnable for a like-for-like comparison. It reuses the *same* feature map and the
*same* regressor factory by importing them, so the two agents are provably
identical except for the two changes under test:

1. **Exploration.** Training prices are drawn **uniformly across the whole legal
   band** ``[price_min, price_max]`` instead of by taking random moves from the
   action set ``{hold, +5%, -5%, -10%}``. That action set has geometric drift
   -2.66%/step, so the original agent's training data concentrates near the
   bottom of the band (on the calibrated config: ``[0.42, 2.43]`` out of a
   ``[0.84, 5.97]`` band).

2. **Optimiser clamp.** The agent records the min/max price it actually observed
   in training and, at decision time, never selects a price above that
   ``train_max``. This makes extrapolation above the training range structurally
   impossible rather than merely unlikely.

Everything else — the feature vector, the boosting backend and hyperparameters,
the uncensored ``demanded`` target, and the myopic one-step
``argmax_p (p - c) * q_hat(p)`` optimiser over the four reachable prices — is
unchanged.

Why the exploration change needs to touch ``env.price``
------------------------------------------------------
The environment's action space only permits *multiplicative moves* on the
current price, so no sequence of actions can place an i.i.d. uniform price on a
given step. During its own data-collection rollouts (and **only** there) this
agent therefore sets the environment's price attribute directly and then issues
the ``hold`` action, which realises exactly that price. This is the standard
"randomised price experiment" design used to build an offline demand dataset.
Evaluation is completely untouched: :meth:`act` returns ordinary action indices
and the harness drives the environment normally.

The two flags ``uniform_exploration`` and ``clamp_to_train_max`` default to the
configuration described above; they exist so the two changes can be ablated
independently to attribute any recovery to one mechanism or the other.
"""

from __future__ import annotations

import numpy as np

from dynpricing.agents.base import Agent
from dynpricing.agents.gbm_agent import _build_regressor, state_price_features
from dynpricing.env.market_env import ACTIONS, MarketState

#: index of the multiplier-1.0 action; issued after setting an exploration price
HOLD_ACTION = next(i for i, (_, mult) in enumerate(ACTIONS) if mult == 1.0)

#: tolerance when comparing a candidate price against the recorded training max
_CLAMP_TOL = 1e-9


class UniformExplorationGBMAgent(Agent):
    """Predict-then-optimise GBM with band-wide exploration and a train-range clamp."""

    name = "gbm_uniform"
    requires_training = True

    def __init__(self, exploration_episodes: int = 40, seed: int = 0, *,
                 uniform_exploration: bool = True,
                 clamp_to_train_max: bool = True,
                 verbose: bool = False):
        self.exploration_episodes = int(exploration_episodes)
        self.seed = int(seed)
        self.uniform_exploration = bool(uniform_exploration)
        self.clamp_to_train_max = bool(clamp_to_train_max)
        self.verbose = bool(verbose)

        self.model = None
        self.backend = None
        self._trained = False

        # -- training-coverage diagnostics (change 1) ----------------------
        self.train_price_min: float | None = None
        self.train_price_max: float | None = None
        self.band_min: float | None = None
        self.band_max: float | None = None
        self.n_train_rows = 0

        # -- clamp diagnostics (change 2) ----------------------------------
        self.n_decisions = 0
        self.n_clamp_binds = 0          # clamp changed the chosen action
        self.n_candidates_excluded = 0  # candidate prices ruled out by the clamp

    # -- Stage 1: collect interaction data and fit -------------------------
    def train(self, make_env, n_episodes: int | None = None,
              seed: int | None = None) -> "UniformExplorationGBMAgent":
        """Collect interaction data with band-wide exploration, then fit demand."""
        n_episodes = self.exploration_episodes if n_episodes is None else n_episodes
        seed = self.seed if seed is None else seed
        rng = np.random.default_rng(seed)

        X, y, prices = [], [], []
        for _ in range(n_episodes):
            env = make_env()
            _, info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            state = info["state"]
            self.band_min, self.band_max = state.price_min, state.price_max
            done = False
            while not done:
                realised_price, action = self._explore(env, state, rng)
                feats = state_price_features(state, realised_price)
                _, _, terminated, truncated, info = env.step(action)
                # the price the environment actually realised must match the one
                # we featurised, or the training targets are mislabelled
                if abs(float(info["price"]) - realised_price) > 1e-6:
                    raise RuntimeError(
                        "exploration price was not realised by the environment: "
                        f"expected {realised_price:.6f}, got {info['price']:.6f}"
                    )
                # demanded (uncensored by inventory) is the cleanest target
                y.append(float(info["demanded"]))
                X.append(feats)
                prices.append(realised_price)
                state = info["state"]
                done = terminated or truncated

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.train_price_min = float(np.min(prices))
        self.train_price_max = float(np.max(prices))
        self.n_train_rows = int(X.shape[0])

        self.model, self.backend = _build_regressor()
        self.model.fit(X, y)
        self._trained = True

        if self.verbose:
            print(self.coverage_report())
        return self

    def _explore(self, env, state: MarketState, rng) -> tuple[float, int]:
        """Return (realised_price, action) for one exploration step.

        Uniform mode draws a price uniformly across the legal band and writes it
        into the environment before issuing ``hold``; the fallback reproduces the
        original agent's random walk over the action set.
        """
        if not self.uniform_exploration:
            action = int(rng.integers(0, len(ACTIONS)))
            _, mult = ACTIONS[action]
            price = float(np.clip(state.own_price * mult,
                                  state.price_min, state.price_max))
            return price, action

        price = float(rng.uniform(state.price_min, state.price_max))
        if not hasattr(env, "price"):
            raise TypeError(
                "uniform exploration needs to set the environment price directly; "
                f"{type(env).__name__} has no 'price' attribute"
            )
        env.price = price  # realised verbatim by the hold action
        return price, HOLD_ACTION

    # -- Stage 2: predict-then-optimise ------------------------------------
    def _predict_units(self, state: MarketState, price: float) -> float:
        feats = np.asarray([state_price_features(state, price)], dtype=float)
        pred = float(self.model.predict(feats)[0])
        return max(pred, 0.0)

    def act(self, state: MarketState) -> int:
        if not self._trained:
            raise RuntimeError(
                "UniformExplorationGBMAgent.act called before train()")

        limit = (self.train_price_max if self.clamp_to_train_max
                 else float("inf"))
        best_idx, best_profit = None, -float("inf")
        raw_idx, raw_profit = 0, -float("inf")
        cheapest_idx, cheapest_price = 0, float("inf")

        for idx, (_, mult) in enumerate(ACTIONS):
            price = float(np.clip(state.own_price * mult,
                                  state.price_min, state.price_max))
            profit = (price - state.unit_cost) * self._predict_units(state, price)
            if profit > raw_profit:  # what the unclamped optimiser would pick
                raw_idx, raw_profit = idx, profit
            if price < cheapest_price:
                cheapest_idx, cheapest_price = idx, price
            if price > limit + _CLAMP_TOL:
                self.n_candidates_excluded += 1
                continue
            if profit > best_profit:
                best_idx, best_profit = idx, profit

        self.n_decisions += 1
        if best_idx is None:
            # every reachable price sits above the training range: retreat to
            # the lowest reachable price rather than extrapolate.
            self.n_clamp_binds += 1
            return cheapest_idx
        if best_idx != raw_idx:
            self.n_clamp_binds += 1
        return best_idx

    # -- diagnostics -------------------------------------------------------
    def coverage_report(self) -> str:
        """Human-readable summary of what the training data covers."""
        if not self._trained:
            return "gbm_uniform: not trained"
        span = (self.train_price_max - self.train_price_min) / (
            self.band_max - self.band_min)
        spans_band = (
            self.train_price_min <= self.band_min + 0.02 * (self.band_max - self.band_min)
            and self.train_price_max >= self.band_max - 0.02 * (self.band_max - self.band_min)
        )
        return (
            f"gbm_uniform training coverage ({self.n_train_rows} rows, "
            f"backend={self.backend}):\n"
            f"  band           : [{self.band_min:.3f}, {self.band_max:.3f}]\n"
            f"  training prices: [{self.train_price_min:.3f}, "
            f"{self.train_price_max:.3f}]  ({100 * span:.1f}% of the band)\n"
            f"  spans full band: {'YES' if spans_band else 'NO'}"
        )

    def clamp_report(self) -> str:
        """Human-readable summary of whether the train-range clamp ever bound."""
        if not self.clamp_to_train_max:
            return "gbm_uniform: clamp disabled"
        pct = 100 * self.n_clamp_binds / self.n_decisions if self.n_decisions else 0.0
        return (
            f"gbm_uniform clamp (train_max={self.train_price_max:.3f}): "
            f"bound on {self.n_clamp_binds}/{self.n_decisions} decisions "
            f"({pct:.2f}%); {self.n_candidates_excluded} candidate prices excluded"
        )
