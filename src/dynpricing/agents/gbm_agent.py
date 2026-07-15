"""Two-stage gradient-boosting *predict-then-optimise* pricing agent (E4).

Stage 1 (predict): a gradient-boosted regressor learns a demand model from
*interaction data* the agent collects by exploring the market. It never sees the
environment's true demand function.

Stage 2 (optimise): at decision time the agent predicts demand for each
reachable next price and chooses the move that maximises predicted gross profit.

This is the credible industry-standard pattern, not a weak baseline. It uses
XGBoost or LightGBM if installed, falling back to scikit-learn's
``HistGradientBoostingRegressor``.
"""

from __future__ import annotations

import math

import numpy as np

from dynpricing.agents.base import Agent
from dynpricing.env.market_env import ACTIONS, MarketState


def _build_regressor():
    """Return (model, label) using the best available boosting backend."""
    try:
        from xgboost import XGBRegressor

        return (
            XGBRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                n_jobs=0,
                random_state=0,
            ),
            "xgboost",
        )
    except Exception:
        pass
    try:
        from lightgbm import LGBMRegressor

        return (
            LGBMRegressor(
                n_estimators=400,
                max_depth=-1,
                num_leaves=31,
                learning_rate=0.05,
                subsample=0.9,
                random_state=0,
                verbose=-1,
            ),
            "lightgbm",
        )
    except Exception:
        pass
    from sklearn.ensemble import HistGradientBoostingRegressor

    return (
        HistGradientBoostingRegressor(
            max_iter=400, max_depth=None, learning_rate=0.05, random_state=0
        ),
        "sklearn_hgbr",
    )


def state_price_features(state: MarketState, price: float) -> list[float]:
    """Feature vector for predicting demand at a *candidate* price.

    Uses only observable quantities (no true demand parameters).
    """
    comp_mean = state.competitor_mean
    return [
        price,
        price / state.ref_price,
        comp_mean,
        price / max(comp_mean, 1e-9),
        float(state.day),
        float(state.day_of_week),
        float(state.season),
        float(state.inventory),
    ]


class GradientBoostingAgent(Agent):
    name = "gbm"
    requires_training = True

    def __init__(self, exploration_episodes: int = 40, seed: int = 0):
        self.exploration_episodes = int(exploration_episodes)
        self.seed = int(seed)
        self.model = None
        self.backend = None
        self._trained = False

    # -- Stage 1: collect interaction data and fit -------------------------
    def train(self, make_env, n_episodes: int | None = None,
              seed: int | None = None) -> "GradientBoostingAgent":
        """Collect interaction data with exploratory pricing, then fit demand."""
        n_episodes = self.exploration_episodes if n_episodes is None else n_episodes
        seed = self.seed if seed is None else seed
        rng = np.random.default_rng(seed)

        X, y = [], []
        for ep in range(n_episodes):
            env = make_env()
            _, info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            state = info["state"]
            done = False
            while not done:
                action = int(rng.integers(0, len(ACTIONS)))  # explore
                # record the price we are about to realise
                _, mult = ACTIONS[action]
                realised_price = float(np.clip(
                    state.own_price * mult, state.price_min, state.price_max
                ))
                feats = state_price_features(state, realised_price)
                _, _, terminated, truncated, info = env.step(action)
                # demanded (uncensored by inventory) is the cleanest target
                y.append(float(info["demanded"]))
                X.append(feats)
                state = info["state"]
                done = terminated or truncated

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.model, self.backend = _build_regressor()
        self.model.fit(X, y)
        self._trained = True
        return self

    # -- Stage 2: predict-then-optimise ------------------------------------
    def _predict_units(self, state: MarketState, price: float) -> float:
        feats = np.asarray([state_price_features(state, price)], dtype=float)
        pred = float(self.model.predict(feats)[0])
        return max(pred, 0.0)

    def act(self, state: MarketState) -> int:
        if not self._trained:
            raise RuntimeError("GradientBoostingAgent.act called before train()")
        best_idx, best_profit = 0, -float("inf")
        for idx, (_, mult) in enumerate(ACTIONS):
            price = float(np.clip(
                state.own_price * mult, state.price_min, state.price_max
            ))
            units = self._predict_units(state, price)
            profit = (price - state.unit_cost) * units
            if profit > best_profit:
                best_idx, best_profit = idx, profit
        return best_idx
