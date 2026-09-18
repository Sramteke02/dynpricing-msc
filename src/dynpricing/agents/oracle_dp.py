"""Exact backward-induction oracle (optional rigour).

The default :class:`~dynpricing.agents.baselines.OracleAgent` is a fluid
(rate-control) heuristic: it is *constructed* to pace inventory well, but it is
not proven optimal. This module provides a second ceiling that is optimal **by
backward induction** rather than by construction, so the dissertation can state
"no agent exceeds the oracle by optimality" and cross-check the fluid oracle.

Scope and assumptions (stated honestly).
The full environment is stochastic (log-normal demand noise and a random-walk
competitor process), so an *exact* DP over the true transition kernel is
intractable. We therefore solve the **deterministic relaxation** of the MDP
exactly:

* demand is replaced by its expectation (no noise);
* the competitor price follows its deterministic expected mean-reversion path
  ``c_t = ref + (c_0 - ref)(1 - reversion)^t``;
* the state ``(day, inventory, current price)`` is discretised — inventory onto
  a uniform grid and price onto the exact lattice of values reachable from the
  start price under the four relative actions.

Backward induction on this relaxation yields the optimal value
``V[0, inv_0, price_0]`` (a deterministic ceiling) and an optimal policy, which
we then execute in the real stochastic environment. Because the relaxation uses
expected demand, the fluid oracle and this DP oracle should agree closely; any
gap quantifies the value of the fluid approximation. See ``verify-oracle``.
"""

from __future__ import annotations

import numpy as np

from dynpricing.agents.base import Agent
from dynpricing.env.config import EnvConfig
from dynpricing.env.demand import DemandModel
from dynpricing.env.market_env import ACTIONS, MarketState


class BackwardInductionOracle(Agent):
    name = "oracle_dp"

    def __init__(self, cfg: EnvConfig, inv_buckets: int = 400, price_decimals: int = 2):
        self.cfg = cfg
        self.demand = DemandModel(cfg)
        self.inv_buckets = int(inv_buckets)
        self.price_decimals = int(price_decimals)
        self._build()
        self._solve()

    def _build(self) -> None:
        cfg = self.cfg
        r = self.price_decimals

        start = round(cfg.init_price, r)
        prices = {start}
        frontier = [start]
        while frontier:
            nxt = []
            for p in frontier:
                for _, m in ACTIONS:
                    q = round(float(np.clip(p * m, cfg.price_min, cfg.price_max)), r)
                    if q not in prices:
                        prices.add(q)
                        nxt.append(q)
            frontier = nxt
        self.prices = np.array(sorted(prices), dtype=float)
        self.n_prices = self.prices.size
        idx_of = {float(round(p, r)): i for i, p in enumerate(self.prices)}

        self.trans = np.zeros((len(ACTIONS), self.n_prices), dtype=int)
        for a, (_, m) in enumerate(ACTIONS):
            for i, p in enumerate(self.prices):
                q = float(round(float(np.clip(p * m, cfg.price_min, cfg.price_max)), r))
                self.trans[a, i] = idx_of[q]

        self.inv_step = cfg.init_inventory / self.inv_buckets
        self.inv_levels = np.linspace(0.0, cfg.init_inventory, self.inv_buckets + 1)

        c0 = float(np.mean(cfg.competitor_init[: cfg.n_competitors]))
        days = np.arange(cfg.horizon)
        self.comp_mean = cfg.ref_price + (c0 - cfg.ref_price) * (1 - cfg.competitor_reversion) ** days

        self.units_table = np.zeros((cfg.horizon, self.n_prices), dtype=float)
        for t in range(cfg.horizon):
            cm = [float(self.comp_mean[t])]
            self.units_table[t] = [
                self.demand.expected_units(p, cm, t) for p in self.prices
            ]

    def _solve(self) -> None:
        cfg = self.cfg
        n_inv = self.inv_levels.size
        cost = cfg.unit_cost
        inv_col = self.inv_levels[:, None]

        V_next = np.zeros((n_inv, self.n_prices), dtype=float)
        self.policy = np.zeros((cfg.horizon, n_inv, self.n_prices), dtype=np.int8)

        for t in range(cfg.horizon - 1, -1, -1):
            best_val = np.full((n_inv, self.n_prices), -np.inf)
            best_act = np.zeros((n_inv, self.n_prices), dtype=np.int8)
            for a in range(len(ACTIONS)):
                q_idx = self.trans[a]
                q_price = self.prices[q_idx]
                units_q = self.units_table[t][q_idx]
                units = np.minimum(units_q[None, :], inv_col)
                reward = (q_price[None, :] - cost) * units
                inv_next = np.clip(inv_col - units, 0.0, cfg.init_inventory)
                inv_idx = np.clip(np.round(inv_next / self.inv_step).astype(int),
                                  0, n_inv - 1)
                future = V_next[inv_idx, q_idx[None, :]]
                total = reward + future
                better = total > best_val
                best_val = np.where(better, total, best_val)
                best_act = np.where(better, a, best_act).astype(np.int8)
            V_next = best_val
            self.policy[t] = best_act

        self.V = V_next
        self._inv_init_idx = int(round(cfg.init_inventory / self.inv_step))
        self._price_init_idx = int(np.argmin(np.abs(self.prices - cfg.init_price)))
        self.optimal_value = float(self.V[self._inv_init_idx, self._price_init_idx])

    def act(self, state: MarketState) -> int:
        t = min(state.day, self.cfg.horizon - 1)
        inv_idx = int(np.clip(round(state.inventory / self.inv_step),
                              0, self.inv_levels.size - 1))
        price_idx = int(np.argmin(np.abs(self.prices - state.own_price)))
        return int(self.policy[t, inv_idx, price_idx])
