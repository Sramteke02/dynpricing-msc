"""Run ONE LLM episode for RQ2, paced to the account's live rate limit.

Strict mode is on: if any decision falls back to the heuristic after retries the
episode aborts and writes no result, so a contaminated episode can never enter
the RQ2 sample.

Pacing is derived from ``x-ratelimit-limit-req-minute`` read off a real response
rather than guessed — on a 4 req/min free tier that is ~16s between calls, so a
365-day episode takes ~100 minutes. Progress is checkpointed every 10 days.

    export MISTRAL_API_KEY=...        # or: set -a; . ./.env; set +a
    python scripts/rq2_llm_episode.py --seed 1

Results land in ``results/rq2_llm/seed<N>.json`` with the full price path, and
the per-decision audit log alongside it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dynpricing.env.config import EnvConfig
from dynpricing.env.market_env import MarketEnv
from dynpricing.agents.base import Agent
from dynpricing.agents.llm_agent import (LLMAgent, LLMFallbackError,
                                         RPM_SAFETY, live_rpm_limit)
from dynpricing.eval.harness import Scenario, default_scenarios, run_episode

IN_PRICE, OUT_PRICE = 0.50, 1.50
ORACLE_CACHE = ROOT / "results" / "seasonal_sweep" / "oracle_cache.json"


def config_key(cfg: EnvConfig) -> str:
    blob = json.dumps(cfg.to_dict(), sort_keys=True, default=list).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


METRICS_CSV = ROOT / "results" / "gbm_uniform" / "metrics.csv"


def oracle_profit(cfg: EnvConfig, seed: int, scenario: str) -> float | None:
    """The committed oracle for this exact (config, seed) — never recomputed.

    Two sources, in order:

    1. ``oracle_cache.json`` — keyed by config fingerprint, written by the
       amplitude sweep. Only covers baseline-with-varied-amplitude configs.
    2. ``results/gbm_uniform/metrics.csv`` — the committed 5-agent run, which
       has oracle rows for all four scenarios x 30 seeds. This is what covers
       the non-baseline scenarios: e.g. strong_seasonality also changes
       holiday_days, so its fingerprint is not in the sweep cache.

    Using the committed oracle rather than recomputing keeps the LLM scored
    against exactly the ceiling the other agents were scored against.
    """
    if ORACLE_CACHE.exists():
        cache = json.loads(ORACLE_CACHE.read_text())
        entry = cache.get(f"oracle:{config_key(cfg)}:{seed}")
        if entry:
            return entry["gross_profit"]
    if METRICS_CSV.exists():
        import csv
        with METRICS_CSV.open() as fh:
            for row in csv.DictReader(fh):
                if (row["agent"] == "oracle" and row["scenario"] == scenario
                        and int(row["seed"]) == seed):
                    return float(row["gross_profit"])
    return None


class ProgressAgent(Agent):
    """Delegates to the LLM agent, printing progress and checkpointing."""

    def __init__(self, inner: LLMAgent, total: int, started: float, out: Path):
        self.inner, self.total, self.started, self.out = inner, total, started, out
        self.name = inner.name
        self.prices: list[float] = []

    def reset(self, state):
        self.inner.reset(state)

    def act(self, state):
        return self.inner.act(state)

    def observe(self, state, action, reward, next_state, info):
        self.inner.observe(state, action, reward, next_state, info)
        self.prices.append(info["price"])
        n = len(self.prices)
        if n % 10 == 0 or n == self.total:
            el = time.time() - self.started
            eta = (self.total - n) * (el / n) / 60
            print(f"  day {n}/{self.total}  price={info['price']:.3f}  "
                  f"elapsed={el/60:.1f}min  eta={eta:.1f}min  "
                  f"retries={self.inner.n_retries}", flush=True)
            self.out.write_text(json.dumps(
                {"days_done": n, "prices": [round(p, 4) for p in self.prices],
                 "usage": self.inner.usage_summary()}, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--scenario", default="baseline")
    ap.add_argument("--template", default=None, help="prompt template name")
    ap.add_argument("--config", default=str(ROOT / "configs" / "calibrated.json"))
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "rq2_llm"))
    args = ap.parse_args()

    cfg = EnvConfig.load(args.config)
    scenario = {s.name: s for s in default_scenarios(cfg)}[args.scenario]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"seed{args.seed}" + ("" if args.scenario == "baseline" else f"_{args.scenario}")

    kwargs = {"mode": "api", "provider": "mistral"}
    if args.template:
        kwargs["template_name"] = args.template
    limit, hdrs, _ = live_rpm_limit(
        on_wait=lambda w, i, n: print(
            f"  rate-limit probe got 429; waiting {w:.0f}s "
            f"(attempt {i}/{n})", flush=True))
    pacing = 60.0 / limit * RPM_SAFETY
    print(f"rate-limit headers: {hdrs}", flush=True)
    print(f"limit {limit} req/min -> pacing {pacing:.2f}s ({60/pacing:.2f} req/min)",
          flush=True)
    print(f"seed={args.seed} scenario={args.scenario} "
          f"projected {cfg.horizon * pacing / 60:.0f} min\n", flush=True)

    agent = LLMAgent(min_call_interval=pacing, max_retries=8, backoff_base=2.0,
                     backoff_cap=90.0, strict_llm=True, **kwargs)
    print(f"model={agent.model} template={agent.template_name} "
          f"strict={agent.strict_llm}\n", flush=True)

    started = time.time()
    wrapper = ProgressAgent(agent, cfg.horizon, started,
                            out_dir / f"{stem}_partial.json")
    failed, metrics = None, None
    try:
        metrics = run_episode(wrapper, scenario.make_env(), seed=args.seed,
                              scenario=args.scenario)
    except LLMFallbackError as exc:
        failed = exc
        print("\n*** STRICT MODE ABORTED ***\n", exc, flush=True)
    wall = time.time() - started

    summary = agent.usage_summary()
    cost = (summary["prompt_tokens"] / 1e6 * IN_PRICE
            + summary["completion_tokens"] / 1e6 * OUT_PRICE)
    result = {
        "seed": args.seed, "scenario": args.scenario,
        "model": agent.model, "template": agent.template_name,
        "aborted": failed is not None,
        "abort_message": str(failed) if failed else None,
        "abort_day": agent.log[-1].day if failed and agent.log else None,
        "rate_limit_rpm": limit, "pacing_s": pacing,
        "wall_clock_s": wall, "usage": summary, "cost_usd": cost,
    }
    if metrics is not None:
        orc = oracle_profit(scenario.config, args.seed, args.scenario)
        result.update({
            "gross_profit": metrics.gross_profit, "revenue": metrics.revenue,
            "market_share": metrics.market_share,
            "pricing_stability": metrics.pricing_stability,
            "n_steps": metrics.n_steps,
            "prices": [round(p, 4) for p in metrics.prices],
            "oracle_gross_profit": orc,
            "pct_of_oracle": (100 * metrics.gross_profit / orc) if orc else None,
        })
    (out_dir / f"{stem}.json").write_text(json.dumps(result, indent=2))
    agent.dump_log(out_dir / f"{stem}_log.jsonl")

    print(f"\nwall {wall/60:.1f} min | decisions {len(agent.log)} "
          f"api_calls={summary['api_calls']} fallbacks={summary['fallbacks']} "
          f"contaminated={summary['contaminated']} retries={summary['retries']}")
    print(f"tokens {summary['total_tokens']:,}  cost ${cost:.4f}")
    if metrics is not None and result["pct_of_oracle"] is not None:
        print(f"gross_profit {metrics.gross_profit:,.1f} "
              f"({result['pct_of_oracle']:.2f}% of oracle)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
