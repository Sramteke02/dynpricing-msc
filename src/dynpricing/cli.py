"""Command-line entry point (E6).

Subcommands:

* ``calibrate``  build an :class:`EnvConfig` from data under ``data/``.
* ``run``        run agents across scenarios/seeds and write a metrics CSV.
* ``dashboard``  render the results dashboard from a metrics CSV.
* ``demo``       quick smoke run of all agents on the baseline scenario.

Example::

    dynpricing calibrate --out configs/calibrated.json
    dynpricing run --config configs/calibrated.json --agents all --seeds 20 --out results/
    dynpricing dashboard --results results/metrics.csv --out results/dashboard.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dynpricing.env.config import EnvConfig
from dynpricing.env.market_env import MarketEnv
from dynpricing.agents.registry import AGENT_NAMES, build_agent
from dynpricing.eval.harness import (
    Scenario,
    default_scenarios,
    evaluate_agent,
    run_episode,
    run_experiment,
)
from dynpricing.eval.metrics import aggregate


def _load_cfg(path: str | None) -> EnvConfig:
    if path and Path(path).exists():
        return EnvConfig.load(path)
    if path:
        print(f"[warn] config '{path}' not found; using defaults", file=sys.stderr)
    return EnvConfig()


def _resolve_agents(spec: str) -> list[str]:
    if spec.strip().lower() == "all":
        return list(AGENT_NAMES)
    names = [s.strip() for s in spec.split(",") if s.strip()]
    for n in names:
        if n not in AGENT_NAMES:
            raise SystemExit(f"unknown agent {n!r}; choose from {AGENT_NAMES} or 'all'")
    return names


_REQUIRED_DATA = {
    "UCI Online Retail II (CC BY 4.0)":
        "data/online_retail_II.xlsx  (or online_retail_II.csv)",
    "ONS Retail Sales Index (OGL, optional)":
        "data/ons_retail_sales.csv",
    "UK Bank Holidays (OGL, optional)":
        "fetched live with --holidays (no file needed)",
}


def cmd_calibrate(args) -> None:
    from dynpricing.calibration import calibrate, sanity_report

    result = calibrate(args.data_dir, fetch_holidays=args.holidays)

    if not result.sources_used and not args.allow_synthetic:
        print("=" * 70, file=sys.stderr)
        print("BLOCKED: no real calibration data found under "
              f"'{args.data_dir}'.", file=sys.stderr)
        print("I will NOT silently fall back to synthetic defaults.\n", file=sys.stderr)
        print("Please add at least the UCI dataset, then re-run. Expected files:",
              file=sys.stderr)
        for name, loc in _REQUIRED_DATA.items():
            print(f"  - {name}\n      -> {loc}", file=sys.stderr)
        print("\nDownload UCI Online Retail II from:\n"
              "  https://doi.org/10.24432/C5CG6D", file=sys.stderr)
        print("\nTo proceed with documented synthetic defaults instead, re-run "
              "with --allow-synthetic.", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        raise SystemExit(2)

    print(result.summary())
    print()
    report, ok = sanity_report(result.config)
    print(report)
    result.config.save(args.out)
    print(f"\n[ok] wrote calibrated config -> {args.out}")
    if not ok:
        print("[warn] demand-curve sanity check FAILED; inspect parameters above",
              file=sys.stderr)
        raise SystemExit(3)


def cmd_run(args) -> None:
    cfg = _load_cfg(args.config)
    agents = _resolve_agents(args.agents)
    seeds = list(range(args.seeds))
    scenarios = (
        default_scenarios(cfg) if args.scenarios == "all"
        else [Scenario("baseline", cfg)]
    )

    agent_kwargs = {"llm_mode": args.llm_mode}
    if args.llm_model:
        agent_kwargs["llm_model"] = args.llm_model
    if args.llm_provider:
        agent_kwargs["llm_provider"] = args.llm_provider
    if "llm" in agents and args.llm_mode == "api":
        from dynpricing.agents.llm_agent import DEFAULT_PROVIDER, PROVIDERS

        env_var = PROVIDERS[args.llm_provider or DEFAULT_PROVIDER]["env_var"]
        if not os.environ.get(env_var):
            print(f"[warn] LLM agent is in 'api' mode but {env_var} is not set; "
                  "the run will fail loudly when it reaches the LLM agent.\n"
                  "       Use --llm-mode heuristic to run the explicit offline "
                  "baseline.", file=sys.stderr)

    print(f"Running {len(agents)} agents x {len(scenarios)} scenarios x {len(seeds)} seeds")
    results = run_experiment(
        cfg, agents, seeds, scenarios=scenarios, progress=print,
        agent_kwargs=agent_kwargs,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [m.to_row() for m in results]

    import pandas as pd

    df = pd.DataFrame(rows)
    metrics_csv = out_dir / "metrics.csv"
    df.to_csv(metrics_csv, index=False)

    from dynpricing.eval.stats import summarize, paired_comparison, resolve_agent_name

    summ = summarize(rows)
    (out_dir / "metrics_aggregated.json").write_text(json.dumps(summ, indent=2))

    paired = _paired_report(rows, scenarios)
    (out_dir / "paired_comparisons.json").write_text(json.dumps(paired, indent=2))
    agg = summ

    if args.save_paths:
        paths = _collect_price_paths(cfg, agents, agent_kwargs=agent_kwargs)
        (out_dir / "price_paths.json").write_text(json.dumps(paths, indent=2))
        print(f"[ok] wrote price paths -> {out_dir / 'price_paths.json'}")

    print(f"\n[ok] wrote {len(rows)} episode rows -> {metrics_csv}")
    _print_ladder(agg)
    _print_paired(paired)


def _collect_price_paths(cfg: EnvConfig, agents: list[str], seed: int = 0,
                         agent_kwargs: dict | None = None) -> dict:
    paths: dict[str, list[float]] = {}
    base = Scenario("baseline", cfg)
    for name in agents:
        agent = build_agent(name, cfg, seed=seed, **(agent_kwargs or {}))
        if getattr(agent, "requires_training", False):
            agent.train(make_env=lambda: MarketEnv(cfg), seed=seed)
        env = base.make_env()
        m = run_episode(agent, env, seed=seed, scenario="baseline")
        paths[agent.name] = [round(p, 4) for p in m.prices]
    return paths


def cmd_llm_smoke(args) -> None:
    """Cheap reliability/cost check for the LLM agent before any full run."""
    from dynpricing.agents.llm_agent import (
        LLMAgent, DEFAULT_MODEL, DEFAULT_PROVIDER, PROVIDERS,
    )

    cfg = _load_cfg(args.config)
    provider = getattr(args, "provider", None) or DEFAULT_PROVIDER
    env_var = PROVIDERS[provider]["env_var"]
    has_key = bool(os.environ.get(env_var))
    want_api = args.mode == "api" and not args.dry_render

    if want_api and not has_key:
        print("=" * 70)
        print(f"BLOCKED: --mode api requires {env_var} (provider={provider}), "
              "which is not set.")
        print("No real API call will be made and no tokens will be spent.")
        print("Showing a token-free DRY RENDER instead so you can review the")
        print("exact prompts and the JSON parser's reliability now.")
        print(f"To run the real smoke test: export {env_var}=... and re-run.")
        print("=" * 70 + "\n")
        args.dry_render = True

    if args.dry_render:
        _llm_dry_render(cfg, args)
        return

    agent = LLMAgent(model=args.model, mode=args.mode, temperature=args.temperature,
                     provider=getattr(args, "provider", None) or DEFAULT_PROVIDER)
    env = MarketEnv(cfg)
    _, info = env.reset(seed=args.seed)
    state = info["state"]
    print(f"LLM smoke test | mode={agent.mode} model={agent.model} "
          f"temp={agent.temperature} seed={args.seed} steps={args.steps}\n")
    for t in range(args.steps):
        action = agent.act(state)
        _print_decision(t, agent.log[-1])
        _, _, terminated, truncated, info = env.step(action)
        state = info["state"]
        if terminated or truncated:
            break

    print("\n--- usage summary ---")
    print(json.dumps(agent.usage_summary(), indent=2))
    if args.out:
        agent.dump_log(args.out)
        print(f"[ok] wrote full audit log -> {args.out}")
    print("\nReview the prompts/responses/parse results above before launching a "
          "full multi-seed run.")


def _print_decision(step: int, d) -> None:
    print(f"========== step {step} (episode day {d.day}) ==========")
    if d.prompt:
        print("PROMPT (user):")
        print(_indent(d.prompt))
    if d.raw_response:
        print("RAW RESPONSE:")
        print(_indent(d.raw_response))
    print(f"PARSED -> action={d.action}  used_fallback={d.used_fallback}"
          + (f"  reason={d.fallback_reason}" if d.used_fallback else ""))
    print(f"reasoning: {d.reasoning}")
    if d.usage:
        print(f"tokens: {d.usage}  latency={d.latency_s:.2f}s")
    print()


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def _llm_dry_render(cfg, args) -> None:
    """Show the exact prompts that WOULD be sent and self-test the parser.

    This makes NO API call and spends NO tokens. It is explicitly a preview of
    the real path, not a substitute for the LLM.
    """
    from dynpricing.agents.llm_agent import LLMAgent

    agent = LLMAgent(mode="heuristic")
    env = MarketEnv(cfg)
    _, info = env.reset(seed=args.seed)
    state = info["state"]

    print("=== DRY RENDER: prompts that WOULD be sent (no tokens spent) ===")
    print(f"model that would be used : {args.model}")
    print(f"temperature              : {args.temperature}")
    print(f"response_format          : json_object  (forced valid JSON)")
    print("\nSYSTEM PROMPT:")
    print(_indent(agent.system_prompt))

    n_show = min(args.steps, 3)
    print(f"\nShowing the first {n_show} user prompts:\n")
    for t in range(n_show):
        print(f"---------- user prompt, step {t} (day {state.day}) ----------")
        print(_indent(agent.render_prompt(state)))
        print()
        action = agent.act(state)
        _, _, terminated, truncated, info = env.step(action)
        state = info["state"]
        if terminated or truncated:
            break

    sample = agent.render_prompt(info["state"])
    checks = {
        "exposes remaining inventory": "Remaining inventory" in sample,
        "exposes periods remaining": "PERIODS REMAINING" in sample,
        "invites pacing reasoning": "pacing" in sample.lower(),
        "no infeasible unit quota": "units per remaining period" not in sample,
    }
    print("--- prompt content checks (for RQ3) ---")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")

    print("\n--- JSON parser self-test (representative model outputs) ---")
    cases = [
        ('clean json', '{"action": 2, "reasoning": "undercut competitors"}'),
        ('json + prose', 'Sure. {"action": 1, "reasoning": "raise, scarce stock"} done'),
        ('whitespace/newlines', '\n  {"action": 0,\n "reasoning": "hold"}\n'),
        ('wrong type coercible', '{"action": "3", "reasoning": "drop hard"}'),
        ('out of range -> reject', '{"action": 9, "reasoning": "bad id"}'),
        ('no json -> reject', 'I think we should lower the price a bit.'),
    ]
    for label, raw in cases:
        action, reasoning = LLMAgent._parse(raw)
        verdict = "parsed" if action is not None else "rejected (-> per-call fallback)"
        print(f"  {label:24} -> action={action!s:>4}  {verdict}")

    print("\nDRY RENDER complete. No API calls were made.")


def cmd_verify_oracle(args) -> None:
    """Cross-check the fluid oracle against the exact backward-induction oracle."""
    import numpy as np
    from dynpricing.agents.baselines import OracleAgent
    from dynpricing.agents.oracle_dp import BackwardInductionOracle
    from dynpricing.env.demand import DemandModel

    cfg = _load_cfg(args.config)
    seeds = list(range(args.seeds))
    print("Building exact backward-induction oracle (deterministic relaxation)...")
    dp = BackwardInductionOracle(cfg, inv_buckets=args.inv_buckets)
    print(f"  price-lattice size : {dp.n_prices}")
    print(f"  inventory buckets  : {dp.inv_buckets}")
    print(f"  deterministic optimal value V0 = {dp.optimal_value:,.1f}\n")

    fluid_profits, dp_profits, fixed_profits = [], [], []
    for s in seeds:
        env = MarketEnv(cfg)
        fluid_profits.append(run_episode(OracleAgent(DemandModel(cfg)), env, seed=s).gross_profit)
        env = MarketEnv(cfg)
        dp_profits.append(run_episode(dp, env, seed=s).gross_profit)
        env = MarketEnv(cfg)
        fixed_profits.append(run_episode(build_agent("fixed", cfg), env, seed=s).gross_profit)

    fluid_m = float(np.mean(fluid_profits))
    dp_m = float(np.mean(dp_profits))
    fixed_m = float(np.mean(fixed_profits))
    gap = (fluid_m - dp_m) / dp_m * 100 if dp_m else 0.0

    print(f"realised mean gross profit over {len(seeds)} seeds:")
    print(f"  fixed (sanity)              : {fixed_m:12,.1f}")
    print(f"  fluid/pacing oracle         : {fluid_m:12,.1f}")
    print(f"  backward-induction oracle   : {dp_m:12,.1f}")
    print(f"  DP deterministic optimum V0 : {dp.optimal_value:12,.1f}")
    print(f"\n  fluid vs DP gap             : {gap:+.2f}%")

    agree = abs(gap) <= args.tol
    print(f"  agreement (|gap| <= {args.tol:.1f}%) : {'YES' if agree else 'NO'}")
    if dp_m + 1e-6 < fixed_m or fluid_m + 1e-6 < fixed_m:
        print("  [warn] an oracle was beaten by fixed-price — investigate")
    print("\nInterpretation: the fluid oracle is",
          "near-optimal (matches the exact ceiling)." if agree
          else "diverging from the exact ceiling; prefer oracle_dp as the ceiling.")


def cmd_dashboard(args) -> None:
    from dynpricing.dashboard import build_dashboard

    price_paths = None
    pp = Path(args.results).parent / "price_paths.json"
    if pp.exists():
        price_paths = json.loads(pp.read_text())

    out = build_dashboard(
        args.results, args.out, scenario=args.scenario, price_paths=price_paths
    )
    print(f"[ok] wrote dashboard -> {out}")


def cmd_demo(args) -> None:
    from dynpricing.eval.stats import summarize

    cfg = _load_cfg(args.config)
    seeds = list(range(args.seeds))
    base = Scenario("baseline", cfg)
    agent_kwargs = {"llm_mode": "heuristic"}
    rows = []
    for name in AGENT_NAMES:
        ms = evaluate_agent(name, cfg, base, seeds, agent_kwargs=agent_kwargs)
        rows.extend(m.to_row() for m in ms)
    _print_ladder(summarize(rows))
    _print_paired(_paired_report(rows, [base]))


def _paired_report(rows, scenarios) -> list[dict]:
    """Paired (by-seed) comparisons: GBM vs LLM, and each learner vs oracle."""
    from dynpricing.eval.stats import paired_comparison, resolve_agent_name

    out = []
    for scen in scenarios:
        sname = scen.name
        gbm = resolve_agent_name(rows, "gbm", sname)
        llm = resolve_agent_name(rows, "llm", sname)
        oracle = resolve_agent_name(rows, "oracle", sname)
        pairs = []
        if gbm and llm:
            pairs.append((gbm, llm))
        for learner in (gbm, llm):
            if learner and oracle:
                pairs.append((learner, oracle))
        for a, b in pairs:
            res = paired_comparison(rows, a, b, metric="gross_profit", scenario=sname)
            if res is not None:
                out.append(res.to_dict())
    return out


def _print_paired(paired: list[dict]) -> None:
    if not paired:
        return
    print("\n=== Paired comparisons (gross profit, matched by seed, "
          "bootstrap 95% CI) ===")
    for p in paired:
        a = p["agent_a"].split(":")[0]
        b = p["agent_b"].split(":")[0]
        sig = "SIGNIFICANT" if p["significant"] else "n.s. (CI spans 0)"
        win = "" if not p["significant"] else f" -> {p['winner'].split(':')[0]} wins"
        print(f"  [{p['scenario']:>18}] {a} - {b}: "
              f"Δ={p['mean_diff']:+10.1f}  "
              f"95%CI=[{p['ci_low']:+.1f}, {p['ci_high']:+.1f}]  "
              f"n={p['n_pairs']}  {sig}{win}")


def _print_ladder(agg: list[dict]) -> None:
    if not agg:
        return
    print("\n=== Results ladder (mean gross profit, bootstrap 95% CI) ===")
    by_scenario: dict[str, list[dict]] = {}
    for r in agg:
        by_scenario.setdefault(r["scenario"], []).append(r)
    order = {n: i for i, n in enumerate(AGENT_NAMES)}
    for scenario, items in by_scenario.items():
        items.sort(key=lambda r: order.get(r["agent"].split(":")[0], 99))
        oracle = next((r for r in items if r["agent"].startswith("oracle")), None)
        ceiling = oracle["gross_profit_mean"] if oracle else None
        n = items[0].get("n_seeds", "?")
        print(f"\n[{scenario}]  (n={n} seeds)")
        for r in items:
            share = ""
            if ceiling:
                share = f"  ({100 * r['gross_profit_mean'] / ceiling:5.1f}% of oracle)"
            ci = f"[{r['gross_profit_ci_low']:.0f}, {r['gross_profit_ci_high']:.0f}]"
            print(f"  {r['agent']:<18} profit={r['gross_profit_mean']:10.1f} "
                  f"{ci:>22}"
                  f"  share={r['market_share_mean']*100:5.1f}%"
                  f"  stab={r['pricing_stability_mean']:.3f}{share}")


def build_parser() -> argparse.ArgumentParser:
    from dynpricing.agents.llm_agent import (
        DEFAULT_MODEL, DEFAULT_PROVIDER, PROVIDERS as _PROVIDERS,
    )

    p = argparse.ArgumentParser(prog="dynpricing", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("calibrate", help="calibrate the environment from data")
    c.add_argument("--data-dir", default="data")
    c.add_argument("--out", default="configs/calibrated.json")
    c.add_argument("--holidays", action="store_true", help="fetch UK bank holidays")
    c.add_argument("--allow-synthetic", action="store_true",
                   help="proceed with synthetic defaults if no real data is found")
    c.set_defaults(func=cmd_calibrate)

    r = sub.add_parser("run", help="run an experiment")
    r.add_argument("--config", default=None)
    r.add_argument("--agents", default="all")
    r.add_argument("--seeds", type=int, default=30,
                   help="number of paired seeds per agent (more -> tighter CIs)")
    r.add_argument("--scenarios", choices=["baseline", "all"], default="baseline")
    r.add_argument("--out", default="results")
    r.add_argument("--save-paths", action="store_true",
                   help="also save representative price paths for the dashboard")
    r.add_argument("--llm-mode", choices=["api", "heuristic"], default="api",
                   help="LLM agent mode; 'api' fails loudly without the "
                        "provider's API key")
    r.add_argument("--llm-model", default=None, help="override the pinned LLM model id")
    r.add_argument("--llm-provider", default=None, choices=sorted(_PROVIDERS),
                   help="LLM provider (default: mistral). Selects the base URL, "
                        "the API-key env var and the default model.")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("llm-smoke", help="cheap reliability/cost check of the LLM agent")
    s.add_argument("--config", default=None)
    s.add_argument("--steps", type=int, default=8)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--model", default=DEFAULT_MODEL)
    s.add_argument("--provider", default=DEFAULT_PROVIDER, choices=sorted(_PROVIDERS))
    s.add_argument("--temperature", type=float, default=0.0)
    s.add_argument("--mode", choices=["api", "heuristic"], default="api")
    s.add_argument("--dry-render", action="store_true",
                   help="render prompts and self-test the parser without any API call")
    s.add_argument("--out", default=None, help="write the full JSONL audit log here")
    s.set_defaults(func=cmd_llm_smoke)

    d = sub.add_parser("dashboard", help="render the results dashboard")
    d.add_argument("--results", default="results/metrics.csv")
    d.add_argument("--out", default="results/dashboard.png")
    d.add_argument("--scenario", default=None)
    d.set_defaults(func=cmd_dashboard)

    v = sub.add_parser("verify-oracle",
                       help="cross-check fluid oracle vs exact backward-induction oracle")
    v.add_argument("--config", default=None)
    v.add_argument("--seeds", type=int, default=20)
    v.add_argument("--inv-buckets", type=int, default=400)
    v.add_argument("--tol", type=float, default=3.0, help="agreement tolerance (%)")
    v.set_defaults(func=cmd_verify_oracle)

    dm = sub.add_parser("demo", help="quick smoke run of all agents")
    dm.add_argument("--config", default=None)
    dm.add_argument("--seeds", type=int, default=3)
    dm.set_defaults(func=cmd_demo)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
