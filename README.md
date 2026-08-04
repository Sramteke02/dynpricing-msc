# Data-Driven Dynamic Pricing in a Competitive E-Commerce Simulation

Comparing gradient-boosted and large-language-model pricing agents in a
custom, data-calibrated, Gymnasium-compatible market simulation.

This repository implements the MSc project described in
`proposal_complete.pdf`. It provides:

* **Layer 1 — Data & calibration.** Offline calibration of the demand model
  from open UK retail datasets (UCI Online Retail II, etc.), with a robust
  synthetic fallback so the pipeline runs without any downloads.
* **Layer 2 — Decision agents.** Five rule-based baselines (fixed-price,
  cost-plus, competitor-matching, random, oracle), a two-stage gradient-boosting
  *predict-then-optimise* agent, and an LLM pricing agent.
* **Layer 3 — Shared interface.** Every agent obeys the same contract:
  `act(state) -> price action`.
* **Layer 4 — Simulation core.** A Gymnasium `Env` modelling pricing as a
  Markov Decision Process, with a price-sensitive demand model.
* **Layer 5 — Evaluation & output.** A shared runner over many seeds/scenarios,
  a metrics store, and a results dashboard (Matplotlib; optional W&B).

## Quick start

```bash
python3.11 -m venv .venv && source .venv/bin/activate   # Python >=3.11 required
pip install -e .

# 1. Calibrate from real data under data/ (REFUSES to use synthetic silently;
#    pass --allow-synthetic to fall back to documented defaults). Prints a
#    demand-curve sanity report.
dynpricing calibrate --out configs/calibrated.json
dynpricing calibrate --allow-synthetic --out configs/calibrated.json   # no data yet

# 2. Cheap LLM reliability/cost check BEFORE any full run (needs OPENAI_API_KEY;
#    without a key it does a token-free dry render of the prompts + parser).
dynpricing llm-smoke --steps 8

# 3. Run all agents across paired seeds; reports means with bootstrap 95% CIs
#    and paired (by-seed) comparisons. LLM is real by default (--llm-mode api),
#    fails loudly without a key; use --llm-mode heuristic for the offline baseline.
dynpricing run --config configs/calibrated.json --agents all --seeds 30 \
    --scenarios all --out results/ --save-paths

# 4. Build the results dashboard (CI error bars + GBM-vs-LLM significance)
dynpricing dashboard --results results/metrics.csv --out results/dashboard.png --scenario baseline

# 5. (optional) Cross-check the fluid oracle against an exact backward-induction
#    oracle, so the ceiling holds by optimality, not just by construction.
dynpricing verify-oracle --config configs/calibrated.json --seeds 20
```

### Regenerating the interactive dashboard

`results/dashboard/index.html` is a single self-contained page (no server, no
network, no external assets — open the file directly in a browser). It is
**generated from the committed results files**, so it never holds numbers of its
own; a panel with no underlying run renders an explicit "not yet run" state.

```bash
# after ANY new run, regenerate so the page matches what is on disk:
python scripts/build_dashboard_html.py
```

It reads `results/metrics_aggregated.json`,
`results/gbm_uniform/{metrics_aggregated,paired_gbm_uniform,price_paths}.json`,
`results/gbm_uniform/diagnostics.log` and
`results/seasonal_sweep/summary.json`. The script prints which panels it filled
and which came out empty, so a stale or missing input is visible at build time.
Two panels are currently empty by design: the oracle-verification KPI (run
`dynpricing verify-oracle` and record its output under `results/`) and the RQ2
LLM tile (needs `OPENAI_API_KEY`).

The page also carries a **run-provenance strip**: the config fingerprint,
horizon, price band, inventory, seed range, and the result of a build-time check
that the two committed runs it merges really are the same experiment (identical
scenarios, identical seeds, and identical numbers on every shared episode row).
If that check ever fails, the merge is refused and the page says so instead of
mixing two experiments on one axis.

The previous static Matplotlib figure has been moved to
`results/archive/dashboard-2026-07-16-isoelastic-SUPERSEDED.png` — it predates
the linear-demand model, the 365-day horizon, the reconciled band and
`gbm_uniform`, so it must not be presented. See `results/archive/README.md`.
Regenerating a *fresh* PNG from current results with `dynpricing dashboard`
is still fine.

### Reproducibility & rigour notes

* The LLM agent pins a dated model snapshot, uses `temperature=0`, forces JSON
  output, sets a request seed, and logs the full prompt/response/parse for every
  call (`--out` writes a JSONL audit log; `LLMAgent.usage_summary()` aggregates
  token cost and fallback rate).
* Calibration figures are **plausible simulation parameters**, not identified
  causal elasticities — price and quantity are jointly determined, so no
  identification is claimed (see `calibration/calibrate.py`).
* All agents share the same seed set, so comparisons are **paired by seed**;
  results are reported as means with bootstrap 95% CIs.

Run the tests with:

```bash
pip install -e ".[dev]"
pytest -q
```

## Mapping to the proposal requirements

| Req | Where |
|-----|-------|
| E1 Gymnasium env w/ demand, inventory, competitors, seasonality | `src/dynpricing/env/` |
| E2 Offline calibration from real data | `src/dynpricing/calibration/` |
| E3 Five rule-based baselines | `src/dynpricing/agents/baselines.py` |
| E4 Two-stage gradient-boosting agent | `src/dynpricing/agents/gbm_agent.py` |
| E5 Shared evaluation harness | `src/dynpricing/eval/` |
| E6 CLI + dashboard | `src/dynpricing/cli.py`, `src/dynpricing/dashboard/` |
| D1–D3 LLM agent + reasoning logs + model comparison | `src/dynpricing/agents/llm_agent.py` |

## Optional dependencies

* `xgboost` or `lightgbm` — used by the gradient-boosting agent if available,
  otherwise scikit-learn's `HistGradientBoostingRegressor` is used.
* `openai` + `OPENAI_API_KEY` — used by the LLM agent; without a key the agent
  falls back to a transparent heuristic so the pipeline still runs end-to-end.
* `wandb` — optional experiment tracking.

## Data

Real datasets are used **offline, once, only to calibrate** the simulation; the
agents never train on historical data, and the true demand function is never
shared with the learning agents. Place raw files under `data/` (see
`src/dynpricing/calibration/calibrate.py` for expected filenames). If none are
present, calibration uses documented synthetic defaults.
