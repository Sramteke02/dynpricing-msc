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

# 2. Cheap LLM reliability/cost check BEFORE any full run (needs MISTRAL_API_KEY;
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

### Live dashboard (Streamlit)

`app/dashboard.py` runs agents **live** through the existing harness and reports
what that run produced — pick agents, a scenario and a seed count in the sidebar
and press Run:

```bash
pip install -e ".[app]"          # or: pip install streamlit
streamlit run app/dashboard.py
```

The oracle is loaded from `results/seasonal_sweep/oracle_cache.json` when the
scenario config fingerprint matches (it is deterministic given config and seed)
and computed live otherwise; the page says which happened. Without
`MISTRAL_API_KEY` the `llm` agent is shown as pending and skipped — never
estimated or faked. It imports the agents, env and harness; it does not modify
them.

**Deploying to Streamlit Community Cloud.** `requirements.txt` at the repo root
is what Community Cloud installs from — it does *not* install the project
itself, which is why `app/dashboard.py` adds `src/` to `sys.path`. Two things to
know:

* This repository is **private**. Community Cloud can only see it if you grant
  the Streamlit GitHub app access to private repositories when you sign in
  ("Authorize streamlit" → include private repos). Without that it reports the
  code as not being in a GitHub repository even though it is.
* The GBM agents retrain per seed, which is CPU-heavy for a free Cloud instance.
  Start at 1–2 seeds there; the 5-seed default takes ~72s locally and will be
  slower on Cloud.
* Set `MISTRAL_API_KEY` under the app's **Secrets**, not in the repo.

### Regenerating the static dashboard

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
`results/gbm_uniform/diagnostics.log`, `results/seasonal_sweep/summary.json` and
the per-episode LLM runs under `results/rq2_llm*/seed*.json`. The script prints
which panels it filled and which came out empty, so a stale or missing input is
visible at build time. One panel is currently empty by design: the
oracle-verification KPI (run `dynpricing verify-oracle` and record its output
under `results/`).

The **RQ2 panel** is built from the committed LLM episodes (Mistral
`mistral-large-2512`). Because the LLM is rate-limited it never joined the
30-seed sweep and has no rows in `metrics.csv`, so each row compares it with
`gbm_uniform` and `fixed` averaged over *exactly the seeds that LLM run used* —
not against their 30-seed means. The build also checks that every episode was
scored against the same oracle profit as the committed sweep.

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
| E3 Four rule-based baselines (+ the oracle, which is the ceiling, not a baseline) | `src/dynpricing/agents/baselines.py` |
| E4 Two-stage gradient-boosting agent | `src/dynpricing/agents/gbm_agent.py` |
| E5 Shared evaluation harness | `src/dynpricing/eval/` |
| E6 CLI + dashboard | `src/dynpricing/cli.py`, `src/dynpricing/dashboard/` |
| D1–D3 LLM agent + reasoning logs + model comparison | `src/dynpricing/agents/llm_agent.py` |

## Optional dependencies

* `xgboost` or `lightgbm` — used by the gradient-boosting agent if available,
  otherwise scikit-learn's `HistGradientBoostingRegressor` is used.
* `openai` + `MISTRAL_API_KEY` — used by the LLM agent. The `openai` package is
  the HTTP client for both providers: Mistral's La Plateforme exposes an
  OpenAI-compatible endpoint, and Mistral is the default provider (set
  `OPENAI_API_KEY` instead only with `provider=openai`). Without a key the
  agent falls back to a transparent heuristic so the pipeline still runs
  end-to-end.
* `wandb` — optional experiment tracking.

## Data

Real datasets are used **offline, once, only to calibrate** the simulation; the
agents never train on historical data, and the true demand function is never
shared with the learning agents. Place raw files under `data/` (see
`src/dynpricing/calibration/calibrate.py` for expected filenames). If none are
present, calibration uses documented synthetic defaults.
