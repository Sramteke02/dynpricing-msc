"""Build the interactive results dashboard from the COMMITTED results files.

Reads only what previous runs wrote under ``results/`` and emits a single
self-contained page at ``results/dashboard/index.html`` — no server, no network,
no external assets. Every number on the page comes from a results file; nothing
is hardcoded. Where a result has not been produced yet, the panel renders an
explicit "not yet run" empty state rather than inventing a value.

Sources
-------
results/metrics_aggregated.json              6-agent run (cost_plus, competitor_match)
results/gbm_uniform/metrics_aggregated.json  5-agent run (adds gbm_uniform)
results/gbm_uniform/paired_gbm_uniform.json  paired gbm_uniform-vs-X comparisons
results/gbm_uniform/price_paths.json         baseline seed-0 price paths
results/gbm_uniform/diagnostics.log          band + choke price
results/seasonal_sweep/summary.json          the amplitude sweep

Regenerate: ``python scripts/build_dashboard_html.py`` (see README).
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "dashboard" / "index.html"

# canonical ladder order; gbm_uniform sits next to the agent it is a control for
AGENT_ORDER = ["fixed", "cost_plus", "competitor_match", "random",
               "gbm", "gbm_uniform", "llm", "oracle"]

PALETTE = {  # validated slots 1-3, light / dark (see dataviz palette.md)
    "light": {"s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a"},
    "dark": {"s1": "#3987e5", "s2": "#d95926", "s3": "#199e70"},
}


def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


# -- run provenance --------------------------------------------------------
def config_provenance() -> dict | None:
    """The config the runs were produced under, plus a content fingerprint."""
    cfg = read_json(ROOT / "configs" / "calibrated.json")
    if not cfg:
        return None
    blob = json.dumps(cfg, sort_keys=True, default=list).encode()
    return {
        "fingerprint": hashlib.sha256(blob).hexdigest()[:12],
        "horizon": cfg.get("horizon"),
        "price_min": cfg.get("price_min"),
        "price_max": cfg.get("price_max"),
        "init_inventory": cfg.get("init_inventory"),
        "seasonal_amplitude": cfg.get("seasonal_amplitude"),
    }


METRIC_COLS = ("gross_profit", "revenue", "market_share", "pricing_stability", "n_steps")


def _load_rows(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def check_run_compatibility() -> dict | None:
    """Compare the two committed runs cell-by-cell, not just on shared means.

    The ladder merges a 6-agent run with the 5-agent gbm_uniform run. That is
    only legitimate if both were produced under the same environment: same
    scenarios, same seed set, same horizon behaviour, and identical numbers
    wherever they overlap. Everything here is recomputed at build time so a
    future mismatch surfaces on the page instead of being silently merged.
    """
    old_p = ROOT / "results" / "metrics.csv"
    new_p = ROOT / "results" / "gbm_uniform" / "metrics.csv"
    if not old_p.exists() or not new_p.exists():
        return None
    old, new = _load_rows(old_p), _load_rows(new_p)

    scen_old = sorted({r["scenario"] for r in old})
    scen_new = sorted({r["scenario"] for r in new})
    seeds_old = sorted({int(r["seed"]) for r in old})
    seeds_new = sorted({int(r["seed"]) for r in new})

    index = {(r["agent"], r["scenario"], r["seed"]): r for r in old}
    worst: dict[str, float] = {c: 0.0 for c in METRIC_COLS}
    n_shared = 0
    for r in new:
        key = (r["agent"], r["scenario"], r["seed"])
        if key not in index:
            continue
        n_shared += 1
        for c in METRIC_COLS:
            worst[c] = max(worst[c], abs(float(r[c]) - float(index[key][c])))

    steps = [int(r["n_steps"]) for r in old + new]
    checks = {
        "scenarios match": scen_old == scen_new,
        "seed sets match": seeds_old == seeds_new,
        "shared cells identical": all(v == 0.0 for v in worst.values()),
    }
    return {
        "compatible": all(checks.values()),
        "checks": checks,
        "n_shared_rows": n_shared,
        "worst_abs_diff": worst,
        "scenarios": scen_old,
        "n_seeds": len(seeds_new),
        "seed_lo": seeds_new[0] if seeds_new else None,
        "seed_hi": seeds_new[-1] if seeds_new else None,
        "n_steps_min": min(steps) if steps else None,
        "n_steps_max": max(steps) if steps else None,
    }


# -- ladder ----------------------------------------------------------------
def build_ladder(compat: dict | None) -> tuple[dict, list[str]]:
    """Merge the committed aggregated runs into {scenario: [agent rows]}.

    The merge happens only if ``check_run_compatibility`` says the two runs are
    the same experiment; otherwise the older run is dropped and the page says so.
    """
    notes: list[str] = []
    newer = read_json(ROOT / "results" / "gbm_uniform" / "metrics_aggregated.json")
    older = read_json(ROOT / "results" / "metrics_aggregated.json")
    if newer is None:
        return {}, ["results/gbm_uniform/metrics_aggregated.json is missing"]

    merged: dict[tuple[str, str], dict] = {}
    for entry in newer:
        merged[(entry["agent"], entry["scenario"])] = entry
    if older:
        if compat and compat["compatible"]:
            for entry in older:
                merged.setdefault((entry["agent"], entry["scenario"]), entry)
            worst = max(compat["worst_abs_diff"].values())
            notes.append(
                f"Agents are merged from two committed runs, verified to be the "
                f"same experiment: identical scenarios and seeds, and all "
                f"{compat['n_shared_rows']} shared episode rows agree on every "
                f"metric (max |Δ| = {worst:.6f}).")
        else:
            failed = ([k for k, v in compat["checks"].items() if not v]
                      if compat else ["the older run could not be read"])
            notes.append("NOT merged — the two runs are not the same experiment "
                         f"({'; '.join(failed)}). Showing the gbm_uniform run only.")

    by_scenario: dict[str, list[dict]] = {}
    for (agent, scenario), entry in merged.items():
        by_scenario.setdefault(scenario, []).append({
            "agent": agent,
            "mean": entry["gross_profit_mean"],
            "ci_low": entry["gross_profit_ci_low"],
            "ci_high": entry["gross_profit_ci_high"],
            "share": entry.get("market_share_mean"),
            "stability": entry.get("pricing_stability_mean"),
            "n_seeds": entry.get("n_seeds"),
        })
    for scenario, rows in by_scenario.items():
        oracle = next((r["mean"] for r in rows if r["agent"] == "oracle"), None)
        for r in rows:
            r["pct"] = (100 * r["mean"] / oracle) if oracle else None
        rows.sort(key=lambda r: AGENT_ORDER.index(r["agent"])
                  if r["agent"] in AGENT_ORDER else len(AGENT_ORDER))
    return by_scenario, notes


# -- KPI tiles -------------------------------------------------------------
def find_oracle_verification() -> dict | None:
    """The verify-oracle gap, if any committed results file records it."""
    for path in sorted((ROOT / "results").rglob("*")):
        if not path.is_file() or path.suffix not in {".json", ".log", ".txt"}:
            continue
        text = path.read_text(errors="ignore")
        m = re.search(r"fluid vs DP gap\s*:\s*([+-]?\d+\.\d+)%", text)
        if m:
            return {"gap_pct": float(m.group(1)), "source": str(path.relative_to(ROOT))}
    return None


def build_kpis(ladder: dict, paired: list | None) -> list[dict]:
    tiles: list[dict] = []

    verif = find_oracle_verification()
    tiles.append({
        "id": "oracle-verification",
        "label": "Oracle verification gap",
        "value": f"{verif['gap_pct']:+.2f}%" if verif else None,
        "sub": (f"fluid oracle vs exact backward-induction oracle · {verif['source']}"
                if verif else None),
        "empty": None if verif else
                 "not yet run — no committed results file records a verify-oracle "
                 "gap. Produce one with: dynpricing verify-oracle "
                 "--config configs/calibrated.json",
    })

    base = ladder.get("baseline", [])
    gbm = next((r for r in base if r["agent"] == "gbm"), None)
    gbmu = next((r for r in base if r["agent"] == "gbm_uniform"), None)
    if gbm and gbmu and gbm["pct"] is not None:
        pair = None
        if paired:
            pair = next((p for p in paired
                         if p["scenario"] == "baseline"
                         and p["agent_a"] == "gbm_uniform" and p["agent_b"] == "gbm"), None)
        sub = (f"was {gbm['pct']:.1f}% of oracle · now {gbmu['pct']:.1f}%")
        if pair:
            sub += (f" · paired Δ {pair['mean_diff']:+,.0f} "
                    f"[{pair['ci_low']:+,.0f}, {pair['ci_high']:+,.0f}]")
        tiles.append({
            "id": "recovery",
            "label": "gbm → gbm_uniform recovery (baseline)",
            "value": f"{gbmu['pct']:.1f}%",
            "delta": f"+{gbmu['pct'] - gbm['pct']:.1f} pts of oracle",
            "sub": sub,
            "hero": True,
            "empty": None,
        })
    else:
        tiles.append({"id": "recovery", "label": "gbm → gbm_uniform recovery (baseline)",
                      "value": None, "empty": "not yet run — no baseline rows for "
                      "gbm and gbm_uniform in the committed results."})

    has_llm = any(r["agent"].split(":")[0] == "llm"
                  for rows in ladder.values() for r in rows)
    tiles.append({
        "id": "rq2",
        "label": "RQ2 — LLM vs GBM",
        "value": None if not has_llm else "see ladder",
        "pending": not has_llm,
        "empty": ("pending — the LLM agent has not been run. No llm rows exist in "
                  "any committed results file (needs OPENAI_API_KEY)."
                  if not has_llm else None),
    })
    return tiles


# -- amplitude sweep -------------------------------------------------------
def build_amplitude() -> dict | None:
    summary = read_json(ROOT / "results" / "seasonal_sweep" / "summary.json")
    if not summary:
        return None
    amps = summary["amplitudes"]
    return {
        "amplitudes": amps,
        "n_seeds": summary.get("n_seeds"),
        "series": {
            name: [summary["pct_of_oracle"][name][f"{a}"] for a in amps]
            for name in ("fixed", "gbm", "gbm_uniform", "oracle")
            if name in summary.get("pct_of_oracle", {})
        },
        "profit": {
            name: [summary["mean_profit"][name][f"{a}"] for a in amps]
            for name in ("fixed", "gbm", "gbm_uniform", "oracle")
            if name in summary.get("mean_profit", {})
        },
        # position-aligned lists, not float-keyed dicts: JS String(0.0) is "0",
        # which would not match a "0.0" key
        "gap": [summary["gbm_uniform_minus_fixed"][f"{a}"] for a in amps],
        "dispersion": [summary.get("optimal_price_dispersion", {}).get(f"{a}")
                       for a in amps],
    }


# -- price paths -----------------------------------------------------------
def build_paths() -> dict | None:
    paths = read_json(ROOT / "results" / "gbm_uniform" / "price_paths.json")
    if not paths or "gbm" not in paths or "gbm_uniform" not in paths:
        return None
    log = ROOT / "results" / "gbm_uniform" / "diagnostics.log"
    band = choke = None
    if log.exists():
        text = log.read_text()
        m = re.search(r"legal band \[([\d.]+), ([\d.]+)\]", text)
        if m:
            band = [float(m.group(1)), float(m.group(2))]
        m = re.search(r"true choke price \(day 0\) ~ ([\d.]+)", text)
        if m:
            choke = float(m.group(1))
    out = {"gbm": paths["gbm"], "gbm_uniform": paths["gbm_uniform"],
           "band": band, "choke": choke, "stats": {}}
    for name in ("gbm", "gbm_uniform"):
        p = paths[name]
        out["stats"][name] = {
            "min": min(p), "max": max(p), "n": len(p),
            "at_max": sum(1 for v in p if band and v >= band[1] - 1e-6),
            "above_choke": sum(1 for v in p if choke and v > choke),
        }
    return out


# -- page ------------------------------------------------------------------
def render(data: dict) -> str:
    payload = json.dumps(data, indent=None, separators=(",", ":"))
    css = CSS.replace("__L1__", PALETTE["light"]["s1"]) \
             .replace("__L2__", PALETTE["light"]["s2"]) \
             .replace("__L3__", PALETTE["light"]["s3"]) \
             .replace("__D1__", PALETTE["dark"]["s1"]) \
             .replace("__D2__", PALETTE["dark"]["s2"]) \
             .replace("__D3__", PALETTE["dark"]["s3"])
    return (HTML_HEAD + "<style>\n" + css + "\n</style>\n" + BODY
            + '\n<script type="application/json" id="data">' + payload + "</script>\n"
            + "<script>\n" + JS + "\n</script>\n</body>\n</html>\n")


HTML_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dynamic Pricing — Results Dashboard</title>
"""

CSS = """
:root {
  color-scheme: light;
  --surface-1: #fcfcfb;
  --plane: #f9f9f7;
  --text-primary: #0b0b0b;
  --text-secondary: #52514e;
  --muted: #898781;
  --grid: #e1e0d9;
  --axis: #c3c2b7;
  --border: rgba(11,11,11,0.10);
  --series-1: __L1__;
  --series-2: __L2__;
  --series-3: __L3__;
  --good: #006300;
  --warning: #fab219;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface-1: #1a1a19;
    --plane: #0d0d0d;
    --text-primary: #ffffff;
    --text-secondary: #c3c2b7;
    --muted: #898781;
    --grid: #2c2c2a;
    --axis: #383835;
    --border: rgba(255,255,255,0.10);
    --series-1: __D1__;
    --series-2: __D2__;
    --series-3: __D3__;
    --good: #0ca30c;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-1: #1a1a19;
  --plane: #0d0d0d;
  --text-primary: #ffffff;
  --text-secondary: #c3c2b7;
  --muted: #898781;
  --grid: #2c2c2a;
  --axis: #383835;
  --border: rgba(255,255,255,0.10);
  --series-1: __D1__;
  --series-2: __D2__;
  --series-3: __D3__;
  --good: #0ca30c;
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 0 1.25rem 4rem;
  background: var(--plane); color: var(--text-primary);
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
}
.wrap { max-width: 1080px; margin: 0 auto; }
header { padding: 2.25rem 0 1.25rem; display: flex; gap: 1rem;
  align-items: flex-start; justify-content: space-between; flex-wrap: wrap; }
h1 { font-size: 1.5rem; margin: 0 0 .3rem; letter-spacing: -0.01em; }
.subtitle { color: var(--text-secondary); font-size: .875rem; margin: 0; max-width: 62ch; }
h2 { font-size: 1.0625rem; margin: 0; letter-spacing: -0.005em; }
.section {
  background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 10px; padding: 1.15rem 1.25rem 1.35rem; margin-bottom: 1.15rem;
}
.section-head { display: flex; gap: .75rem; align-items: baseline;
  justify-content: space-between; flex-wrap: wrap; margin-bottom: .2rem; }
.section-note { color: var(--text-secondary); font-size: .8125rem; margin: .3rem 0 1rem; max-width: 78ch; }
.controls { display: flex; gap: .4rem; align-items: center; flex-wrap: wrap; }
label.ctl { color: var(--text-secondary); font-size: .8125rem; }
select, button {
  font: inherit; font-size: .8125rem; color: var(--text-primary);
  background: var(--surface-1); border: 1px solid var(--axis);
  border-radius: 6px; padding: .3rem .55rem; cursor: pointer;
}
select:focus-visible, button:focus-visible { outline: 2px solid var(--series-1); outline-offset: 1px; }
.kpis { display: grid; gap: 1.15rem; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); }
.kpi { background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 10px; padding: 1.1rem 1.15rem; }
.kpi .label { color: var(--text-secondary); font-size: .8125rem; margin-bottom: .45rem; }
.kpi .value { font-size: 2rem; font-weight: 600; letter-spacing: -0.02em; line-height: 1.1; }
.kpi.hero .value { font-size: 3rem; }
.kpi .delta { color: var(--good); font-size: .875rem; font-weight: 600; margin-top: .2rem; }
.kpi .sub { color: var(--text-secondary); font-size: .78125rem; margin-top: .45rem; }
.kpi .empty { color: var(--text-secondary); font-size: .8125rem; margin-top: .3rem; }
.badge { display: inline-block; font-size: .6875rem; font-weight: 600;
  letter-spacing: .04em; text-transform: uppercase; padding: .15rem .45rem;
  border-radius: 4px; border: 1px solid var(--axis); color: var(--text-secondary); }
.chart { position: relative; width: 100%; }
svg { width: 100%; height: auto; display: block; overflow: visible; }
.legend { display: flex; gap: 1rem; flex-wrap: wrap; margin: .1rem 0 .6rem; }
.legend span { display: inline-flex; align-items: center; gap: .4rem;
  color: var(--text-secondary); font-size: .8125rem; }
.legend i { display: inline-block; width: 14px; height: 2px; border-radius: 1px; }
.legend i.rect { height: 10px; width: 10px; border-radius: 2px; }
.tooltip {
  position: absolute; pointer-events: none; z-index: 5; opacity: 0;
  transition: opacity .1s; background: var(--surface-1);
  border: 1px solid var(--border); border-radius: 8px; padding: .5rem .6rem;
  box-shadow: 0 6px 20px rgba(0,0,0,.13); font-size: .8125rem; min-width: 130px;
}
.tooltip .tt-title { color: var(--text-secondary); font-size: .75rem; margin-bottom: .3rem; }
.tooltip .tt-row { display: flex; align-items: center; gap: .45rem; margin-top: .16rem; }
.tooltip .tt-key { display: inline-block; width: 12px; height: 2px; border-radius: 1px; flex: none; }
.tooltip .tt-val { font-weight: 600; font-variant-numeric: tabular-nums; }
.tooltip .tt-name { color: var(--text-secondary); }
table { border-collapse: collapse; width: 100%; font-size: .8125rem; margin-top: .5rem; }
th, td { text-align: right; padding: .35rem .5rem; border-bottom: 1px solid var(--grid);
  font-variant-numeric: tabular-nums; }
th:first-child, td:first-child { text-align: left; font-variant-numeric: normal; }
th { color: var(--text-secondary); font-weight: 600; }
.tablewrap { overflow-x: auto; }
.tablewrap[hidden] { display: none; }
.empty-panel { border: 1px dashed var(--axis); border-radius: 8px; padding: 1.5rem;
  color: var(--text-secondary); font-size: .875rem; text-align: center; }
.prov { font-size: .78125rem; color: var(--text-secondary); border: 1px solid var(--border);
  border-radius: 8px; padding: .55rem .7rem; margin: 0 0 1rem;
  display: flex; gap: .35rem 1.1rem; flex-wrap: wrap; align-items: baseline; }
.prov b { color: var(--text-primary); font-weight: 600; font-variant-numeric: tabular-nums; }
.prov .ok { color: var(--good); font-weight: 600; }
.prov.warn { border-color: var(--warning); border-width: 2px; }
.prov .bad { color: var(--warning); font-weight: 600; }
footer { color: var(--text-secondary); font-size: .78125rem; padding-top: .5rem; }
footer code { font-size: .95em; }
"""

BODY = """</head>
<body>
<div class="wrap">
<header>
  <div>
    <h1>Dynamic Pricing — Results Dashboard</h1>
    <p class="subtitle">Every number is read from the committed results files; nothing is
    hardcoded. Panels with no underlying run show an explicit empty state.</p>
  </div>
  <div class="controls">
    <button id="theme-toggle" type="button" aria-label="Toggle colour theme">Theme</button>
  </div>
</header>

<section class="kpis" id="kpis" aria-label="Key results"></section>

<section class="section" aria-labelledby="ladder-h">
  <div class="section-head">
    <h2 id="ladder-h">Agent ladder — % of oracle</h2>
    <div class="controls">
      <label class="ctl" for="scenario">Scenario</label>
      <select id="scenario"></select>
      <button type="button" data-table="ladder">Show table</button>
    </div>
  </div>
  <p class="section-note" id="ladder-note"></p>
  <div id="provenance"></div>
  <div class="chart" id="ladder-chart"></div>
  <div class="tablewrap" id="ladder-table" hidden></div>
</section>

<section class="section" aria-labelledby="amp-h">
  <div class="section-head">
    <h2 id="amp-h">RQ1 — profit vs seasonal amplitude</h2>
    <div class="controls"><button type="button" data-table="amp">Show table</button></div>
  </div>
  <p class="section-note" id="amp-note"></p>
  <div class="legend" id="amp-legend"></div>
  <div class="chart" id="amp-chart"></div>
  <div class="tablewrap" id="amp-table" hidden></div>
</section>

<section class="section" aria-labelledby="paths-h">
  <div class="section-head">
    <h2 id="paths-h">GBM diagnosis — price path, baseline seed 0</h2>
    <div class="controls"><button type="button" data-table="paths">Show table</button></div>
  </div>
  <p class="section-note" id="paths-note"></p>
  <div class="legend" id="paths-legend"></div>
  <div class="chart" id="paths-chart"></div>
  <div class="tablewrap" id="paths-table" hidden></div>
</section>

<footer id="footer"></footer>
</div>
"""

JS = r"""
const DATA = JSON.parse(document.getElementById('data').textContent);
const SVGNS = 'http://www.w3.org/2000/svg';
const fmt = (v, d = 1) => v == null ? '—' : v.toLocaleString(undefined,
  { minimumFractionDigits: d, maximumFractionDigits: d });
const el = (tag, attrs = {}, text) => {
  const n = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (text != null) n.textContent = text;
  return n;
};
const html = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/* ---------- theme ---------- */
const toggle = document.getElementById('theme-toggle');
toggle.addEventListener('click', () => {
  const dark = matchMedia('(prefers-color-scheme: dark)').matches;
  const cur = document.documentElement.getAttribute('data-theme') || (dark ? 'dark' : 'light');
  document.documentElement.setAttribute('data-theme', cur === 'dark' ? 'light' : 'dark');
  renderAll();
});
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', renderAll);

/* ---------- tooltip ---------- */
function makeTooltip(host) {
  const tip = html('div', 'tooltip');
  host.appendChild(tip);
  return {
    show(x, y, title, rows) {
      tip.replaceChildren();
      tip.appendChild(html('div', 'tt-title', title));
      for (const r of rows) {
        const row = html('div', 'tt-row');
        const key = html('span', 'tt-key');
        key.style.background = r.color;
        row.appendChild(key);
        row.appendChild(html('span', 'tt-val', r.value));
        row.appendChild(html('span', 'tt-name', r.name));
        tip.appendChild(row);
      }
      tip.style.opacity = '1';
      const hw = host.clientWidth, tw = tip.offsetWidth;
      tip.style.left = Math.max(0, Math.min(x + 12, hw - tw - 4)) + 'px';
      tip.style.top = Math.max(0, y - tip.offsetHeight - 10) + 'px';
    },
    hide() { tip.style.opacity = '0'; },
  };
}

/* ---------- KPI strip ---------- */
function renderKpis() {
  const host = document.getElementById('kpis');
  host.replaceChildren();
  for (const k of DATA.kpis) {
    const card = html('div', 'kpi' + (k.hero ? ' hero' : ''));
    card.appendChild(html('div', 'label', k.label));
    if (k.value != null) {
      card.appendChild(html('div', 'value', k.value));
      if (k.delta) card.appendChild(html('div', 'delta', k.delta));
      if (k.sub) card.appendChild(html('div', 'sub', k.sub));
    } else {
      card.appendChild(html('span', 'badge', k.pending ? 'pending' : 'not yet run'));
      card.appendChild(html('div', 'empty', k.empty || ''));
    }
    host.appendChild(card);
  }
}

/* ---------- run provenance ---------- */
function renderProvenance() {
  const host = document.getElementById('provenance');
  host.replaceChildren();
  const c = DATA.config, k = DATA.compat;
  if (!c && !k) return;
  const box = html('div', 'prov' + (k && !k.compatible ? ' warn' : ''));
  const add = (label, value, cls) => {
    const s = html('span');
    s.appendChild(document.createTextNode(label + ' '));
    s.appendChild(html('b', cls, value));
    box.appendChild(s);
  };
  if (c) {
    add('config', c.fingerprint);
    add('horizon', String(c.horizon));
    add('band', `[${c.price_min.toFixed(3)}, ${c.price_max.toFixed(3)}]`);
    add('inventory', c.init_inventory.toLocaleString());
  }
  if (k) {
    add('seeds', `${k.seed_lo}–${k.seed_hi} (${k.n_seeds})`);
    add('episode steps', `${k.n_steps_min}–${k.n_steps_max}`);
    if (k.compatible) {
      add('same experiment', 'verified', 'ok');
    } else {
      const failed = Object.keys(k.checks).filter(n => !k.checks[n]);
      add('runs differ', failed.join('; '), 'bad');
    }
  }
  host.appendChild(box);
}

/* ---------- 2. agent ladder ---------- */
const sel = document.getElementById('scenario');
function renderLadder() {
  const host = document.getElementById('ladder-chart');
  host.replaceChildren();
  const rows = (DATA.ladder[sel.value] || []).filter(r => r.pct != null);
  const note = document.getElementById('ladder-note');
  if (!rows.length) {
    host.appendChild(html('div', 'empty-panel', 'not yet run — no rows for this scenario.'));
    return;
  }
  const n = rows[0].n_seeds;
  note.textContent = `Mean gross profit as a share of the oracle ceiling, `
    + `${n} paired seeds. Bar values are direct-labelled; hover for the profit and its `
    + `bootstrap 95% CI. ${DATA.ladderNotes.join(' · ')}`;

  const W = 800, rowH = 34, padL = 128, padR = 62, padT = 12;
  const H = padT + rows.length * rowH + 34;
  const xmax = 105;
  const x = v => padL + (v / xmax) * (W - padL - padR);
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img',
    'aria-label': 'Agent ladder, percent of oracle' });

  for (const t of [0, 25, 50, 75, 100]) {
    svg.appendChild(el('line', { x1: x(t), x2: x(t), y1: padT, y2: padT + rows.length * rowH,
      stroke: cssVar('--grid'), 'stroke-width': 1 }));
    svg.appendChild(el('text', { x: x(t), y: padT + rows.length * rowH + 18,
      'text-anchor': 'middle', fill: cssVar('--muted'), 'font-size': 11 }, t + '%'));
  }
  // oracle ceiling: solid hairline at 100%
  svg.appendChild(el('line', { x1: x(100), x2: x(100), y1: padT - 2,
    y2: padT + rows.length * rowH, stroke: cssVar('--axis'), 'stroke-width': 1 }));
  svg.appendChild(el('text', { x: x(100) + 4, y: padT + 8, fill: cssVar('--muted'),
    'font-size': 10 }, 'oracle = 100%'));

  const tip = makeTooltip(host);
  const barH = Math.min(24, rowH - 12);
  rows.forEach((r, i) => {
    const y = padT + i * rowH + (rowH - barH) / 2;
    const w = Math.max(2, x(r.pct) - x(0));
    const rad = Math.min(4, w);
    // square at the baseline, 4px rounded data-end
    const d = `M${x(0)},${y} H${x(0) + w - rad} a${rad},${rad} 0 0 1 ${rad},${rad}`
      + ` V${y + barH - rad} a${rad},${rad} 0 0 1 ${-rad},${rad} H${x(0)} Z`;
    const bar = el('path', { d, fill: cssVar('--series-1') });
    svg.appendChild(bar);
    svg.appendChild(el('text', { x: padL - 10, y: y + barH / 2 + 4, 'text-anchor': 'end',
      fill: cssVar('--text-secondary'), 'font-size': 12 }, r.agent));
    svg.appendChild(el('text', { x: x(r.pct) + 7, y: y + barH / 2 + 4,
      fill: cssVar('--text-primary'), 'font-size': 12 }, r.pct.toFixed(1) + '%'));
    // hit target spans the whole row (bigger than the mark)
    const hit = el('rect', { x: padL, y: padT + i * rowH, width: W - padL - padR,
      height: rowH, fill: 'transparent', tabindex: 0, role: 'img',
      'aria-label': `${r.agent}: ${r.pct.toFixed(1)}% of oracle, `
        + `profit ${Math.round(r.mean)}` });
    const show = (ev) => {
      const bb = host.getBoundingClientRect();
      const px = ev.clientX != null ? ev.clientX - bb.left : x(r.pct) * bb.width / W;
      const py = ev.clientY != null ? ev.clientY - bb.top : (padT + i * rowH) * bb.width / W;
      tip.show(px, py, r.agent, [
        { color: cssVar('--series-1'), value: r.pct.toFixed(1) + '%', name: 'of oracle' },
        { color: cssVar('--series-1'), value: fmt(r.mean, 0), name: 'gross profit' },
        { color: cssVar('--axis'),
          value: `[${fmt(r.ci_low, 0)}, ${fmt(r.ci_high, 0)}]`, name: '95% CI' },
      ]);
      bar.setAttribute('opacity', '0.82');
    };
    const hide = () => { tip.hide(); bar.setAttribute('opacity', '1'); };
    hit.addEventListener('pointermove', show);
    hit.addEventListener('pointerleave', hide);
    hit.addEventListener('focus', show);
    hit.addEventListener('blur', hide);
    svg.appendChild(hit);
  });
  host.appendChild(svg);

  const t = html('table');
  const head = html('tr');
  ['Agent', '% of oracle', 'Gross profit', 'CI low', 'CI high', 'Market share', 'Stability']
    .forEach(h => head.appendChild(html('th', null, h)));
  t.appendChild(head);
  for (const r of rows) {
    const tr = html('tr');
    [r.agent, r.pct.toFixed(2) + '%', fmt(r.mean, 1), fmt(r.ci_low, 1), fmt(r.ci_high, 1),
     r.share == null ? '—' : (100 * r.share).toFixed(1) + '%',
     r.stability == null ? '—' : r.stability.toFixed(3)]
      .forEach(v => tr.appendChild(html('td', null, v)));
    t.appendChild(tr);
  }
  const wrap = document.getElementById('ladder-table');
  wrap.replaceChildren(t);
}

/* ---------- shared line-chart renderer ---------- */
function lineChart({ hostId, xs, series, colors, yMin, yMax, xLabel, yLabel,
                     xTickFmt, yTickFmt, valFmt, endLabels, refLines = [], xIsIndex }) {
  const host = document.getElementById(hostId);
  host.replaceChildren();
  const W = 800, H = 350, padL = 54, padR = endLabels ? 118 : 22, padT = 30, padB = 44;
  const x = v => padL + ((v - xs[0]) / (xs[xs.length - 1] - xs[0])) * (W - padL - padR);
  const y = v => padT + (1 - (v - yMin) / (yMax - yMin)) * (H - padT - padB);
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': yLabel });

  const ticks = 5;
  for (let i = 0; i <= ticks; i++) {
    const v = yMin + (i / ticks) * (yMax - yMin);
    svg.appendChild(el('line', { x1: padL, x2: W - padR, y1: y(v), y2: y(v),
      stroke: cssVar('--grid'), 'stroke-width': 1 }));
    svg.appendChild(el('text', { x: padL - 8, y: y(v) + 4, 'text-anchor': 'end',
      fill: cssVar('--muted'), 'font-size': 11 }, yTickFmt(v)));
  }
  svg.appendChild(el('line', { x1: padL, x2: W - padR, y1: y(yMin), y2: y(yMin),
    stroke: cssVar('--axis'), 'stroke-width': 1 }));

  const xticks = xIsIndex ? [0, 0.25, 0.5, 0.75, 1].map(f =>
    xs[Math.round(f * (xs.length - 1))]) : xs;
  for (const t of xticks) {
    svg.appendChild(el('text', { x: x(t), y: H - padB + 20, 'text-anchor': 'middle',
      fill: cssVar('--muted'), 'font-size': 11 }, xTickFmt(t)));
  }
  svg.appendChild(el('text', { x: (padL + W - padR) / 2, y: H - 6, 'text-anchor': 'middle',
    fill: cssVar('--text-secondary'), 'font-size': 11.5 }, xLabel));
  svg.appendChild(el('text', { x: 4, y: 12, fill: cssVar('--text-secondary'),
    'font-size': 11.5 }, yLabel));

  for (const r of refLines) {
    if (r.v == null || r.v < yMin || r.v > yMax) continue;
    svg.appendChild(el('line', { x1: padL, x2: W - padR, y1: y(r.v), y2: y(r.v),
      stroke: cssVar('--axis'), 'stroke-width': 1 }));
    // right-anchored: the left edge is where the failing path climbs steeply
    svg.appendChild(el('text', { x: W - padR - 4, y: y(r.v) - 5, 'text-anchor': 'end',
      fill: cssVar('--muted'), 'font-size': 10.5 }, r.label));
  }

  const names = Object.keys(series);
  names.forEach((name) => {
    const pts = series[name];
    const d = pts.map((v, i) => `${i ? 'L' : 'M'}${x(xs[i]).toFixed(2)},${y(v).toFixed(2)}`).join(' ');
    svg.appendChild(el('path', { d, fill: 'none', stroke: colors[name], 'stroke-width': 2,
      'stroke-linejoin': 'round', 'stroke-linecap': 'round' }));
    if (endLabels) {
      const lastX = x(xs[xs.length - 1]), lastY = y(pts[pts.length - 1]);
      svg.appendChild(el('circle', { cx: lastX, cy: lastY, r: 4.5, fill: colors[name],
        stroke: cssVar('--surface-1'), 'stroke-width': 2 }));
      svg.appendChild(el('text', { x: lastX + 10, y: lastY + 4,
        fill: cssVar('--text-primary'), 'font-size': 11.5 },
        `${name} ${valFmt(pts[pts.length - 1])}`));
    }
  });

  // crosshair + one tooltip listing every series at that X
  const tip = makeTooltip(host);
  const cross = el('line', { x1: 0, x2: 0, y1: padT, y2: y(yMin),
    stroke: cssVar('--axis'), 'stroke-width': 1, opacity: 0 });
  svg.appendChild(cross);
  const dots = names.map(n => {
    const c = el('circle', { r: 4.5, fill: colors[n], stroke: cssVar('--surface-1'),
      'stroke-width': 2, opacity: 0 });
    svg.appendChild(c);
    return c;
  });
  const overlay = el('rect', { x: padL, y: padT, width: W - padL - padR,
    height: H - padT - padB, fill: 'transparent', tabindex: 0 });
  const move = (ev) => {
    const bb = host.getBoundingClientRect();
    const scale = W / bb.width;
    const mx = (ev.clientX - bb.left) * scale;
    let best = 0, bd = Infinity;
    xs.forEach((v, i) => { const d = Math.abs(x(v) - mx); if (d < bd) { bd = d; best = i; } });
    cross.setAttribute('x1', x(xs[best]));
    cross.setAttribute('x2', x(xs[best]));
    cross.setAttribute('opacity', 1);
    names.forEach((n, k) => {
      dots[k].setAttribute('cx', x(xs[best]));
      dots[k].setAttribute('cy', y(series[n][best]));
      dots[k].setAttribute('opacity', 1);
    });
    tip.show((x(xs[best])) / scale, (ev.clientY - bb.top),
      xLabel.split('(')[0].trim() + ' ' + xTickFmt(xs[best]),
      names.map(n => ({ color: colors[n], value: valFmt(series[n][best]), name })));
  };
  const leave = () => {
    tip.hide(); cross.setAttribute('opacity', 0);
    dots.forEach(d => d.setAttribute('opacity', 0));
  };
  overlay.addEventListener('pointermove', move);
  overlay.addEventListener('pointerleave', leave);
  svg.appendChild(overlay);
  host.appendChild(svg);
}

function renderLegend(hostId, names, colors, kind) {
  const host = document.getElementById(hostId);
  host.replaceChildren();
  for (const n of names) {
    const s = html('span');
    const i = html('i', kind === 'rect' ? 'rect' : null);
    i.style.background = colors[n];
    s.appendChild(i);
    s.appendChild(document.createTextNode(n));
    host.appendChild(s);
  }
}

/* ---------- 3. amplitude curve ---------- */
function renderAmplitude() {
  const host = document.getElementById('amp-chart');
  const note = document.getElementById('amp-note');
  if (!DATA.amplitude) {
    host.replaceChildren(html('div', 'empty-panel',
      'not yet run — results/seasonal_sweep/summary.json is missing.'));
    document.getElementById('amp-legend').replaceChildren();
    return;
  }
  const A = DATA.amplitude;
  const names = ['fixed', 'gbm_uniform', 'oracle'].filter(n => A.series[n]);
  const colors = { fixed: cssVar('--series-2'), gbm_uniform: cssVar('--series-1'),
    oracle: cssVar('--series-3') };
  const vals = names.flatMap(n => A.series[n]);
  // keep the top tick at exactly 100 so no tick reads above the ceiling
  const lo = Math.floor(Math.min(...vals) - 1), hi = 100;
  note.textContent = `Mean gross profit as a share of the oracle ceiling at each seasonal `
    + `amplitude, ${A.n_seeds} seeds. The y-axis starts at ${lo}%, not 0. gbm is omitted `
    + `from the plot (it ranges ${Math.min(...A.series.gbm).toFixed(1)}–`
    + `${Math.max(...A.series.gbm).toFixed(1)}% and would flatten the others); it is in the table.`;
  renderLegend('amp-legend', names, colors);
  lineChart({
    hostId: 'amp-chart', xs: A.amplitudes,
    series: Object.fromEntries(names.map(n => [n, A.series[n]])),
    colors, yMin: lo, yMax: hi,
    xLabel: 'seasonal amplitude', yLabel: '% of oracle',
    xTickFmt: v => v.toFixed(2), yTickFmt: v => v.toFixed(0) + '%',
    valFmt: v => v.toFixed(2) + '%', endLabels: true,
  });
  const t = html('table');
  const head = html('tr');
  ['Amplitude', 'fixed', 'gbm', 'gbm_uniform', 'oracle', 'gbm_uniform − fixed (pts)',
   'paired Δ profit', '95% CI'].forEach(h => head.appendChild(html('th', null, h)));
  t.appendChild(head);
  A.amplitudes.forEach((a, i) => {
    const g = A.gap[i];
    const tr = html('tr');
    [a.toFixed(2),
     A.series.fixed[i].toFixed(2) + '%',
     A.series.gbm ? A.series.gbm[i].toFixed(2) + '%' : '—',
     A.series.gbm_uniform[i].toFixed(2) + '%',
     A.series.oracle[i].toFixed(2) + '%',
     g.pts_of_oracle.toFixed(2),
     fmt(g.mean_diff, 1),
     `[${fmt(g.ci_low, 1)}, ${fmt(g.ci_high, 1)}]`]
      .forEach(v => tr.appendChild(html('td', null, v)));
    t.appendChild(tr);
  });
  document.getElementById('amp-table').replaceChildren(t);
}

/* ---------- 4. price paths ---------- */
function renderPaths() {
  const host = document.getElementById('paths-chart');
  const note = document.getElementById('paths-note');
  if (!DATA.paths) {
    host.replaceChildren(html('div', 'empty-panel',
      'not yet run — results/gbm_uniform/price_paths.json is missing.'));
    document.getElementById('paths-legend').replaceChildren();
    return;
  }
  const P = DATA.paths;
  const colors = { gbm: cssVar('--series-2'), gbm_uniform: cssVar('--series-1') };
  const xs = P.gbm.map((_, i) => i);
  const all = P.gbm.concat(P.gbm_uniform);
  const hi = Math.max(...all, P.band ? P.band[1] : 0);
  const s = P.stats;
  note.textContent = `Realised price each day of the baseline episode on seed 0 — the seed `
    + `the original agent fails on. gbm spends ${s.gbm.at_max}/${s.gbm.n} steps at price_max `
    + `and ${s.gbm.above_choke}/${s.gbm.n} above the choke; gbm_uniform `
    + `${s.gbm_uniform.at_max}/${s.gbm_uniform.n} and ${s.gbm_uniform.above_choke}/`
    + `${s.gbm_uniform.n}.`;
  renderLegend('paths-legend', ['gbm', 'gbm_uniform'], colors);
  lineChart({
    hostId: 'paths-chart', xs,
    series: { gbm: P.gbm, gbm_uniform: P.gbm_uniform }, colors,
    yMin: 0, yMax: Math.ceil(hi * 10) / 10,
    xLabel: 'day of episode', yLabel: 'price',
    xTickFmt: v => String(v), yTickFmt: v => v.toFixed(1),
    valFmt: v => v.toFixed(3), endLabels: false, xIsIndex: true,
    refLines: [
      P.choke != null ? { v: P.choke, label: `choke ${P.choke.toFixed(3)} (demand = 0 above)` } : {},
      P.band ? { v: P.band[1], label: `price_max ${P.band[1].toFixed(3)}` } : {},
    ].filter(r => r.v != null),
  });
  const t = html('table');
  const head = html('tr');
  ['Agent', 'min price', 'max price', 'steps at price_max', 'steps above choke', 'steps']
    .forEach(h => head.appendChild(html('th', null, h)));
  t.appendChild(head);
  for (const n of ['gbm', 'gbm_uniform']) {
    const tr = html('tr');
    [n, s[n].min.toFixed(3), s[n].max.toFixed(3), `${s[n].at_max}/${s[n].n}`,
     `${s[n].above_choke}/${s[n].n}`, s[n].n]
      .forEach(v => tr.appendChild(html('td', null, v)));
    t.appendChild(tr);
  }
  document.getElementById('paths-table').replaceChildren(t);
}

/* ---------- table toggles ---------- */
for (const btn of document.querySelectorAll('button[data-table]')) {
  btn.addEventListener('click', () => {
    const wrap = document.getElementById(btn.dataset.table + '-table');
    const hidden = wrap.hasAttribute('hidden');
    if (hidden) wrap.removeAttribute('hidden'); else wrap.setAttribute('hidden', '');
    btn.textContent = hidden ? 'Hide table' : 'Show table';
  });
}

/* ---------- boot ---------- */
for (const s of DATA.scenarios) {
  const o = document.createElement('option');
  o.value = s; o.textContent = s;
  sel.appendChild(o);
}
sel.value = DATA.scenarios.includes('baseline') ? 'baseline' : DATA.scenarios[0];
sel.addEventListener('change', renderLadder);
document.getElementById('footer').textContent = DATA.footer;

function renderAll() {
  renderKpis(); renderProvenance(); renderLadder(); renderAmplitude(); renderPaths();
}
renderAll();
addEventListener('resize', () => { renderLadder(); renderAmplitude(); renderPaths(); });
"""


def main() -> int:
    compat = check_run_compatibility()
    config = config_provenance()
    ladder, ladder_notes = build_ladder(compat)
    if not ladder:
        print("[error] no ladder data found under results/", file=sys.stderr)
        return 1
    paired = read_json(ROOT / "results" / "gbm_uniform" / "paired_gbm_uniform.json")
    data = {
        "kpis": build_kpis(ladder, paired),
        "config": config,
        "compat": compat,
        "ladder": ladder,
        "ladderNotes": ladder_notes,
        "scenarios": sorted(ladder.keys()),
        "amplitude": build_amplitude(),
        "paths": build_paths(),
        "footer": ("Generated by scripts/build_dashboard_html.py from the committed "
                   "results files. Regenerate after a new run: "
                   "python scripts/build_dashboard_html.py"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(data))
    size = OUT.stat().st_size
    print(f"[ok] wrote {OUT.relative_to(ROOT)} ({size:,} bytes)")
    print(f"     scenarios: {', '.join(data['scenarios'])}")
    print(f"     agents   : {', '.join(r['agent'] for r in ladder['baseline'])}")
    for k in data["kpis"]:
        state = k["value"] if k["value"] is not None else "(empty: " + \
            ("pending" if k.get("pending") else "not yet run") + ")"
        print(f"     KPI {k['id']:<20} {state}")
    print(f"     amplitude panel: {'present' if data['amplitude'] else 'EMPTY'}")
    print(f"     price paths    : {'present' if data['paths'] else 'EMPTY'}")
    if config:
        print(f"     config         : {config['fingerprint']} horizon={config['horizon']} "
              f"band=[{config['price_min']:.3f}, {config['price_max']:.3f}]")
    if compat:
        for name, passed in compat["checks"].items():
            print(f"     [{'PASS' if passed else 'FAIL'}] {name}")
        print(f"     shared rows    : {compat['n_shared_rows']} "
              f"(worst |Δ| {max(compat['worst_abs_diff'].values()):.6f} "
              f"across {', '.join(METRIC_COLS)})")
        if not compat["compatible"]:
            print("[warn] runs are NOT the same experiment; the page says so and "
                  "shows the gbm_uniform run only", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
