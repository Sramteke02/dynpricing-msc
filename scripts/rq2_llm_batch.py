"""Run several RQ2 LLM episodes back to back, then summarise the sample.

Episodes run STRICTLY SEQUENTIALLY: on a 4 req/min tier two concurrent episodes
would rate-limit each other and strict mode would abort both.

An episode that aborts (strict mode, or a network outage that outlasts the
retries) is retried once and then recorded as failed. Failed seeds are excluded
from the mean and reported separately — never silently dropped.

    export MISTRAL_API_KEY=...
    python scripts/rq2_llm_batch.py --seeds 2,3,4,5,6
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EPISODE = ROOT / "scripts" / "rq2_llm_episode.py"


def stem(seed: int, scenario: str) -> str:
    """Must match the episode script's own naming."""
    return f"seed{seed}" + ("" if scenario == "baseline" else f"_{scenario}")


def run_seed(seed: int, template: str | None, out_dir: Path, attempt: int,
             scenario: str = "baseline") -> bool:
    cmd = [sys.executable, "-u", str(EPISODE), "--seed", str(seed),
           "--out-dir", str(out_dir), "--scenario", scenario]
    if template:
        cmd += ["--template", template]
    log = out_dir / f"{stem(seed, scenario)}.log"
    print(f"[{time.strftime('%H:%M:%S')}] seed {seed} attempt {attempt} -> {log.name}",
          flush=True)
    with log.open("w") as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT)
    ok = rc == 0
    print(f"[{time.strftime('%H:%M:%S')}] seed {seed} exit={rc} "
          f"{'OK' if ok else 'FAILED'}", flush=True)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True, help="comma-separated, e.g. 2,3,4,5,6")
    ap.add_argument("--template", default=None)
    ap.add_argument("--scenario", default="baseline")
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "rq2_llm_v4"))
    ap.add_argument("--cooldown", type=float, default=90.0,
                    help="seconds to let the rate-limit window drain between "
                         "episodes and before a retry")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    done, failed = [], []

    for seed in seeds:
        result = out_dir / f"{stem(seed, args.scenario)}.json"
        if result.exists() and not json.loads(result.read_text()).get("aborted", True):
            print(f"seed {seed}: already complete, skipping", flush=True)
            done.append(seed)
            continue
        ok = run_seed(seed, args.template, out_dir, 1, args.scenario)
        if not ok:
            print(f"    seed {seed} failed; waiting {args.cooldown}s before retry",
                  flush=True)
            time.sleep(args.cooldown)
            ok = run_seed(seed, args.template, out_dir, 2, args.scenario)
        (done if ok else failed).append(seed)
        if seed != seeds[-1]:
            time.sleep(args.cooldown)
        el = (time.time() - started) / 60
        print(f"    elapsed {el:.0f} min | done {len(done)} | failed {len(failed)}\n",
              flush=True)

    print("=" * 62)
    rows = []
    for seed in done:
        r = json.loads((out_dir / f"{stem(seed, args.scenario)}.json").read_text())
        if r.get("aborted") or r.get("pct_of_oracle") is None:
            failed.append(seed)
            continue
        rows.append((seed, r["pct_of_oracle"], r["gross_profit"],
                     r["usage"]["fallbacks"], r["cost_usd"]))

    print(f"{'seed':>5}{'% of oracle':>13}{'gross profit':>14}{'fallbacks':>11}{'cost':>9}")
    for seed, pct, gp, fb, c in rows:
        print(f"{seed:>5}{pct:>12.2f}%{gp:>14,.1f}{fb:>11}{c:>9.4f}")
    if rows:
        vals = [r[1] for r in rows]
        print(f"\nn={len(vals)}  mean {st.mean(vals):.2f}%  "
              f"min {min(vals):.2f}%  max {max(vals):.2f}%  "
              f"spread {max(vals)-min(vals):.2f} pts"
              + (f"  sd {st.stdev(vals):.2f}" if len(vals) > 1 else ""))
        print(f"total cost ${sum(r[4] for r in rows):.4f}  "
              f"wall {(time.time()-started)/60:.0f} min")
    if failed:
        print(f"\nFAILED (excluded from the mean, not silently dropped): {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
