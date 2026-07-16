# Why `test_gbm_trains_and_beats_random` xfails under linear demand

`tests/test_agents.py::test_gbm_trains_and_beats_random` asserts the GBM
predict-then-optimise agent beats random pricing. Under the new linear
(differentiated-Bertrand) demand it fails at its 5-seed sample (it still passes
on average at 10 seeds). This is a **real finding**, not a flaky assertion, so it
is marked `xfail` rather than hidden by widening the seed count or clamping the
optimiser.

## Mechanism (confirmed on the calibrated config, ref_price=2.10)

1. **The demand band is mostly dead.** With `a0=86.4, b=34.2857, d=10.2857`, the
   choke price at S=1, cbar=2.10 is `a/b = 108/34.2857 = 3.150`. But
   `price_max = 9.95` (UCI's 95th-percentile observed price, see
   `calibration/calibrate.py`: `price_max = max(p_hi, ref_price*1.8)`) was never
   reconciled with the demand parameters. Result: `q(9.95) = 0` exactly, and only
   **28.65%** of `[0.42, 9.95]` has `q>0` (24.24% yields positive profit). ~71% of
   the band is economically dead.

2. **Exploration is downward-biased.** The action multipliers
   `{1.00, 1.05, 0.95, 0.90}` have arithmetic mean 0.975 and geometric drift
   **-2.66%/step**. The GBM explores with random actions from `init_price`, so its
   training data concentrates at low prices: on failing seed 0 the training prices
   span **[0.420, 2.431]** (median 0.577).

3. **Boosted trees extrapolate flat.** Above the training range the fitted model
   predicts a constant ~30.1 units at 2.63 / 4.46 / 6.29 / 8.12 / 9.95, where the
   true demand is **0**.

4. **The myopic optimiser ratchets into the dead zone.** With predicted demand
   flat, predicted profit `(p - c) * 30.1` increases in `p`, so the one-step
   optimiser keeps raising price past the choke (3.15) up to `price_max` 9.95,
   where realised demand — and profit — is ~0. On seed 0 it holds p*≈1.995 for
   ~18 steps, then climbs monotonically and crosses the choke at step 28, after
   which cumulative profit flatlines. The optimiser selects prices **7.52 above**
   the training max.

The identical mechanism drives the failure on the test's own default config
(`EnvConfig(horizon=40, init_inventory=20000)`, band `[5, 20]`, choke ≈15.08).

## What this is (and is not)

- It is **not** a bug in the linear demand model: p*(t) is correct and the oracle
  exploits it.
- It **is** a genuine weakness of myopic predict-then-optimise when (i) exploration
  under-covers the profitable region and (ii) the action/price band extends well
  past the demand choke. The more-discriminating demand simply exposes it.

## Update: tested against a reconciled band (does the band explain it?)

The original write-up flagged the over-wide band (choke 3.15 vs `price_max` 9.95,
~71% dead) as part of the mechanism, which could be read as implying the band was
a major driver of the severity. We tested that directly by reconciling the band
in `calibration/calibrate.py` (`price_min = unit_cost`;
`price_max = 1.2 * max_t choke(t)` → **[0.84, 5.97]**, dead zone 52.2% → 19.4%)
and re-running 6 agents × 30 seeds × all scenarios.

Result — the pathology is **still material**:

| scenario | GBM % of oracle (wide band) | GBM % of oracle (reconciled band) |
|---|---:|---:|
| baseline | 8.6% | 10.3% |
| high_competition | 6.1% | 8.3% |
| scarce_inventory | 1.8% | 3.9% |
| strong_seasonality | 9.6% | 11.4% |

GBM recovers only a few points and **still ratchets to `price_max`**: on the
previously-failing seed 0 it climbs monotonically from 1.89, crosses the choke
(3.15) at step 12, reaches `price_max` (5.97), and sits in the dead zone.

**Verdict (committed):** the tree-extrapolation pathology is **fundamental to the
agent**, not an artifact of the band. Downward-biased exploration + flat tree
extrapolation + a myopic one-step optimiser make GBM ratchet past *whatever* choke
exists, up to *whatever* `price_max` exists. The wide band **amplified** the
severity (a larger dead zone to sit in → GBM as low as 1.8%), but reconciling it
recovered only ~2–3 points and GBM still collapses to single digits. So the band
was a compounding amplifier, not the cause — and this *strengthens* the original
claim (the weakness survives band reconciliation) while correcting its emphasis:
the band mattered less than the write-up implied.

Separately, reconciling the band did fix a different agent: **random** is no longer
catastrophically negative (−18…−52% → +7.6…9.8% of oracle), because
`price_min = unit_cost` removes below-cost pricing.

## Fixes deliberately NOT applied here (out of scope; would hide the GBM finding)

- Clamping the GBM optimiser to its explored price range (`gbm_agent.py`).
- Widening exploration or de-biasing the action set (`gbm_agent.py`).
- Increasing the test seed count (`test_agents.py`).

(The band **has** now been reconciled in `calibration/calibrate.py` — see the
update above — but that was to fix the band itself, and it does not resolve the
GBM pathology.)
