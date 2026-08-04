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

## Causal test: `gbm_uniform` (uniform exploration + train-range clamp)

The account above is a *mechanism story*: it says the ratchet is driven by trees
extrapolating flat **above the training range**. That is a causal claim, and it
is testable — remove the extrapolation and the collapse should disappear.

`agents/gbm_uniform_agent.py` is a sibling agent built for exactly this test.
`gbm_agent.py` is untouched, so the broken version stays runnable side by side.
`gbm_uniform` imports the same feature map and the same regressor factory, so the
two agents are identical except for two changes:

1. **Exploration** draws training prices uniformly across `[price_min, price_max]`
   instead of taking random moves from `{hold, +5%, -5%, -10%}`.
2. **Optimiser clamp**: the agent records the min/max training price and never
   selects a price above `train_max`.

Because the action space can only *scale* the current price, no action sequence
can realise an i.i.d. uniform price; during its own data-collection rollouts (and
only there) the agent writes the drawn price into `env.price` and issues `hold`,
which realises it verbatim — a randomised price experiment. Evaluation is
unchanged: `act()` returns ordinary action indices and the harness drives the
environment normally.

### Coverage (seed 0, 40 exploration episodes, band [0.840, 5.972])

| agent | training prices | covers |
|---|---|---:|
| `gbm` (drifting action set) | [0.840, 2.553] | 33% of band, all below the choke 3.150 |
| `gbm_uniform` | [0.841, 5.972] | **100% of band** |

### Headline result (30 seeds, % of oracle)

| scenario | `gbm` | `gbm_uniform` | Δ (paired, 95% CI) |
|---|---:|---:|---|
| baseline | 10.3% | **99.0%** | +16,843 [+15,668, +17,694] |
| high_competition | 8.3% | **96.7%** | +22,199 [+21,079, +22,920] |
| scarce_inventory | 3.9% | **95.0%** | +16,953 [+16,128, +17,427] |
| strong_seasonality | 11.4% | **95.2%** | +18,031 [+16,573, +19,115] |

All four gaps are significant (paired by seed, bootstrap CI excludes 0).
`gbm_uniform` also beats `random` in every scenario (it previously did not) and
beats `fixed` in three of four — it loses `scarce_inventory` by 434 (−2.3%),
where it has no inventory-pacing logic.

The ratchet is gone. On the previously-failing baseline seed 0:

| | `gbm` | `gbm_uniform` |
|---|---|---|
| price range | 1.890 → 5.972 | 1.512 → 2.424 |
| steps at `price_max` | **340/366** | **0/366** |
| steps above the choke (3.150) | 353/366 (first at step 13) | **0/366** |
| settles at | `price_max` 5.972 | ≈2.047 (true `p*` ≈ 2.0) |
| gross profit | 555 | 18,633 |

### Which change did the work? A 2x2 ablation (baseline, 30 seeds)

The clamp only affects `act()`, never training, so each exploration mode is
fitted **once per seed and evaluated twice** (clamp off / clamp on). The two
cells in a row therefore share an identical fitted model and differ by the clamp
alone. `(drift, no clamp)` reproduces the original `gbm` agent **exactly** —
max per-seed |Δ| = 0.0000 — so the sibling is a faithful control.

| variant | profit | % oracle | clamp binds | paired Δ vs original | 95% CI |
|---|---:|---:|---:|---:|---|
| drift-only (**original** `gbm`) | 1,965.9 | 10.3% | — | — | — |
| drift + clamp | 15,473.8 | **81.5%** | 91.70% | +13,508.0 | [+11,909.9, +14,821.2] |
| uniform-only | 18,809.3 | 99.0% | — | +16,843.4 | [+15,668.3, +17,693.9] |
| uniform + clamp (`gbm_uniform`) | 18,809.3 | **99.0%** | **0.00%** | +16,843.4 | [+15,668.3, +17,693.9] |
| oracle (ceiling) | 18,997.1 | 100.0% | — | — | — |

Further paired contrasts (bootstrap 95% CI over the 30 matched seeds):

| contrast | paired Δ | 95% CI | |
|---|---:|---|---|
| drift+clamp − drift-only | +13,508.0 | [+11,909.9, +14,821.2] | significant |
| uniform-only − drift+clamp | +3,335.4 | [+2,619.0, +4,107.0] | significant |
| **uniform-only − (uniform+clamp)** | **+0.0** | **[+0.0, +0.0]** | **identical on 30/30 seeds** |

This decomposes the failure cleanly:

- **Blocking extrapolation alone recovers 71 of the 89 lost points.** The
  clamp-only variant keeps the original's narrow, downward-biased training data
  and changes *nothing* except forbidding prices above `train_max` — and it goes
  10.3% → 81.5%. It binds on **91.7%** of decisions, which is the mechanism made
  visible: with predicted demand flat above the training range, predicted profit
  is monotonically increasing in price, so the myopic optimiser wants "up" almost
  every step and simply pins at the top of its explored range (2.553) instead of
  ratcheting to `price_max` (5.972). Note 2.553 is still *above* the true
  `p*`≈2.0, which is why this route stops at 81.5% rather than at the ceiling.
- **Fixing coverage supplies the remaining 17.6 points** (+3,335.4, CI excludes
  0), and makes the clamp inert: with training data spanning the band,
  `train_max == price_max`, so no reachable price can exceed it and the clamp
  binds **0/365** decisions and changes **no** decision on **any** of the 30
  seeds. The recovery there is not the clamp — it is that the learned demand
  curve now *turns over* inside the covered region, so the optimiser stops at
  ≈2.05 ≈ `p*` of its own accord.

**Verdict: the extrapolation diagnosis is confirmed causally.** Removing
extrapolation past the training range recovers the collapse by *either* route,
independently: censoring the optimiser at the training boundary (10.3% → 81.5%)
or eliminating the boundary by covering the band (10.3% → 99.0%). The two routes
attack the same thing, which is why they do not compose — applied together the
clamp adds exactly nothing (Δ = 0.0 on 30/30 seeds).

A note on how *not* to read this ablation. One might expect a clean dissociation
in which the clamp fails and only exploration works, treating a clamp recovery as
evidence *against* the extrapolation story. That would be the wrong test: the
clamp's sole effect is to forbid prices above `train_max`, so under the
extrapolation diagnosis it is *predicted* to help. What the diagnosis forbids is
the opposite pattern — neither intervention helping, or the collapse persisting
once the model is never queried outside its training support. Both routes
removing the collapse is the confirmation, not a confound.

### Correction to the earlier verdict

The band-reconciliation update above concluded the pathology was "**fundamental
to the agent**". That was too strong, and this result corrects it. What is
fundamental is the *interaction* of a myopic one-step optimiser with a
**downward-biased exploration policy**: the action set `{hold, +5%, -5%, -10%}`
has geometric drift −2.66%/step, so on-policy exploration never visits the upper
band, and boosted trees are then asked to extrapolate exactly where they cannot.
Predict-then-optimise itself is fine here — with band-wide exploration the same
model, features and optimiser reach 95–99% of oracle. The finding is a finding
about **exploration design**, not about gradient boosting.

The original `gbm` agent and its `xfail`-ed test are deliberately left in place:
the contrast is the result.

## Fixes deliberately NOT applied to `gbm_agent.py` (the original stays broken)

- Clamping the GBM optimiser to its explored price range.
- Widening exploration or de-biasing the action set.
- Increasing the test seed count (`test_agents.py`).

Both of the first two are now implemented in the **separate** `gbm_uniform`
agent (above), so the pathology remains reproducible in `gbm` for comparison
while the causal test is available alongside it.

(The band **has** now been reconciled in `calibration/calibrate.py` — see the
update above — but that was to fix the band itself, and it does not resolve the
GBM pathology.)
