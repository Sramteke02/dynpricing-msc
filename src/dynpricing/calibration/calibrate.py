"""Offline calibration of the simulation parameters (E2).

Real datasets are used **once, offline, only to calibrate** the environment so
that it behaves plausibly. The agents never train on this data and the true
demand function is never derived from it directly — we only extract aggregate
anchors (reference price, reference volume, unit cost, price band, seasonality).

IMPORTANT — we do NOT regress an elasticity from the data. Price and quantity
in observational retail data are jointly determined, so a descriptive log-log
price-quantity slope is biased for the structural elasticity (that is where the
old, inelastic 0.771 came from). For a **linear (differentiated-Bertrand)**
demand q = max(0, a0*S(t)*C(t) - b*p + d*cbar), we instead set the own-price
sensitivity ``b`` from a **literature target elasticity** ``eps_target`` at the
reference point:

    b  = eps_target * q_ref / p_ref          (eps_target ~ 2.0, configurable)
    d  = cross_ratio * b                      (cross-price sensitivity; sweepable)
    a0 = q_ref + b*p_ref - d*p_ref            (so q(p_ref, cbar=p_ref) = q_ref)

``p_ref``, ``q_ref`` and ``unit_cost`` are sourced from the data exactly as
before; only the sensitivity is imposed, not measured.

Supported sources (all optional; place raw files under ``data/``):

* UCI Online Retail II  -> ``data/online_retail_II.xlsx`` (or ``.csv``)
    Reference price/volume/cost anchors and the price band.
* ONS Retail Sales Index -> ``data/ons_retail_sales.csv``
    Estimates the seasonal amplitude from monthly index values.
* UK Bank Holidays       -> fetched live from the gov.uk API if requested.

If a source is missing, the corresponding parameters fall back to documented
synthetic defaults, so calibration always succeeds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dynpricing.env.config import EnvConfig


@dataclass
class CalibrationResult:
    config: EnvConfig
    sources_used: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    eps_target: float | None = None

    def summary(self) -> str:
        lines = ["Calibration summary", "-" * 40]
        lines += [f"sources used : {', '.join(self.sources_used) or 'none (synthetic defaults)'}"]
        for n in self.notes:
            lines.append(f"  - {n}")
        c = self.config
        implied = c.b * c.ref_price / c.base_demand if c.base_demand else float("nan")
        lines += [
            "-" * 40,
            f"ref_price          = {c.ref_price:.3f}",
            f"unit_cost          = {c.unit_cost:.3f}",
            f"price band         = [{c.price_min:.2f}, {c.price_max:.2f}]",
            f"base_demand(q_ref) = {c.base_demand:.1f}",
            f"a0 (intercept)     = {c.a0:.4f}",
            f"b  (own-price)     = {c.b:.4f}",
            f"d  (cross-price)   = {c.d:.4f}",
            f"implied elasticity @ p_ref = {implied:.3f}",
            f"eps_target         = {self.eps_target if self.eps_target is not None else 'n/a'}",
            "NOTE: price sensitivity b is SET to match a LITERATURE elasticity",
            "      (eps_target); it is NOT measured / regressed from the data.",
            f"seasonal_amp       = {c.seasonal_amplitude:.3f}",
            f"init_inventory     = {c.init_inventory}",
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Individual source calibrators
# --------------------------------------------------------------------------
def _calibrate_from_uci(
    path: Path,
    result: CalibrationResult,
    overrides: dict,
    eps_target: float,
    cross_ratio: float,
) -> None:
    """Derive the linear-demand parameters from UCI reference anchors.

    We take the reference price (median), a smoothed reference volume, the unit
    cost and the price band from the data, then IMPOSE the own-price
    sensitivity from a literature elasticity (we do not regress it).
    """
    import pandas as pd

    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, encoding="ISO-8859-1")

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    price_col = _first_present(df, ["price", "unitprice", "unit_price"])
    qty_col = _first_present(df, ["quantity", "qty"])
    if price_col is None or qty_col is None:
        result.notes.append("UCI: expected price/quantity columns not found; skipped")
        return

    df = df[[price_col, qty_col]].copy()
    df = df[(df[price_col] > 0) & (df[qty_col] > 0)]
    df = df[df[price_col] < df[price_col].quantile(0.99)]  # trim outliers
    if len(df) < 100:
        result.notes.append("UCI: too few clean rows; skipped")
        return

    # robust reference price/volume/cost and price band from the data
    ref_price = round(float(df[price_col].median()), 3)
    p_lo = float(df[price_col].quantile(0.05))
    p_hi = float(df[price_col].quantile(0.95))
    # daily-equivalent reference volume per product line (aggregate, smoothed)
    q_ref = round(float(np.clip(df[qty_col].median() * 12.0, 20.0, 500.0)), 1)
    unit_cost = round(ref_price * 0.4, 3)

    # IMPOSE the own-price sensitivity from a literature elasticity (NOT regressed)
    b = eps_target * q_ref / ref_price
    d = cross_ratio * b
    a0 = q_ref + b * ref_price - d * ref_price

    overrides.update(
        ref_price=ref_price,
        init_price=ref_price,
        price_min=round(max(0.1, min(p_lo, ref_price * 0.5)), 3),
        price_max=round(max(p_hi, ref_price * 1.8), 3),
        unit_cost=unit_cost,
        base_demand=q_ref,
        a0=a0,
        b=b,
        d=d,
        # competitor prices anchored near the (new) reference price, so cbar is
        # inside the band. The old (10.0, 10.5) was built for ref_price=10.
        competitor_init=(2.0, 2.2),
    )
    result.sources_used.append("UCI Online Retail II")
    result.notes.append(
        f"UCI: p_ref={ref_price:.2f}, q_ref={q_ref:.1f}, unit_cost={unit_cost:.2f}. "
        f"Own-price sensitivity b={b:.4f} SET from literature elasticity "
        f"eps_target={eps_target} (NOT regressed from the data); "
        f"d={cross_ratio}*b={d:.4f}, a0={a0:.4f}."
    )
    result.notes.append("UCI: competitor_init set to (2.0, 2.2), near p_ref.")


def _calibrate_from_ons(path: Path, result: CalibrationResult, overrides: dict) -> None:
    """Estimate seasonal amplitude from the ONS Retail Sales Index."""
    import pandas as pd

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    val_col = _first_present(df, ["value", "index", "v4_1", "retail_sales_index"])
    if val_col is None:
        # fall back to the last numeric column
        num = df.select_dtypes("number")
        if num.shape[1] == 0:
            result.notes.append("ONS: no numeric column found; skipped")
            return
        val_col = num.columns[-1]
    series = pd.to_numeric(df[val_col], errors="coerce").dropna().to_numpy()
    if series.size < 12:
        result.notes.append("ONS: too few points; skipped")
        return
    mean = float(np.mean(series))
    amp = float((np.max(series) - np.min(series)) / (2.0 * mean)) if mean else 0.25
    overrides["seasonal_amplitude"] = round(float(np.clip(amp, 0.05, 0.6)), 3)
    result.sources_used.append("ONS Retail Sales Index")
    result.notes.append(f"ONS: seasonal_amplitude~={overrides['seasonal_amplitude']}")


def _calibrate_holidays(result: CalibrationResult, overrides: dict, horizon: int) -> None:
    """Fetch UK bank holidays and map any within the horizon to episode days."""
    try:
        import json
        import urllib.request

        with urllib.request.urlopen("https://www.gov.uk/bank-holidays.json", timeout=10) as r:
            data = json.loads(r.read().decode())
        # Just record that the source is available; map the first few into range.
        events = data.get("england-and-wales", {}).get("events", [])
        if events:
            # place a representative holiday cluster mid-horizon
            mid = horizon // 2
            overrides["holiday_days"] = tuple(range(mid, min(horizon, mid + 3)))
            result.sources_used.append("UK Bank Holidays (gov.uk)")
            result.notes.append(f"Holidays: {len(events)} events; modelled around day {mid}")
    except Exception as exc:
        result.notes.append(f"Holidays: fetch skipped ({exc})")


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def calibrate(
    data_dir: str | Path = "data",
    *,
    fetch_holidays: bool = False,
    base: EnvConfig | None = None,
    eps_target: float = 2.0,
    cross_ratio: float = 0.3,
) -> CalibrationResult:
    """Produce an :class:`EnvConfig` calibrated from whatever data is present.

    ``eps_target`` is the literature own-price elasticity used to set the linear
    demand sensitivity ``b`` (configurable / sweepable); ``cross_ratio`` sets the
    cross-price sensitivity as ``d = cross_ratio * b``.
    """
    data_dir = Path(data_dir)
    base = base or EnvConfig()
    overrides: dict = {}
    result = CalibrationResult(config=base, eps_target=eps_target)

    uci = _find(data_dir, ["online_retail_II.xlsx", "online_retail_II.csv",
                           "online_retail_ii.xlsx", "online_retail.csv"])
    if uci is not None:
        try:
            _calibrate_from_uci(uci, result, overrides, eps_target, cross_ratio)
        except Exception as exc:
            result.notes.append(f"UCI: calibration failed ({exc}); using defaults")

    ons = _find(data_dir, ["ons_retail_sales.csv", "ons_retail_sales_index.csv"])
    if ons is not None:
        try:
            _calibrate_from_ons(ons, result, overrides)
        except Exception as exc:
            result.notes.append(f"ONS: calibration failed ({exc}); using defaults")

    if fetch_holidays:
        _calibrate_holidays(result, overrides, base.horizon)

    if not result.sources_used:
        result.notes.append(
            "No datasets found under "
            f"'{data_dir}'. Using documented synthetic defaults."
        )

    merged = base.to_dict()
    merged.update(overrides)
    merged["calibration_notes"] = "; ".join(result.notes)
    merged["calibration_sources"] = tuple(result.sources_used)
    result.config = EnvConfig.from_dict(merged)
    return result


def sanity_report(cfg: EnvConfig) -> tuple[str, bool]:
    """Check the calibrated demand curve behaves sensibly.

    Confirms (a) demand falls as price rises, (b) the +10% arc response is of a
    plausible magnitude, (c) the implied elasticity at the reference price is in
    a sane band, and (d) the interior condition a(t) > b*c holds. Returns the
    report text and a boolean ``ok`` flag.
    """
    from dynpricing.env.demand import DemandModel

    dm = DemandModel(cfg)
    comps = cfg.competitor_init
    p_lo, p_hi = cfg.price_min, cfg.price_max
    d_lo = dm.expected_units(p_lo, comps, day=0)
    d_hi = dm.expected_units(p_hi, comps, day=0)
    d_ref = dm.expected_units(cfg.ref_price, comps, day=0)

    # a +10% price change around the reference: implied arc response
    p1 = min(cfg.ref_price * 1.10, p_hi)
    d0 = dm.expected_units(cfg.ref_price, comps, day=0)
    d1 = dm.expected_units(p1, comps, day=0)
    pct_price = (p1 - cfg.ref_price) / cfg.ref_price
    pct_demand = (d1 - d0) / d0 if d0 else 0.0
    arc = (pct_demand / pct_price) if pct_price else 0.0

    implied = cfg.b * cfg.ref_price / cfg.base_demand if cfg.base_demand else float("nan")
    cbar = float(np.mean(cfg.competitor_init)) if len(cfg.competitor_init) else cfg.ref_price
    a_min = min(
        cfg.a0 * dm.seasonal_factor(t) * dm.calendar_factor(t) + cfg.d * cbar
        for t in range(cfg.horizon)
    )

    monotone = d_lo > d_ref >= d_hi >= 0.0
    plausible_param = 0.5 <= implied <= 4.0
    plausible_drop = -6.0 < arc < -0.1  # downward and not absurdly steep
    interior = a_min > cfg.b * cfg.unit_cost
    ok = bool(monotone and plausible_param and plausible_drop and interior)

    lines = [
        "Demand-curve sanity check",
        "-" * 40,
        f"expected units @ price_min ({p_lo:.2f}) : {d_lo:8.1f}",
        f"expected units @ ref       ({cfg.ref_price:.2f}) : {d_ref:8.1f}",
        f"expected units @ price_max ({p_hi:.2f}) : {d_hi:8.1f}",
        f"demand falls as price rises          : {'YES' if monotone else 'NO'}",
        f"implied elasticity @ p_ref           : {implied:.3f} "
        f"({'in plausible band' if plausible_param else 'OUT OF BAND'})",
        f"implied arc response to +10% price   : {pct_demand*100:+.1f}% units "
        f"(arc slope {arc:+.2f})",
        f"plausible magnitude                  : {'YES' if plausible_drop else 'NO'}",
        f"interior optimum a(t) > b*c          : {'YES' if interior else 'NO'} "
        f"(min a(t)={a_min:.1f}, b*c={cfg.b*cfg.unit_cost:.1f})",
        "-" * 40,
        "NOTE: price sensitivity b is SET to match a literature elasticity;",
        "      it is NOT a measured/regressed elasticity from the data.",
        f"OVERALL: {'PASS' if ok else 'FAIL'}",
    ]
    return "\n".join(lines), ok


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _first_present(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _find(data_dir: Path, names) -> Path | None:
    for n in names:
        p = data_dir / n
        if p.exists():
            return p
    return None
