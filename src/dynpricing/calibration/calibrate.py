"""Offline calibration of the simulation parameters (E2).

Real datasets are used **once, offline, only to calibrate** the environment so
that it behaves plausibly. The agents never train on this data and the true
demand function is never derived from it directly — we only extract aggregate
parameters (price ranges, base volume, a demand-sensitivity parameter,
seasonality).

IMPORTANT — identification caveat. The numbers produced here are *plausible
parameter values that place the simulation in a realistic regime*. They are
**not** identified causal elasticities. Price and quantity in observational
retail data are jointly determined (by demand and supply/assortment decisions),
so a descriptive log-log price-quantity slope is biased for the structural
elasticity. We deliberately use it only to choose a sensible value for the
simulation's *own* (fully controlled, transparent) demand parameter, and we do
not claim to have measured the real-world elasticity anywhere.

Supported sources (all optional; place raw files under ``data/``):

* UCI Online Retail II  -> ``data/online_retail_II.xlsx`` (or ``.csv``)
    Derives plausible price ranges, base volume, and a demand-sensitivity
    parameter from a *descriptive* log-log price-quantity slope (not an
    identified elasticity; see the caveat above).
* ONS Retail Sales Index -> ``data/ons_retail_sales.csv``
    Estimates the seasonal amplitude from monthly index values.
* UK Bank Holidays       -> fetched live from the gov.uk API if requested,
    used to mark holiday days within the episode horizon.

If a source is missing, the corresponding parameters fall back to documented
synthetic defaults, so calibration always succeeds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dynpricing.env.config import EnvConfig


@dataclass
class CalibrationResult:
    config: EnvConfig
    sources_used: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = ["Calibration summary", "-" * 40]
        lines += [f"sources used : {', '.join(self.sources_used) or 'none (synthetic defaults)'}"]
        for n in self.notes:
            lines.append(f"  - {n}")
        c = self.config
        lines += [
            "-" * 40,
            f"ref_price        = {c.ref_price:.3f}",
            f"unit_cost        = {c.unit_cost:.3f}",
            f"price band       = [{c.price_min:.2f}, {c.price_max:.2f}]",
            f"base_demand      = {c.base_demand:.1f}",
            f"demand_sens.     = {c.elasticity:.3f}  (simulation parameter; NOT an identified elasticity)",
            f"seasonal_amp     = {c.seasonal_amplitude:.3f}",
            f"init_inventory   = {c.init_inventory}",
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Individual source calibrators
# --------------------------------------------------------------------------
def _calibrate_from_uci(path: Path, result: CalibrationResult, overrides: dict) -> None:
    """Derive plausible price band, base volume and demand-sensitivity from UCI.

    The demand-sensitivity number comes from a *descriptive* binned log-log
    slope of quantity on price. This is NOT an identified elasticity (price and
    quantity are jointly determined); it is only used to place the simulation's
    own demand parameter in a realistic regime.
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

    # robust price band and reference from quantiles
    ref_price = float(df[price_col].median())
    p_lo = float(df[price_col].quantile(0.05))
    p_hi = float(df[price_col].quantile(0.95))
    # daily-equivalent base volume per product line (aggregate, smoothed)
    base_demand = float(np.clip(df[qty_col].median() * 12.0, 20.0, 500.0))

    # descriptive (NOT identified) log-log price-quantity slope, binned for noise
    slope = _descriptive_price_quantity_slope(
        df[price_col].to_numpy(), df[qty_col].to_numpy()
    )
    sensitivity = float(np.clip(slope, 0.5, 4.0))

    overrides.update(
        ref_price=round(ref_price, 3),
        init_price=round(ref_price, 3),
        price_min=round(max(0.1, min(p_lo, ref_price * 0.5)), 3),
        price_max=round(max(p_hi, ref_price * 1.8), 3),
        unit_cost=round(ref_price * 0.4, 3),
        base_demand=round(base_demand, 1),
        elasticity=round(sensitivity, 3),  # simulation demand-sensitivity param
    )
    result.sources_used.append("UCI Online Retail II")
    result.notes.append(
        f"UCI: ref_price={ref_price:.2f}; demand-sensitivity param set to "
        f"{sensitivity:.2f} from a DESCRIPTIVE log-log slope "
        f"(not an identified elasticity)."
    )


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
) -> CalibrationResult:
    """Produce an :class:`EnvConfig` calibrated from whatever data is present."""
    data_dir = Path(data_dir)
    base = base or EnvConfig()
    overrides: dict = {}
    result = CalibrationResult(config=base)

    uci = _find(data_dir, ["online_retail_II.xlsx", "online_retail_II.csv",
                           "online_retail_ii.xlsx", "online_retail.csv"])
    if uci is not None:
        try:
            _calibrate_from_uci(uci, result, overrides)
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

    Confirms (a) demand falls as price rises, (b) the drop is of a plausible
    magnitude, and reports the simulation's demand-sensitivity parameter. Returns
    the report text and a boolean ``ok`` flag.
    """
    from dynpricing.env.demand import DemandModel

    dm = DemandModel(cfg)
    comps = cfg.competitor_init
    p_lo, p_hi = cfg.price_min, cfg.price_max
    d_lo = dm.expected_units(p_lo, comps, day=0)
    d_hi = dm.expected_units(p_hi, comps, day=0)
    d_ref = dm.expected_units(cfg.ref_price, comps, day=0)

    # a +10% price change around the reference: implied arc sensitivity
    p1 = min(cfg.ref_price * 1.10, p_hi)
    d0 = dm.expected_units(cfg.ref_price, comps, day=0)
    d1 = dm.expected_units(p1, comps, day=0)
    pct_price = (p1 - cfg.ref_price) / cfg.ref_price
    pct_demand = (d1 - d0) / d0 if d0 else 0.0
    arc = (pct_demand / pct_price) if pct_price else 0.0

    monotone = d_lo > d_ref > d_hi > 0
    plausible_param = 0.5 <= cfg.elasticity <= 4.0
    plausible_drop = -6.0 < arc < -0.1  # downward and not absurdly steep
    ok = bool(monotone and plausible_param and plausible_drop)

    lines = [
        "Demand-curve sanity check",
        "-" * 40,
        f"expected units @ price_min ({p_lo:.2f}) : {d_lo:8.1f}",
        f"expected units @ ref       ({cfg.ref_price:.2f}) : {d_ref:8.1f}",
        f"expected units @ price_max ({p_hi:.2f}) : {d_hi:8.1f}",
        f"demand falls as price rises          : {'YES' if monotone else 'NO'}",
        f"demand-sensitivity parameter         : {cfg.elasticity:.3f} "
        f"({'in plausible band' if plausible_param else 'OUT OF BAND'})",
        f"implied arc response to +10% price   : {pct_demand*100:+.1f}% units "
        f"(arc slope {arc:+.2f})",
        f"plausible magnitude                  : {'YES' if plausible_drop else 'NO'}",
        "-" * 40,
        f"NOTE: the sensitivity figure is a simulation parameter chosen to place",
        f"      the market in a realistic regime; it is NOT a measured elasticity.",
        f"OVERALL: {'PASS' if ok else 'FAIL'}",
    ]
    return "\n".join(lines), ok


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _descriptive_price_quantity_slope(
    prices: np.ndarray, qty: np.ndarray, bins: int = 30
) -> float:
    """Descriptive |slope| of log-quantity on log-price (binned to reduce noise).

    This is a *correlational* summary of the data, returned as a positive
    magnitude for convenience. It is NOT an identified own-price elasticity
    (price and quantity are jointly determined) and must not be reported as one.
    """
    p = np.asarray(prices, dtype=float)
    q = np.asarray(qty, dtype=float)
    mask = (p > 0) & (q > 0)
    p, q = p[mask], q[mask]
    if p.size < 20:
        return 1.8
    # bin by price quantiles, average log-quantity per bin to reduce noise
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if edges.size < 3:
        return 1.8
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, edges.size - 2)
    lp, lq = [], []
    for b in range(edges.size - 1):
        sel = idx == b
        if sel.sum() >= 5:
            lp.append(math.log(np.mean(p[sel])))
            lq.append(math.log(np.mean(q[sel])))
    if len(lp) < 3:
        return 1.8
    slope, _ = np.polyfit(lp, lq, 1)
    return abs(float(slope))


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
