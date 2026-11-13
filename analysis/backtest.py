"""Rolling-origin backtest of LightGBM + CQR on the hub dengue series.

Features include lag-1 ENSO/ERA5 climate (see models.lgbm_cqr.build_features).
The target remains the hub case series.

Default evaluation window: monthly origins from 2023-09 through the last
origin that still has lead-5 observed (2026-04 given last obs 2026-09).
Raw forecasts begin earlier so the CQR window is warm by the first
evaluation origin.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from models.hub import QUANTILES
from models.lgbm_cqr import (
    CALIB_W,
    MODEL_LEADS,
    ROOT,
    apply_cqr_to_frame,
    build_features,
    collect_rolling_raw,
)
from models.hub import load_dengue_series

DATA_PATH = ROOT / "data" / "time-series.csv"
OUT_DIR = ROOT / "analysis"
QCOLS = [f"q_{q}" for q in QUANTILES]
CQR_COLS = [f"cqr_{q}" for q in QUANTILES]


def interval_score(y: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> np.ndarray:
    return (upper - lower) + (2.0 / alpha) * np.maximum(lower - y, 0.0) + (2.0 / alpha) * np.maximum(y - upper, 0.0)


def weighted_interval_score(y: np.ndarray, qmat: np.ndarray, quantiles: tuple[float, ...] = QUANTILES) -> np.ndarray:
    """Bracher et al. (2021) WIS over the 23 hub quantile levels."""
    q_to_idx = {float(q): i for i, q in enumerate(quantiles)}
    median = qmat[:, q_to_idx[0.5]]
    total = 0.5 * np.abs(y - median)
    k = 0
    for q in quantiles:
        q = float(q)
        if q >= 0.5:
            continue
        alpha = 2.0 * q
        lo = qmat[:, q_to_idx[q]]
        hi = qmat[:, q_to_idx[round(1.0 - q, 10)]]
        total = total + (alpha / 2.0) * interval_score(y, lo, hi, alpha)
        k += 1
    return total / (k + 0.5)


def metrics_table(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """prefix is 'q' (raw) or 'cqr'."""
    qcols = [f"{prefix}_{q}" for q in QUANTILES]
    rows = []
    parts = [("all", df)]
    for lead, part in df.groupby("lead"):
        parts.append((f"lead_{int(lead)}", part))
    for month in sorted(df["target_end_date"].dt.month.unique()):
        part = df[df["target_end_date"].dt.month == month]
        if len(part) >= 8:
            parts.append((f"month_{int(month):02d}", part))
    for name, part in parts:
        if part.empty or part["y"].isna().all():
            continue
        y = part["y"].to_numpy(dtype=float)
        qmat = part[qcols].to_numpy(dtype=float)
        mid = qmat[:, list(QUANTILES).index(0.5)]
        wis = weighted_interval_score(y, qmat)
        cov50 = ((y >= qmat[:, list(QUANTILES).index(0.25)]) & (y <= qmat[:, list(QUANTILES).index(0.75)])).mean()
        cov90 = ((y >= qmat[:, list(QUANTILES).index(0.05)]) & (y <= qmat[:, list(QUANTILES).index(0.95)])).mean()
        cov95 = ((y >= qmat[:, list(QUANTILES).index(0.025)]) & (y <= qmat[:, list(QUANTILES).index(0.975)])).mean()
        rows.append(
            {
                "subset": name,
                "model": prefix,
                "n": int(len(part)),
                "WIS": float(np.mean(wis)),
                "MAE": float(np.mean(np.abs(y - mid))),
                "cov50": float(cov50),
                "cov90": float(cov90),
                "cov95": float(cov95),
                "median_pred_mean": float(np.mean(mid)),
                "y_mean": float(np.mean(y)),
            }
        )
    return pd.DataFrame(rows)


def default_origins(y: pd.Series) -> tuple[list[pd.Timestamp], pd.Timestamp, pd.Timestamp]:
    last_obs = pd.Timestamp(y.index.max())
    # Last origin with lead 5 observed.
    eval_end = last_obs - pd.DateOffset(months=max(MODEL_LEADS))
    eval_start = pd.Timestamp("2023-09-01")
    # Warm-up so the first eval origin has a full CQR window.
    gen_start = eval_start - pd.DateOffset(months=CALIB_W)
    return (
        [t for t in y.index if gen_start <= t <= last_obs - pd.DateOffset(months=1)],
        eval_start,
        eval_end,
    )


def main() -> None:
    y = load_dengue_series(DATA_PATH)
    feat = build_features(y)
    gen_origins, eval_start, eval_end = default_origins(y)
    # Only origins with complete features.
    gen_origins = [t for t in gen_origins if t in feat.dropna().index]
    print(
        f"Backtest raw forecasts: {len(gen_origins)} origins "
        f"{gen_origins[0].date()} .. {gen_origins[-1].date()}"
    )
    print(f"Evaluation origins (all leads observed): {eval_start.date()} .. {eval_end.date()}")
    raw = collect_rolling_raw(y, feat, gen_origins, leads=MODEL_LEADS)
    cqr = apply_cqr_to_frame(raw)
    observed = cqr[cqr["y"].notna()].copy()
    # Rectangular window: every lead is observed.
    rect = observed[(observed["origin"] >= eval_start) & (observed["origin"] <= eval_end)].copy()
    print(f"Rectangular evaluation rows: {len(rect)}")

    raw_m = metrics_table(rect, "q")
    cqr_m = metrics_table(rect, "cqr")
    table = pd.concat([cqr_m, raw_m], ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = OUT_DIR / "backtest_predictions.csv"
    met_path = OUT_DIR / "backtest_metrics.csv"
    cqr.to_csv(pred_path, index=False, encoding="utf-8")
    table.to_csv(met_path, index=False, encoding="utf-8")
    # Compact print.
    show = table[table["subset"].isin(["all"] + [f"lead_{h}" for h in MODEL_LEADS])]
    print("\nMetrics (rectangular window):")
    with pd.option_context("display.max_columns", 20, "display.width", 140, "display.float_format", "{:.3f}".format):
        print(show.to_string(index=False))
    print(f"\nWrote {met_path}")
    print(f"Wrote {pred_path}")


if __name__ == "__main__":
    main()
