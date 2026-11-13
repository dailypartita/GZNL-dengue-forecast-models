"""LightGBM quantile regression with sliding-window conformalized quantile regression.

Train one LightGBM quantile model per (horizon lead, hub quantile) on log1p
counts, using lag and calendar features available at the origin month (no
leakage).  After raw quantile prediction, apply online CQR on every
complementary pair (q, 1-q), then enforce monotonicity, non-negativity, and
integer counts.
"""

from __future__ import annotations

import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from models.hub import (
    HUB_HORIZONS,
    MODEL_ID,
    MODEL_LEADS,
    QUANTILES,
    REFERENCE_DATE,
    TARGET,
    hub_horizon_to_lead,
    load_dengue_series,
    target_end_date,
    write_hub_file,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "time-series.csv"
OUTPUT_DIR = ROOT / "output" / MODEL_ID

FEATURE_COLS = (
    "lag1",
    "lag2",
    "lag3",
    "lag6",
    "lag12",
    "r3m",
    "r12m",
    "d1",
    "cal_sin1",
    "cal_cos1",
)

CALIB_W = 24
BURN_IN = 10

# LightGBM settings taken from the original backtest (extended to 23 quantiles).
LGBM_BASE = dict(
    objective="quantile",
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=15,
    min_child_samples=5,
    verbose=-1,
    random_state=42,
    n_jobs=1,
    deterministic=True,
    force_row_wise=True,
)


def shift_months(dates, months: int) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(dates) + pd.DateOffset(months=int(months))


def build_features(y: pd.Series) -> pd.DataFrame:
    """Features at origin month t, using only observations up to t.

    lag_k is the k-th most recent observation (lag1 = y_t, lag12 = y_{t-11}).
    r3m / r12m are trailing means of the last 3 / 12 observed months.
    d1 = lag1 - lag2.  Fourier terms use the origin calendar month.
    """
    y = y.astype(float).sort_index()
    feat = pd.DataFrame(index=y.index)
    feat["lag1"] = y
    feat["lag2"] = y.shift(1)
    feat["lag3"] = y.shift(2)
    feat["lag6"] = y.shift(5)
    feat["lag12"] = y.shift(11)
    feat["r3m"] = y.rolling(window=3, min_periods=3).mean()
    feat["r12m"] = y.rolling(window=12, min_periods=12).mean()
    feat["d1"] = feat["lag1"] - feat["lag2"]
    month = feat.index.month.astype(float)
    feat["cal_sin1"] = np.sin(2.0 * np.pi * month / 12.0)
    feat["cal_cos1"] = np.cos(2.0 * np.pi * month / 12.0)
    return feat[list(FEATURE_COLS)]


def _training_xy(
    feat: pd.DataFrame,
    y: pd.Series,
    origin: pd.Timestamp,
    lead: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y_target) for origins s where y_{s+lead} is known by `origin`."""
    origin = pd.Timestamp(origin)
    complete = feat.dropna()
    complete = complete.loc[complete.index <= origin]
    if complete.empty:
        return complete, pd.Series(dtype=float)
    target_index = shift_months(complete.index, lead)
    y_target = pd.Series(y.reindex(target_index).to_numpy(), index=complete.index)
    observed = y_target.notna() & (target_index <= origin)
    X = complete.loc[observed]
    y_out = y_target.loc[observed]
    return X, y_out


def _fit_one(X: pd.DataFrame, y_log: np.ndarray, q: float) -> lgb.Booster:
    # Native LightGBM API (no scikit-learn dependency).
    params = {
        "objective": "quantile",
        "alpha": float(q),
        "learning_rate": LGBM_BASE["learning_rate"],
        "num_leaves": LGBM_BASE["num_leaves"],
        "min_child_samples": LGBM_BASE["min_child_samples"],
        "verbosity": -1,
        "seed": LGBM_BASE["random_state"],
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 1,
        "feature_pre_filter": False,
    }
    dtrain = lgb.Dataset(X, label=y_log, feature_name=list(X.columns), free_raw_data=False)
    return lgb.train(params, dtrain, num_boost_round=int(LGBM_BASE["n_estimators"]))


def predict_raw_quantiles(
    feat: pd.DataFrame,
    y: pd.Series,
    origin: pd.Timestamp,
    leads: tuple[int, ...] = MODEL_LEADS,
    quantiles: tuple[float, ...] = QUANTILES,
) -> dict[int, np.ndarray]:
    """Train per-lead, per-quantile LightGBM models at `origin` and predict.

    Returns lead -> raw (pre-CQR) quantile vector on the count scale, already
    expm1-transformed, clipped at 0, and made monotone / non-negative (but not
    yet rounded or conformalized).
    """
    origin = pd.Timestamp(origin)
    x_row = feat.loc[[origin], list(FEATURE_COLS)]
    if x_row.isna().any().any():
        raise ValueError(f"incomplete features at origin {origin.date()}")
    out: dict[int, np.ndarray] = {}
    for lead in leads:
        X, y_tgt = _training_xy(feat, y, origin, lead)
        if len(X) < 20:
            raise RuntimeError(f"not enough training rows at {origin.date()} lead={lead}: {len(X)}")
        y_log = np.log1p(np.clip(y_tgt.to_numpy(dtype=float), 0, None))
        preds = []
        for q in quantiles:
            model = _fit_one(X, y_log, q)
            pred = float(np.expm1(model.predict(x_row)[0]))
            preds.append(max(pred, 0.0))
        out[int(lead)] = enforce_monotone(np.asarray(preds, dtype=float))
    return out


def enforce_monotone(values: np.ndarray) -> np.ndarray:
    """Non-negative and non-decreasing along the quantile axis."""
    arr = np.maximum(np.asarray(values, dtype=float), 0.0)
    return np.maximum.accumulate(arr)


def conformal_adjust(
    pred: np.ndarray,
    calib_preds: np.ndarray,
    calib_y: np.ndarray,
    quantiles: tuple[float, ...] = QUANTILES,
) -> np.ndarray:
    """CQR on all complementary pairs (q, 1-q).  The median is not moved."""
    pred = np.asarray(pred, dtype=float).copy()
    calib_preds = np.atleast_2d(np.asarray(calib_preds, dtype=float))
    calib_y = np.asarray(calib_y, dtype=float)
    if calib_preds.shape[0] != calib_y.shape[0]:
        raise ValueError("calib_preds and calib_y length mismatch")
    n = int(calib_y.shape[0])
    if n == 0:
        return pred

    q_to_idx = {float(q): i for i, q in enumerate(quantiles)}
    median = pred[q_to_idx[0.5]]
    out = pred.copy()
    for q in quantiles:
        q = float(q)
        if q >= 0.5:
            continue
        alpha = 2.0 * q
        i_lo = q_to_idx[q]
        i_hi = q_to_idx[round(1.0 - q, 10)]
        scores = np.maximum(calib_preds[:, i_lo] - calib_y, calib_y - calib_preds[:, i_hi])
        level = math.ceil((n + 1) * (1.0 - alpha)) / n
        Q = float(np.quantile(scores, min(1.0, level), method="higher"))
        out[i_lo] = pred[i_lo] - Q
        out[i_hi] = pred[i_hi] + Q

    # Clip lower bound at 0; do not move the median.
    out = np.maximum(out, 0.0)
    mid = q_to_idx[0.5]
    out[mid] = median
    out[:mid] = np.minimum(out[:mid], median)
    out[mid + 1 :] = np.maximum(out[mid + 1 :], median)
    out = enforce_monotone(out)
    out[mid] = median
    return out


def finalize_quantiles(values: np.ndarray) -> np.ndarray:
    """Monotone, non-negative integers (round after the monotone fix)."""
    arr = enforce_monotone(values)
    rounded = np.rint(arr).astype(int)
    rounded = np.maximum(rounded, 0)
    return np.maximum.accumulate(rounded)


def collect_rolling_raw(
    y: pd.Series,
    feat: pd.DataFrame,
    origins: list[pd.Timestamp],
    leads: tuple[int, ...] = MODEL_LEADS,
    quantiles: tuple[float, ...] = QUANTILES,
) -> pd.DataFrame:
    """Fit at each origin and return raw quantile forecasts (plus outcome if known)."""
    last_obs = y.index.max()
    rows = []
    n = len(origins)
    for i, origin in enumerate(origins):
        origin = pd.Timestamp(origin)
        raw = predict_raw_quantiles(feat, y, origin, leads=leads, quantiles=quantiles)
        for lead, qvec in raw.items():
            target_date = shift_months(pd.DatetimeIndex([origin]), lead)[0]
            y_true = float(y.loc[target_date]) if target_date in y.index and target_date <= last_obs else np.nan
            rec = {
                "origin": origin,
                "lead": int(lead),
                "target_end_date": target_date,
                "y": y_true,
            }
            for q, v in zip(quantiles, qvec):
                rec[f"q_{q}"] = float(v)
            rows.append(rec)
        if (i + 1) % 5 == 0 or i == n - 1:
            print(f"  raw forecasts {i + 1}/{n}  origin={origin.date()}", flush=True)
    return pd.DataFrame(rows)


def apply_cqr_to_frame(
    raw_df: pd.DataFrame,
    quantiles: tuple[float, ...] = QUANTILES,
    calib_w: int = CALIB_W,
    burn_in: int = BURN_IN,
) -> pd.DataFrame:
    """Add CQR-adjusted integer quantiles using a causal sliding window per lead."""
    qcols = [f"q_{q}" for q in quantiles]
    out_rows = []
    for lead, part in raw_df.groupby("lead", sort=True):
        part = part.sort_values("origin").reset_index(drop=True)
        history_preds: list[np.ndarray] = []
        history_y: list[float] = []
        history_origins: list[pd.Timestamp] = []
        for _, rec in part.iterrows():
            pred = rec[qcols].to_numpy(dtype=float)
            # Use only past origins whose outcome is already observed by this origin.
            origin = pd.Timestamp(rec["origin"])
            usable = []
            for o, p, yy in zip(history_origins, history_preds, history_y):
                if pd.isna(yy):
                    continue
                outcome_date = shift_months(pd.DatetimeIndex([o]), int(lead))[0]
                if outcome_date <= origin:
                    usable.append((o, p, yy))
            usable = usable[-calib_w:]
            if len(usable) >= burn_in:
                calib_preds = np.vstack([u[1] for u in usable])
                calib_y = np.asarray([u[2] for u in usable], dtype=float)
                adjusted = conformal_adjust(pred, calib_preds, calib_y, quantiles)
            else:
                adjusted = enforce_monotone(pred)
            final = finalize_quantiles(adjusted)
            new = rec.to_dict()
            for q, v in zip(quantiles, final):
                new[f"cqr_{q}"] = int(v)
            new["n_calib"] = len(usable)
            out_rows.append(new)
            history_origins.append(origin)
            history_preds.append(pred)
            history_y.append(rec["y"])
    return pd.DataFrame(out_rows)


def build_submission(
    final_row_by_lead: dict[int, np.ndarray],
    last_obs: pd.Timestamp,
    reference_date: pd.Timestamp = REFERENCE_DATE,
    quantiles: tuple[float, ...] = QUANTILES,
) -> pd.DataFrame:
    records = []
    ref = pd.Timestamp(reference_date)
    for h in HUB_HORIZONS:
        lead = hub_horizon_to_lead(h, last_obs, ref)
        if lead not in final_row_by_lead:
            raise KeyError(f"missing forecast for hub horizon {h} (model lead {lead})")
        qvec = np.asarray(final_row_by_lead[lead], dtype=int)
        ted = target_end_date(ref, h)
        for q, v in zip(quantiles, qvec):
            records.append(
                {
                    "reference_date": ref.strftime("%Y-%m-%d"),
                    "target": TARGET,
                    "horizon": int(h),
                    "target_end_date": ted.strftime("%Y-%m-%d"),
                    "location": "CN",
                    "output_type": "quantile",
                    "output_type_id": q,
                    "value": int(v),
                }
            )
    return pd.DataFrame.from_records(records)


def forecast_round(
    y: pd.Series,
    feat: pd.DataFrame | None = None,
    reference_date: pd.Timestamp = REFERENCE_DATE,
    calib_w: int = CALIB_W,
    burn_in: int = BURN_IN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Produce the 2026-11-01 (or given) dengue submission plus the rolling table."""
    if feat is None:
        feat = build_features(y)
    last_obs = pd.Timestamp(y.index.max())
    leads = MODEL_LEADS
    # Need CALIB_W observed outcomes for the longest lead, plus a little slack
    # so shorter leads also have a full window ending at last_obs - lead.
    max_lead = max(leads)
    first_needed = last_obs - pd.DateOffset(months=max_lead + calib_w - 1)
    # Origins with complete features, up to last_obs (final forecast origin).
    candidate = [t for t in feat.dropna().index if first_needed <= t <= last_obs]
    if last_obs not in candidate:
        raise ValueError(f"last observation {last_obs.date()} has incomplete features")
    print(
        f"Rolling raw forecasts for {len(candidate)} origins "
        f"{candidate[0].date()} .. {candidate[-1].date()} (last obs {last_obs.date()})",
        flush=True,
    )
    raw = collect_rolling_raw(y, feat, candidate, leads=leads)
    cqr = apply_cqr_to_frame(raw, calib_w=calib_w, burn_in=burn_in)
    final_part = cqr[cqr["origin"] == last_obs]
    final_by_lead = {}
    for lead in leads:
        row = final_part[final_part["lead"] == lead]
        if row.empty:
            raise RuntimeError(f"no CQR forecast at origin {last_obs.date()} lead={lead}")
        final_by_lead[int(lead)] = row.iloc[0][[f"cqr_{q}" for q in QUANTILES]].to_numpy(dtype=int)
        n_cal = int(row.iloc[0]["n_calib"])
        print(f"  final lead={lead} n_calib={n_cal} q50={int(final_by_lead[int(lead)][QUANTILES.index(0.5)])}")
    sub = build_submission(final_by_lead, last_obs=last_obs, reference_date=reference_date)
    return sub, cqr


def main() -> None:
    print(f"Loading {DATA_PATH}")
    y = load_dengue_series(DATA_PATH)
    print(f"Dengue series: {y.index.min().date()} .. {y.index.max().date()}  n={len(y)}")
    expected_tail = {
        pd.Timestamp("2026-01-01"): 93.0,
        pd.Timestamp("2026-02-01"): 94.0,
        pd.Timestamp("2026-03-01"): 92.0,
        pd.Timestamp("2026-04-01"): 137.0,
        pd.Timestamp("2026-05-01"): 222.0,
        pd.Timestamp("2026-06-01"): 255.0,
        pd.Timestamp("2026-07-01"): 1120.0,
        pd.Timestamp("2026-08-01"): 836.0,
        pd.Timestamp("2026-09-01"): 486.0,
    }
    for d, v in expected_tail.items():
        if d not in y.index or float(y.loc[d]) != v:
            raise ValueError(f"unexpected value at {d.date()}: got {y.get(d)} expected {v}")
    if y.index.max() != pd.Timestamp("2026-09-01"):
        raise ValueError(f"expected last obs 2026-09-01, got {y.index.max().date()}")

    submission, rolling = forecast_round(y)
    out_csv = OUTPUT_DIR / f"{REFERENCE_DATE.strftime('%Y-%m-%d')}-{MODEL_ID}.csv"
    write_hub_file(submission, out_csv, reference_date=REFERENCE_DATE)
    roll_path = OUTPUT_DIR / "rolling_raw_cqr.csv"
    rolling.to_csv(roll_path, index=False, encoding="utf-8")
    print(f"Wrote {out_csv}")
    print(f"Wrote {roll_path}")
    wide = submission.pivot(index="horizon", columns="output_type_id", values="value")
    wide.columns = [float(c) for c in wide.columns]
    print(wide[[0.05, 0.5, 0.95]])


if __name__ == "__main__":
    main()
