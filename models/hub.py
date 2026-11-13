"""Hub constants and writers for monthly dengue submissions.

Dates are advanced with pandas DateOffset(months=...), never weeks.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGET = "mo inc dengue cases"
LOCATION = "CN"
OUTPUT_TYPE = "quantile"

REFERENCE_DATE = pd.Timestamp("2026-11-01")

# Hub forecast window for the 2026-11-01 dengue round.
HUB_HORIZONS = (-1, 0, 1, 2, 3)

# Model leads from the last observed month (2026-09-01):
#   hub h=-1 -> 2026-10-01 -> lead 1
#   hub h=0  -> 2026-11-01 -> lead 2
#   hub h=1  -> 2026-12-01 -> lead 3
#   hub h=2  -> 2027-01-01 -> lead 4
#   hub h=3  -> 2027-02-01 -> lead 5
MODEL_LEADS = (1, 2, 3, 4, 5)

QUANTILES = (
    0.01,
    0.025,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    0.975,
    0.99,
)

# Exact CSV spellings required by the hub (avoid 0.10 / 0.0250000001).
QUANTILE_ID_STR = {
    0.01: "0.01",
    0.025: "0.025",
    0.05: "0.05",
    0.1: "0.1",
    0.15: "0.15",
    0.2: "0.2",
    0.25: "0.25",
    0.3: "0.3",
    0.35: "0.35",
    0.4: "0.4",
    0.45: "0.45",
    0.5: "0.5",
    0.55: "0.55",
    0.6: "0.6",
    0.65: "0.65",
    0.7: "0.7",
    0.75: "0.75",
    0.8: "0.8",
    0.85: "0.85",
    0.9: "0.9",
    0.95: "0.95",
    0.975: "0.975",
    0.99: "0.99",
}

HUB_COLUMNS = (
    "reference_date",
    "target",
    "horizon",
    "target_end_date",
    "location",
    "output_type",
    "output_type_id",
    "value",
)

TEAM_ABBR = "GZNL"
MODEL_ABBR = "DengueLGBMCQR"
MODEL_ID = f"{TEAM_ABBR}-{MODEL_ABBR}"


def add_months(dates, months: int) -> pd.DatetimeIndex | pd.Timestamp:
    """Shift timestamps by a whole number of calendar months."""
    return dates + pd.DateOffset(months=int(months))


def hub_horizon_to_lead(horizon: int, last_obs: pd.Timestamp, reference_date: pd.Timestamp) -> int:
    """Months from the last observed target month to the hub target month."""
    target_end = add_months(pd.Timestamp(reference_date), horizon)
    # Both dates are month starts; count month differences.
    lo = pd.Timestamp(last_obs)
    te = pd.Timestamp(target_end)
    return (te.year - lo.year) * 12 + (te.month - lo.month)


def target_end_date(reference_date: pd.Timestamp, horizon: int) -> pd.Timestamp:
    return add_months(pd.Timestamp(reference_date), horizon)


def load_dengue_series(path: str | Path) -> pd.Series:
    """Load the national monthly dengue series from a hub time-series.csv."""
    df = pd.read_csv(path)
    needed = {"target_end_date", "target", "location", "observation"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"time-series.csv missing columns: {sorted(missing)}")
    dengue = df.loc[
        (df["target"] == TARGET) & (df["location"] == LOCATION),
        ["target_end_date", "observation"],
    ].copy()
    dengue["target_end_date"] = pd.to_datetime(dengue["target_end_date"])
    dengue = dengue.sort_values("target_end_date")
    if dengue["target_end_date"].dt.day.ne(1).any():
        raise ValueError("dengue target_end_date values must be the first of the month")
    series = dengue.set_index("target_end_date")["observation"].astype(float)
    series = series[~series.index.duplicated(keep="last")].sort_index()
    series.name = "observation"
    return series


def write_hub_file(
    records: pd.DataFrame,
    path: str | Path,
    reference_date: pd.Timestamp | str | None = None,
) -> Path:
    """Write a UTF-8 (no BOM) dengue quantile submission CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ref = pd.Timestamp(reference_date or REFERENCE_DATE).strftime("%Y-%m-%d")

    df = records.copy()
    if list(df.columns) != list(HUB_COLUMNS):
        df = df.reindex(columns=list(HUB_COLUMNS))
    if df.shape[1] != 8 or df.columns.tolist() != list(HUB_COLUMNS):
        raise ValueError(f"submission must have exactly these columns: {HUB_COLUMNS}")

    df["reference_date"] = pd.to_datetime(df["reference_date"]).dt.strftime("%Y-%m-%d")
    if not df["reference_date"].eq(ref).all():
        raise ValueError("reference_date must match the filename date and be constant")
    df["target_end_date"] = pd.to_datetime(df["target_end_date"]).dt.strftime("%Y-%m-%d")
    df["target"] = TARGET
    df["location"] = LOCATION
    df["output_type"] = OUTPUT_TYPE
    df["horizon"] = df["horizon"].astype(int)
    df["value"] = df["value"].astype(int)
    df["output_type_id"] = df["output_type_id"].map(
        lambda q: QUANTILE_ID_STR[float(q)] if float(q) in QUANTILE_ID_STR else str(q)
    )

    if df["value"].lt(0).any():
        raise ValueError("values must be non-negative")
    if df.isnull().any().any():
        raise ValueError("submission contains missing values")
    if df.duplicated(subset=["horizon", "output_type_id"]).any():
        raise ValueError("duplicate (horizon, output_type_id) rows")

    expected = {(h, QUANTILE_ID_STR[q]) for h in HUB_HORIZONS for q in QUANTILES}
    got = set(zip(df["horizon"].tolist(), df["output_type_id"].tolist()))
    if got != expected:
        missing = expected - got
        extra = got - expected
        raise ValueError(f"quantile/horizon coverage mismatch; missing={missing} extra={extra}")

    # Check dates and monotonicity.
    for h, part in df.groupby("horizon"):
        ted = part["target_end_date"].iloc[0]
        expect_ted = target_end_date(ref, int(h)).strftime("%Y-%m-%d")
        if ted != expect_ted:
            raise ValueError(f"horizon {h}: target_end_date {ted} != {expect_ted}")
        if not part["target_end_date"].eq(ted).all():
            raise ValueError(f"horizon {h}: mixed target_end_date values")
        vals = part.sort_values("output_type_id", key=lambda s: s.astype(float))["value"].to_numpy()
        if (vals[1:] < vals[:-1]).any():
            raise ValueError(f"horizon {h}: quantiles are not non-decreasing")

    df = df.sort_values(["horizon", "output_type_id"], key=lambda s: s.astype(float) if s.name == "output_type_id" else s)
    df = df.reset_index(drop=True)

    # utf-8, no BOM, no index, no trailing empty column.
    df.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    text = path.read_bytes()
    if text.startswith(b"\xef\xbb\xbf"):
        raise ValueError("wrote a BOM; UTF-8 without BOM is required")
    # Expected: header + 115 data rows.
    n_lines = text.decode("utf-8").count("\n")
    if n_lines != 116:  # 1 header + 115 rows, all newline-terminated
        raise ValueError(f"expected 116 newline-terminated lines, got {n_lines}")
    return path


def validate_hub_csv(path: str | Path, reference_date: pd.Timestamp | str | None = None) -> dict:
    """Re-read a submission and return a small summary if it is valid."""
    path = Path(path)
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError("CSV has a UTF-8 BOM")
    df = pd.read_csv(path, dtype={"output_type_id": str})
    write_hub_file(df, path, reference_date=reference_date or REFERENCE_DATE)
    summary = []
    for h in HUB_HORIZONS:
        part = df[df["horizon"] == h]
        qmap = {float(q): int(v) for q, v in zip(part["output_type_id"], part["value"])}
        summary.append(
            {
                "horizon": int(h),
                "target_end_date": str(part["target_end_date"].iloc[0]),
                "q05": qmap[0.05],
                "q50": qmap[0.5],
                "q95": qmap[0.95],
            }
        )
    return {"n_rows": len(df), "n_cols": df.shape[1], "horizons": summary}
