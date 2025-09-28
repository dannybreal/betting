from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import duckdb
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "database" / "betting.duckdb"
DEFAULT_DRAW_BASE = 0.21
CALIB_FEATURES: List[str] = [
    "intercept",
    "abs_elo",
    "strength_gap",
    "xg_gap",
    "sot_gap",
    "prob_draw_raw",
    "draw_rate",
]
DEFAULT_RIDGE = 1e-3


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _parse_date(value: Optional[str]) -> Optional[pd.Timestamp]:
    if not value:
        return None
    ts = pd.to_datetime(value, utc=False, errors="coerce")
    if ts is pd.NaT:
        raise ValueError(f"Invalid date: {value}")
    return ts


def fetch_training_data(
    con: duckdb.DuckDBPyConnection,
    min_date: Optional[pd.Timestamp] = None,
    max_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    filters = []
    params: List[object] = []
    if min_date is not None:
        filters.append("fr.match_date >= ?")
        params.append(min_date.to_pydatetime())
    if max_date is not None:
        filters.append("fr.match_date <= ?")
        params.append(max_date.to_pydatetime())
    date_filter = " AND " + " AND ".join(filters) if filters else ""

    query = f"""
        WITH results AS (
            SELECT fixt_id, div, match_date, result, status
            FROM fixture_results fr
            WHERE status IN ('FT', 'AET', 'PEN')
              AND result IS NOT NULL
              {date_filter}
        ), preview_ranked AS (
            SELECT fph.*, r.match_date AS result_match_date,
                   ROW_NUMBER() OVER (PARTITION BY fph.fixt_id ORDER BY generated_at DESC) AS rn
            FROM fixture_previews_history fph
            JOIN results r ON r.fixt_id = fph.fixt_id
            WHERE fph.generated_at <= r.match_date
        ), latest_preview AS (
            SELECT *
            FROM preview_ranked
            WHERE rn = 1
        ), league_draw AS (
            SELECT div,
                   CAST(SUM(CASE WHEN result = 'D' THEN 1 ELSE 0 END) AS DOUBLE) / COUNT(*) AS draw_rate
            FROM matches
            WHERE result IS NOT NULL
            GROUP BY div
        )
        SELECT r.fixt_id,
               r.div,
               r.match_date,
               r.result,
               lp.elo_edge,
               lp.home_strength,
               lp.away_strength,
               lp.home_xg,
               lp.away_xg,
               lp.home_sot,
               lp.away_sot,
               lp.prob_draw,
               lp.prob_draw_raw,
               ld.draw_rate
        FROM results r
        JOIN latest_preview lp ON lp.fixt_id = r.fixt_id
        LEFT JOIN league_draw ld ON ld.div = r.div
    """
    df = con.execute(query, params).fetchdf()
    return df


def prepare_design(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    working = df.copy()
    working["prob_draw_raw"] = working["prob_draw_raw"].fillna(working["prob_draw"])
    working["prob_draw_raw"] = working["prob_draw_raw"].clip(0.01, 0.9)

    for col, default in (("home_xg", 1.2), ("away_xg", 1.2), ("home_sot", 3.0), ("away_sot", 3.0)):
        working[col] = working[col].fillna(default)

    working["draw_rate"] = working["draw_rate"].fillna(DEFAULT_DRAW_BASE)
    working["abs_elo"] = working["elo_edge"].abs()
    working["strength_gap"] = (working["home_strength"].fillna(0.0) - working["away_strength"].fillna(0.0)).abs()
    working["xg_gap"] = (working["home_xg"] - working["away_xg"]).abs()
    working["sot_gap"] = (working["home_sot"] - working["away_sot"]).abs()

    feature_matrix = np.column_stack([
        np.ones(len(working)),
        working["abs_elo"].values,
        working["strength_gap"].values,
        working["xg_gap"].values,
        working["sot_gap"].values,
        working["prob_draw_raw"].values,
        working["draw_rate"].values,
    ])
    targets = (working["result"].str.upper() == "D").astype(float).values
    return feature_matrix, targets, working


def fit_logistic(X: np.ndarray, y: np.ndarray, ridge: float = DEFAULT_RIDGE, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    beta = np.zeros(X.shape[1])
    eye = np.eye(X.shape[1])
    for _ in range(max_iter):
        z = X @ beta
        p = _sigmoid(z)
        W = p * (1.0 - p)
        grad = X.T @ (p - y) + ridge * beta
        H = X.T @ (W[:, None] * X) + ridge * eye
        try:
            delta = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(H, grad, rcond=None)[0]
        beta -= delta
        if np.max(np.abs(delta)) < tol:
            break
    return beta


def compute_metrics(y: np.ndarray, baseline: np.ndarray, calibrated: np.ndarray) -> dict:
    clip_base = np.clip(baseline, 1e-9, 1 - 1e-9)
    clip_cal = np.clip(calibrated, 1e-9, 1 - 1e-9)
    return {
        "samples": int(len(y)),
        "draw_rate": float(y.mean()),
        "baseline_mean": float(baseline.mean()),
        "calibrated_mean": float(calibrated.mean()),
        "brier_baseline": float(np.mean((baseline - y) ** 2)),
        "brier_calibrated": float(np.mean((calibrated - y) ** 2)),
        "logloss_baseline": float(-np.mean(y * np.log(clip_base) + (1 - y) * np.log(1 - clip_base))),
        "logloss_calibrated": float(-np.mean(y * np.log(clip_cal) + (1 - y) * np.log(1 - clip_cal))),
    }


def store_coefficients(
    con: duckdb.DuckDBPyConnection,
    coefficients: np.ndarray,
    metrics: dict,
    feature_names: List[str] = CALIB_FEATURES,
) -> None:
    con.execute("CREATE TABLE IF NOT EXISTS draw_calibration (created_at TIMESTAMP, feature_names JSON, coefficients JSON, metrics JSON)")
    con.execute("CREATE TABLE IF NOT EXISTS draw_calibration_history (created_at TIMESTAMP, feature_names JSON, coefficients JSON, metrics JSON)")
    entry = [
        pd.Timestamp.utcnow().to_pydatetime(),
        json.dumps(feature_names),
        json.dumps(coefficients.tolist()),
        json.dumps(metrics),
    ]
    con.execute("INSERT INTO draw_calibration_history (created_at, feature_names, coefficients, metrics) VALUES (?, ?, ?, ?)", entry)
    con.execute("DELETE FROM draw_calibration")
    con.execute("INSERT INTO draw_calibration (created_at, feature_names, coefficients, metrics) VALUES (?, ?, ?, ?)", entry)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train draw calibration logistic model")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--min-date", type=str, default=None, help="Only include fixtures on/after this date (YYYY-MM-DD)")
    parser.add_argument("--max-date", type=str, default=None, help="Only include fixtures on/before this date (YYYY-MM-DD)")
    parser.add_argument("--min-samples", type=int, default=400, help="Minimum matches required to fit the model")
    parser.add_argument("--ridge", type=float, default=DEFAULT_RIDGE, help="L2 regularisation strength")
    args = parser.parse_args()

    min_date = _parse_date(args.min_date)
    max_date = _parse_date(args.max_date)

    con = duckdb.connect(str(args.db))
    try:
        raw = fetch_training_data(con, min_date, max_date)
        if raw.empty or len(raw) < args.min_samples:
            raise SystemExit(f"Not enough samples for calibration ({len(raw)} found)")
        X, y, working = prepare_design(raw)
        baseline = working["prob_draw_raw"].values
        coefficients = fit_logistic(X, y, ridge=args.ridge)
        calibrated = _sigmoid(X @ coefficients)
        metrics = compute_metrics(y, baseline, calibrated)
        store_coefficients(con, coefficients, metrics)
    finally:
        con.close()

    print(f"Calibrated on {metrics['samples']} matches")
    print(json.dumps(metrics, indent=2))
    print("Coefficients:")
    for name, value in zip(CALIB_FEATURES, coefficients.tolist()):
        print(f"  {name}: {value:.6f}")


if __name__ == "__main__":
    main()
