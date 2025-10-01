from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "database" / "betting.duckdb"
REPORTS_DIR = BASE_DIR / "reports"
DEFAULT_WINDOW_DAYS = 7


def _latest_previews(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    query = """
        WITH ranked AS (
            SELECT *,
                   ROW_NUMBER() OVER (PARTITION BY fixt_id ORDER BY generated_at DESC) AS rn
            FROM fixture_previews_history
        )
        SELECT fixt_id,
               div,
               prob_home,
               prob_draw,
               prob_away
        FROM ranked
        WHERE rn = 1
    """
    return con.execute(query).fetchdf()


def _load_results(con: duckdb.DuckDBPyConnection, cutoff: dt.datetime) -> pd.DataFrame:
    query = """
        SELECT fixt_id,
               div,
               match_date,
               result
        FROM fixture_results
        WHERE status IN ('FT', 'AET', 'PEN')
          AND match_date >= ?
    """
    return con.execute(query, [cutoff]).fetchdf()


def _merge_data(previews: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    previews = previews.drop(columns=['div'], errors='ignore')
    merged = results.merge(previews, on="fixt_id", how="left", suffixes=("", ""))
    merged["match_date"] = pd.to_datetime(merged["match_date"])
    merged = merged.dropna(subset=["prob_home", "prob_draw", "prob_away"])
    return merged


def _brier(row: pd.Series) -> float:
    actual = [row["result"] == code for code in ("H", "D", "A")]
    predicted = [row["prob_home"], row["prob_draw"], row["prob_away"]]
    return float(np.sum((np.array(predicted) - np.array(actual, dtype=float)) ** 2))


def _log_loss(row: pd.Series) -> float:
    mapping = {"H": row["prob_home"], "D": row["prob_draw"], "A": row["prob_away"]}
    prob = float(mapping.get(row["result"], np.nan))
    if prob <= 0 or np.isnan(prob):
        prob = 1e-9
    return float(-np.log(prob))


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["brier"] = df.apply(_brier, axis=1)
    df["log_loss"] = df.apply(_log_loss, axis=1)
    one_hot = pd.get_dummies(df["result"])
    result_cols = {"H": "actual_home", "D": "actual_draw", "A": "actual_away"}
    for code, col in result_cols.items():
        df[col] = one_hot.get(code, pd.Series(dtype=float)).fillna(0.0)
    summaries = (
        df.groupby("div")
        .agg(
            matches=("fixt_id", "count"),
            avg_brier=("brier", "mean"),
            avg_log_loss=("log_loss", "mean"),
            pred_home=("prob_home", "mean"),
            pred_draw=("prob_draw", "mean"),
            pred_away=("prob_away", "mean"),
            actual_home=("actual_home", "mean"),
            actual_draw=("actual_draw", "mean"),
            actual_away=("actual_away", "mean"),
        )
        .reset_index()
    )
    totals = pd.DataFrame({
        "div": ["ALL"],
        "matches": [summaries["matches"].sum()],
        "avg_brier": [df["brier"].mean()],
        "avg_log_loss": [df["log_loss"].mean()],
        "pred_home": [df["prob_home"].mean()],
        "pred_draw": [df["prob_draw"].mean()],
        "pred_away": [df["prob_away"].mean()],
        "actual_home": [df["actual_home"].mean()],
        "actual_draw": [df["actual_draw"].mean()],
        "actual_away": [df["actual_away"].mean()],
    })
    return pd.concat([summaries, totals], ignore_index=True)


def log_league_metrics(window_days: int = DEFAULT_WINDOW_DAYS, db_path: Path = DB_PATH) -> Path | None:
    cutoff = dt.datetime.utcnow() - dt.timedelta(days=window_days)
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        previews = _latest_previews(con)
        results = _load_results(con, cutoff)
    finally:
        con.close()
    merged = _merge_data(previews, results)
    if merged.empty:
        return None
    report = _aggregate(merged)
    REPORTS_DIR.mkdir(exist_ok=True)
    timestamp = dt.datetime.utcnow().strftime("%Y%m%d")
    output_path = REPORTS_DIR / f"league_metrics_{timestamp}.csv"
    report.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Log recent league calibration metrics")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW_DAYS, help="Days back to include")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    args = parser.parse_args()
    path = log_league_metrics(window_days=args.window, db_path=args.db)
    if not path:
        print("No matches found in window")
    else:
        print(f"Metrics written to {path}")


if __name__ == "__main__":
    main()
