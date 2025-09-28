from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "database" / "betting.duckdb"
REPORTS_DIR = BASE_DIR / "reports"


def load_predictions(con: duckdb.DuckDBPyConnection, season: Optional[str] = None) -> pd.DataFrame:
    season_filter = "AND m.season = ?" if season else ""
    params = [season] if season else []
    query = f"""
        WITH ranked AS (
            SELECT h.*,
                   ROW_NUMBER() OVER (
                       PARTITION BY h.fixt_id
                       ORDER BY h.generated_at DESC
                   ) AS rn
            FROM fixture_previews_history h
            JOIN matches m ON m.match_id = h.fixt_id
            WHERE h.generated_at <= m.match_date
              AND m.home_goals IS NOT NULL
              AND m.away_goals IS NOT NULL
              {season_filter}
        )
        SELECT r.*, m.home_goals, m.away_goals, m.result
        FROM ranked r
        JOIN matches m ON m.match_id = r.fixt_id
        WHERE rn = 1
    """
    return con.execute(query, params).fetchdf()


def classify_result(row: pd.Series) -> tuple[int, int, int]:
    if row.home_goals > row.away_goals:
        return 1, 0, 0
    if row.home_goals < row.away_goals:
        return 0, 0, 1
    return 0, 1, 0


def safe_log(p: float) -> float:
    return math.log(max(p, 1e-12))


def evaluation_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), df

    df = df.copy()
    df[["act_home", "act_draw", "act_away"]] = df.apply(classify_result, axis=1, result_type="expand")
    df["brier"] = (
        (df["prob_home"] - df["act_home"]) ** 2
        + (df["prob_draw"] - df["act_draw"]) ** 2
        + (df["prob_away"] - df["act_away"]) ** 2
    )

    df["log_loss"] = df.apply(
        lambda r: -(
            r["act_home"] * safe_log(r["prob_home"])
            + r["act_draw"] * safe_log(r["prob_draw"])
            + r["act_away"] * safe_log(r["prob_away"])
        ),
        axis=1,
    )

    df["pred_label"] = df[["prob_home", "prob_draw", "prob_away"]].idxmax(axis=1)
    df["actual_label"] = df.apply(
        lambda r: "prob_home" if r.act_home else ("prob_draw" if r.act_draw else "prob_away"),
        axis=1,
    )
    df["correct"] = df["pred_label"] == df["actual_label"]

    overall = pd.DataFrame(
        {
            "matches": [len(df)],
            "accuracy": [df["correct"].mean()],
            "brier": [df["brier"].mean()],
            "log_loss": [df["log_loss"].mean()],
        }
    )

    labels = ("prob_home", "prob_draw", "prob_away")
    bucket_frames = []
    for idx, label in enumerate(labels):
        bucket_df = df[[label, "act_home", "act_draw", "act_away"]].copy()
        bucket_df["bucket"] = (bucket_df[label] * 20).clip(0, 19).astype(int) / 20.0
        bucket_df["hit"] = bucket_df[["act_home", "act_draw", "act_away"]].values.argmax(axis=1) == idx
        summary = (
            bucket_df.groupby("bucket")
            .agg(pred_mean=(label, "mean"), actual_rate=("hit", "mean"), count=(label, "count"))
            .reset_index()
        )
        summary["label"] = label.replace("prob_", "")
        bucket_frames.append(summary)

    bucket_report = pd.concat(bucket_frames, ignore_index=True)

    return overall, bucket_report, df


def run_backtest(db_path: Path, season: Optional[str], export: Path, buckets: Path, details: Path) -> None:
    con = duckdb.connect(str(db_path))
    try:
        df = load_predictions(con, season=season)
    finally:
        con.close()

    overall, bucket_report, details_df = evaluation_metrics(df)

    REPORTS_DIR.mkdir(exist_ok=True)
    overall.to_csv(export, index=False)
    bucket_report.to_csv(buckets, index=False)
    details_df.to_csv(details, index=False)

    print(overall)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest fixture previews vs actual results")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--season", type=str, default=None)
    parser.add_argument("--export", type=Path, default=REPORTS_DIR / "backtest_report.csv")
    parser.add_argument("--buckets", type=Path, default=REPORTS_DIR / "calibration.csv")
    parser.add_argument("--details", type=Path, default=REPORTS_DIR / "backtest_details.csv")
    args = parser.parse_args()

    run_backtest(args.db, args.season, args.export, args.buckets, args.details)


if __name__ == "__main__":
    main()
