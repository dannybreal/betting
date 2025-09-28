from __future__ import annotations

import argparse
import hashlib
from datetime import time
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "database" / "betting.duckdb"

RESULT_STATS = {
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HC": "home_corners",
    "AC": "away_corners",
    "HY": "home_yellow",
    "AY": "away_yellow",
    "HR": "home_red",
    "AR": "away_red",
}


def _connect(db_path: Path | None = None) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(db_path or DB_PATH))


def _season_from_date(date: pd.Timestamp) -> str:
    year = int(date.year)
    if date.month >= 7:
        return f"{year}/{year + 1}"
    return f"{year - 1}/{year}"


def _standardise_datetime(date_series: pd.Series, time_series: pd.Series | None = None) -> pd.Series:
    dt = pd.to_datetime(date_series, dayfirst=True, errors="coerce")
    if time_series is not None:
        time_text = time_series.fillna("00:00").astype(str)
        parsed = pd.to_datetime(time_text, format="%H:%M", errors="coerce")
        dt = dt.dt.floor("D")
        offsets: list[int] = []
        for value in parsed:
            if pd.isna(value):
                offsets.append(0)
                continue
            t: time = value.time()
            offsets.append(t.hour * 60 + t.minute)
        dt = dt + pd.to_timedelta(offsets, unit="m")
    return dt


def _compute_match_ids(rows: Iterable[str]) -> list[int]:
    ids: list[int] = []
    for row in rows:
        digest = hashlib.md5(row.encode("utf-8")).hexdigest()[:16]
        ids.append(int(digest, 16) % 9223372036854775783)
    return ids


def load_results(csv_path: str | Path, *, db_path: Path | None = None, source_tag: str | None = None) -> dict[str, int]:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    if df.empty:
        return {"matches": 0, "stats": 0}

    df["match_date"] = _standardise_datetime(df["Date"], df.get("Time"))
    df["season"] = df["match_date"].apply(_season_from_date)
    key_series = (
        df["Div"].astype(str)
        + "|"
        + df["match_date"].dt.strftime("%Y-%m-%d %H:%M")
        + "|"
        + df["HomeTeam"].astype(str)
        + "|"
        + df["AwayTeam"].astype(str)
    )
    df["match_id"] = _compute_match_ids(key_series)

    matches_df = df[
        [
            "match_id",
            "Div",
            "season",
            "match_date",
            "HomeTeam",
            "AwayTeam",
            "FTHG",
            "FTAG",
            "FTR",
        ]
    ].rename(
        columns={
            "Div": "div",
            "HomeTeam": "home_team",
            "AwayTeam": "away_team",
            "FTHG": "home_goals",
            "FTAG": "away_goals",
            "FTR": "result",
        }
    )
    matches_df["source_file"] = source_tag or csv_path.name

    stat_records: list[tuple[int, str, float]] = []
    for raw_name, mapped_name in RESULT_STATS.items():
        if raw_name not in df.columns:
            continue
        col = pd.to_numeric(df[raw_name], errors="coerce")
        for match_id, value in zip(df["match_id"], col):
            if pd.isna(value):
                continue
            stat_records.append((match_id, mapped_name, float(value)))

    con = _connect(db_path)
    try:
        con.register("matches_tmp", matches_df)
        con.execute("DELETE FROM matches WHERE match_id IN (SELECT match_id FROM matches_tmp)")
        con.execute(
            "INSERT INTO matches SELECT match_id, div, season, match_date, home_team, away_team, home_goals, away_goals, result, source_file FROM matches_tmp"
        )
        con.unregister("matches_tmp")

        if stat_records:
            stats_df = pd.DataFrame(stat_records, columns=["match_id", "stat_name", "stat_value"])
            con.register("stats_tmp", stats_df)
            con.execute(
                "DELETE FROM match_stats WHERE (match_id, stat_name) IN (SELECT match_id, stat_name FROM stats_tmp)"
            )
            con.execute("INSERT INTO match_stats SELECT * FROM stats_tmp")
            con.unregister("stats_tmp")
    finally:
        con.close()

    return {"matches": len(matches_df), "stats": len(stat_records)}


def load_fixtures(csv_path: str | Path, *, db_path: Path | None = None, source_tag: str | None = None) -> dict[str, int]:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    if df.empty:
        return {"fixtures": 0}

    df["match_date"] = _standardise_datetime(df["Date"], df.get("Time"))
    key_series = (
        df["Div"].astype(str)
        + "|"
        + df["match_date"].dt.strftime("%Y-%m-%d %H:%M")
        + "|"
        + df["HomeTeam"].astype(str)
        + "|"
        + df["AwayTeam"].astype(str)
    )
    df["fixt_id"] = _compute_match_ids(key_series)

    fixtures_df = df[["fixt_id", "Div", "match_date", "HomeTeam", "AwayTeam"]].rename(
        columns={
            "Div": "div",
            "HomeTeam": "home_team",
            "AwayTeam": "away_team",
        }
    )
    fixtures_df["source_file"] = source_tag or csv_path.name

    con = _connect(db_path)
    try:
        con.register("fixtures_tmp", fixtures_df)
        con.execute("DELETE FROM fixtures_queue WHERE fixt_id IN (SELECT fixt_id FROM fixtures_tmp)")
        con.execute(
            "INSERT INTO fixtures_queue SELECT fixt_id, div, match_date, home_team, away_team, source_file FROM fixtures_tmp"
        )
        con.unregister("fixtures_tmp")
    finally:
        con.close()

    return {"fixtures": len(fixtures_df)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load CSV data into DuckDB")
    parser.add_argument("kind", choices=["results", "fixtures"], help="type of CSV to load")
    parser.add_argument("csv", type=Path, help="path to CSV file")
    parser.add_argument("--db", type=Path, default=DB_PATH, help="path to DuckDB database")
    parser.add_argument("--tag", type=str, default=None, help="optional source tag")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.kind == "results":
        info = load_results(args.csv, db_path=args.db, source_tag=args.tag)
    else:
        info = load_fixtures(args.csv, db_path=args.db, source_tag=args.tag)
    for key, value in info.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
