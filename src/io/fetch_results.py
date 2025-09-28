from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import duckdb
import pandas as pd

from src.integrations.api_football import ApiFootballClient, ApiFootballError

BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "database" / "betting.duckdb"
DEFAULT_BUFFER_HOURS = 2
DEFAULT_CHUNK_SIZE = 20
FINAL_STATUSES = {"FT", "AET", "PEN"}


def _chunked(seq: List[int], size: int) -> Iterable[list[int]]:
    for idx in range(0, len(seq), size):
        yield seq[idx : idx + size]


def _season_label(season_year: int | None) -> str | None:
    if season_year is None:
        return None
    return f"{season_year}/{season_year + 1}"


def _season_from_date(value: pd.Timestamp | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    year = int(value.year)
    if value.month >= 7:
        return f"{year}/{year + 1}"
    return f"{year - 1}/{year}"


def _winner_symbol(home: int | None, away: int | None) -> str | None:
    if home is None or away is None:
        return None
    if home > away:
        return "H"
    if away > home:
        return "A"
    return "D"


def _load_pending_fixtures(
    con: duckdb.DuckDBPyConnection,
    cutoff: pd.Timestamp,
    divisions: list[str] | None,
    limit: int | None,
) -> pd.DataFrame:
    query = """
        SELECT f.fixt_id,
               f.div,
               f.match_date,
               f.home_team,
               f.away_team,
               f.source_file
        FROM fixtures_queue AS f
        LEFT JOIN fixture_results AS r
          ON f.fixt_id = r.fixt_id
        WHERE f.match_date <= ?
          AND (r.fixt_id IS NULL OR r.status NOT IN ('FT', 'AET', 'PEN'))
    """
    params: list[object] = [cutoff.to_pydatetime()]
    if divisions:
        placeholders = ",".join("?" for _ in divisions)
        query += f" AND f.div IN ({placeholders})"
        params.extend(divisions)
    query += " ORDER BY f.match_date"
    if limit:
        query += " LIMIT ?"
        params.append(limit)
    return con.execute(query, params).fetchdf()


def _fetch_results(client: ApiFootballClient, fixture_ids: list[int]) -> list[dict]:
    try:
        payload = client.get("/fixtures", params={"ids": "-".join(str(fid) for fid in fixture_ids)})
    except ApiFootballError as exc:
        print(f"[WARN] fixture results fetch failed for batch {fixture_ids[:3]}...: {exc}")
        return []
    return payload.get("response", []) or []


def _prepare_records(
    entries: list[dict],
    fixture_lookup: dict[int, dict],
    fetched_at: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_rows: list[dict] = []
    match_rows: list[dict] = []

    for entry in entries:
        fixture = entry.get("fixture", {}) or {}
        fixture_id = fixture.get("id")
        if fixture_id is None or fixture_id not in fixture_lookup:
            continue

        lookup = fixture_lookup[fixture_id]
        teams = entry.get("teams", {}) or {}
        goals = entry.get("goals", {}) or {}
        league = entry.get("league", {}) or {}

        home_team = lookup.get("home_team") or (teams.get("home") or {}).get("name")
        away_team = lookup.get("away_team") or (teams.get("away") or {}).get("name")
        match_ts = pd.to_datetime(lookup.get("match_date"))

        home_goals = goals.get("home")
        away_goals = goals.get("away")
        status_info = fixture.get("status") or {}
        status_short = status_info.get("short")
        result_symbol = _winner_symbol(home_goals, away_goals)

        season_label = _season_label(league.get("season")) or _season_from_date(match_ts)

        result_rows.append(
            {
                "fixt_id": fixture_id,
                "div": lookup.get("div"),
                "match_date": match_ts,
                "home_team": home_team,
                "away_team": away_team,
                "season": season_label,
                "status": status_short,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "result": result_symbol,
                "fetched_at": fetched_at,
                "source": lookup.get("source_file") or f"api_football_{league.get('id')}",
            }
        )

        if status_short in FINAL_STATUSES and home_goals is not None and away_goals is not None:
            match_rows.append(
                {
                    "match_id": fixture_id,
                    "div": lookup.get("div"),
                    "season": season_label,
                    "match_date": match_ts,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_goals": int(home_goals),
                    "away_goals": int(away_goals),
                    "result": result_symbol,
                    "source_file": lookup.get("source_file") or f"api_football_{league.get('id')}",
                }
            )

    results_df = pd.DataFrame(result_rows)
    matches_df = pd.DataFrame(match_rows)
    return results_df, matches_df


def _store_results(
    con: duckdb.DuckDBPyConnection,
    results_df: pd.DataFrame,
    matches_df: pd.DataFrame,
) -> None:
    if not results_df.empty:
        con.register("fixture_results_tmp", results_df)
        con.execute(
            "DELETE FROM fixture_results WHERE fixt_id IN (SELECT DISTINCT fixt_id FROM fixture_results_tmp)"
        )
        con.execute(
            "INSERT INTO fixture_results SELECT fixt_id, div, match_date, home_team, away_team, season, status, home_goals, away_goals, result, fetched_at, source FROM fixture_results_tmp"
        )
        con.unregister("fixture_results_tmp")

    if not matches_df.empty:
        con.register("matches_tmp", matches_df)
        con.execute(
            "DELETE FROM matches WHERE match_id IN (SELECT DISTINCT match_id FROM matches_tmp)"
        )
        con.execute(
            "INSERT INTO matches SELECT match_id, div, season, match_date, home_team, away_team, home_goals, away_goals, result, source_file FROM matches_tmp"
        )
        con.unregister("matches_tmp")


def fetch_and_store_results(
    *,
    db_path: Path = DB_PATH,
    buffer_hours: int = DEFAULT_BUFFER_HOURS,
    divisions: list[str] | None = None,
    limit: int | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_calls: int | None = None,
) -> dict[str, int]:
    con = duckdb.connect(str(db_path))
    try:
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(hours=buffer_hours)
        pending = _load_pending_fixtures(con, cutoff, divisions, limit)
    finally:
        con.close()

    if pending.empty:
        return {"fixtures": 0, "api_calls": 0}

    fixture_lookup = {
        int(row.fixt_id): {
            "div": row.div,
            "match_date": row.match_date,
            "home_team": row.home_team,
            "away_team": row.away_team,
            "source_file": row.source_file,
        }
        for row in pending.itertuples(index=False)
    }
    fixture_ids = list(fixture_lookup.keys())

    client = ApiFootballClient()
    fetched_at = pd.Timestamp.utcnow()
    total_calls = 0
    total_fixtures = 0

    con = duckdb.connect(str(db_path))
    try:
        for batch in _chunked(fixture_ids, chunk_size):
            if max_calls is not None and total_calls >= max_calls:
                break
            entries = _fetch_results(client, batch)
            total_calls += 1
            if not entries:
                continue
            results_df, matches_df = _prepare_records(entries, fixture_lookup, fetched_at)
            if results_df.empty and matches_df.empty:
                continue
            _store_results(con, results_df, matches_df)
            total_fixtures += len(results_df)
    finally:
        con.close()

    return {"fixtures": total_fixtures, "api_calls": total_calls}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch final results for completed fixtures")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--buffer-hours", type=int, default=DEFAULT_BUFFER_HOURS, help="Skip fixtures played within the last N hours")
    parser.add_argument("--div", action="append", dest="divisions", help="Limit to specific divisions (repeatable)")
    parser.add_argument("--limit", type=int, default=None, help="Hard cap on fixtures pulled from the queue")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--max-calls", type=int, default=None, help="Stop after this many API calls")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    stats = fetch_and_store_results(
        db_path=args.db,
        buffer_hours=args.buffer_hours,
        divisions=args.divisions,
        limit=args.limit,
        chunk_size=args.chunk_size,
        max_calls=args.max_calls,
    )
    print(f"Logged fixtures: {stats['fixtures']}")
    print(f"API calls used: {stats['api_calls']}")


if __name__ == "__main__":
    main()
