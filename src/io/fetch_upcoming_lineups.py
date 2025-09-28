from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd

from src.integrations.api_football import ApiFootballClient

BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "database" / "betting.duckdb"


def _store_lineups(con: duckdb.DuckDBPyConnection, records: list[dict]) -> None:
    if not records:
        return
    df = pd.DataFrame(records)
    con.register("upcoming_lineups_tmp", df)
    con.execute(
        "DELETE FROM fixture_lineups WHERE fixture_id IN (SELECT DISTINCT fixture_id FROM upcoming_lineups_tmp)"
    )
    con.execute(
        "INSERT INTO fixture_lineups SELECT fixture_id, team_id, team_name, formation, coach_name, player_id, player_name, player_number, position, grid, is_starting, lineup_type, updated_at FROM upcoming_lineups_tmp"
    )
    con.unregister("upcoming_lineups_tmp")


def fetch_upcoming_lineups(
    lookahead_minutes: int,
    *,
    max_fixtures: int | None = None,
    db_path: Path = DB_PATH,
) -> dict[str, int]:
    now = datetime.utcnow()
    horizon = now + timedelta(minutes=lookahead_minutes)

    con = duckdb.connect(str(db_path))
    try:
        fixture_ids = con.execute(
            """
            SELECT fixt_id AS fixture_id, match_date
            FROM fixtures_queue
            WHERE match_date >= ? AND match_date <= ?
              AND fixt_id NOT IN (
                  SELECT DISTINCT fixture_id FROM fixture_lineups
              )
            ORDER BY match_date
            """,
            [now, horizon],
        ).fetchdf()
    finally:
        con.close()

    if fixture_ids.empty:
        return {"fixtures": 0, "api_calls": 0}

    if max_fixtures is not None:
        fixture_ids = fixture_ids.head(max_fixtures)

    client = ApiFootballClient()
    records: list[dict] = []
    call_count = 0
    updated_at = pd.Timestamp.utcnow()

    for row in fixture_ids.itertuples(index=False):
        fixture_id = int(row.fixture_id)
        payload = client.get("/fixtures/lineups", params={"fixture": fixture_id})
        call_count += 1
        for lineup in payload.get("response", []) or []:
            team = lineup.get("team", {})
            team_id = team.get("id")
            team_name = team.get("name")
            formation = lineup.get("formation")
            coach_name = (lineup.get("coach") or {}).get("name")

            for section, is_start in (("startXI", True), ("substitutes", False)):
                for player_entry in lineup.get(section, []) or []:
                    player = player_entry.get("player", {})
                    records.append(
                        {
                            "fixture_id": fixture_id,
                            "team_id": team_id,
                            "team_name": team_name,
                            "formation": formation,
                            "coach_name": coach_name,
                            "player_id": player.get("id"),
                            "player_name": player.get("name"),
                            "player_number": player.get("number"),
                            "position": player.get("pos"),
                            "grid": player.get("grid"),
                            "is_starting": is_start,
                            "lineup_type": section,
                            "updated_at": updated_at,
                        }
                    )

    con = duckdb.connect(str(db_path))
    try:
        _store_lineups(con, records)
    finally:
        con.close()

    return {"fixtures": len(fixture_ids), "api_calls": call_count}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch lineups for upcoming fixtures")
    parser.add_argument("--lookahead", type=int, default=90, help="Minutes ahead to look for fixtures")
    parser.add_argument("--max-fixtures", type=int, default=None, help="Optional limit on number of fixtures")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = fetch_upcoming_lineups(
        args.lookahead,
        max_fixtures=args.max_fixtures,
        db_path=args.db,
    )
    print(f"Fixtures processed: {stats['fixtures']}")
    print(f"API calls used: {stats['api_calls']}")


if __name__ == "__main__":
    main()
