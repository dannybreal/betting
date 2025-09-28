from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import duckdb
import pandas as pd

from src.integrations.api_football import ApiFootballClient

BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "database" / "betting.duckdb"
DEFAULT_CHUNK = 20


def _chunked(seq: List[int], size: int) -> Iterable[list[int]]:
    for idx in range(0, len(seq), size):
        yield seq[idx : idx + size]


def _convert_stat_value(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith('%'):
            try:
                return float(text[:-1])
            except ValueError:
                return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def fetch_fixture_details(
    season: int,
    *,
    chunk_size: int = DEFAULT_CHUNK,
    max_calls: int | None = None,
    db_path: Path = DB_PATH,
) -> dict[str, int]:
    con = duckdb.connect(str(db_path))
    try:
        fixture_ids = con.execute(
            """
            SELECT match_id
            FROM matches
            WHERE season = ?
              AND source_file LIKE 'api_football_%'
              AND result IS NOT NULL
              AND match_id NOT IN (
                  SELECT fixture_id FROM fixture_details_status
              )
            ORDER BY match_date
            """,
            [f"{season}/{season + 1}"],
        ).fetchdf()["match_id"].tolist()
    finally:
        con.close()

    if not fixture_ids:
        return {"fixtures": 0, "api_calls": 0}

    client = ApiFootballClient()
    stats_records: list[dict] = []
    event_records: list[dict] = []
    lineup_records: list[dict] = []
    processed_ids: list[int] = []
    call_count = 0
    updated_at = pd.Timestamp.utcnow()

    for chunk in _chunked(fixture_ids, chunk_size):
        if max_calls is not None and call_count >= max_calls:
            break
        ids_param = "-".join(str(fid) for fid in chunk)
        payload = client.get("/fixtures", params={"ids": ids_param})
        call_count += 1

        for entry in payload.get("response", []) or []:
            fixture = entry.get("fixture", {})
            fixture_id = fixture.get("id")
            if fixture_id is None:
                continue
            processed_ids.append(fixture_id)

            for team_stats in entry.get("statistics", []) or []:
                team_info = team_stats.get("team", {})
                team_id = team_info.get("id")
                team_name = team_info.get("name")
                for stat in team_stats.get("statistics", []) or []:
                    stats_records.append(
                        {
                            "fixture_id": fixture_id,
                            "team_id": team_id,
                            "team_name": team_name,
                            "stat_type": stat.get("type"),
                            "stat_value": _convert_stat_value(stat.get("value")),
                            "updated_at": updated_at,
                        }
                    )

            for ev in entry.get("events", []) or []:
                time_info = ev.get("time", {})
                team_info = ev.get("team", {})
                player_info = ev.get("player", {})
                assist_info = ev.get("assist", {})
                event_records.append(
                    {
                        "fixture_id": fixture_id,
                        "minute": time_info.get("elapsed"),
                        "extra": time_info.get("extra"),
                        "team_name": team_info.get("name"),
                        "player_name": player_info.get("name"),
                        "assist_name": assist_info.get("name"),
                        "event_type": ev.get("type"),
                        "detail": ev.get("detail"),
                        "comments": ev.get("comments"),
                        "updated_at": updated_at,
                    }
                )

            for lineup in entry.get("lineups", []) or []:
                team_info = lineup.get("team", {})
                team_id = team_info.get("id")
                team_name = team_info.get("name")
                formation = lineup.get("formation")
                coach_name = (lineup.get("coach") or {}).get("name")

                for section, is_start in (("startXI", True), ("substitutes", False)):
                    for player_entry in lineup.get(section, []) or []:
                        player = player_entry.get("player", {})
                        lineup_records.append(
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

    if not processed_ids:
        return {"fixtures": 0, "api_calls": call_count}

    con = duckdb.connect(str(db_path))
    try:
        if stats_records:
            df = pd.DataFrame(stats_records)
            con.register("fixture_stats_tmp", df)
            con.execute(
                "DELETE FROM fixture_statistics WHERE fixture_id IN (SELECT DISTINCT fixture_id FROM fixture_stats_tmp)"
            )
            con.execute(
                "INSERT INTO fixture_statistics SELECT fixture_id, team_id, team_name, stat_type, stat_value, updated_at FROM fixture_stats_tmp"
            )
            con.unregister("fixture_stats_tmp")

        if event_records:
            df = pd.DataFrame(event_records)
            con.register("fixture_events_tmp", df)
            con.execute(
                "DELETE FROM fixture_events WHERE fixture_id IN (SELECT DISTINCT fixture_id FROM fixture_events_tmp)"
            )
            con.execute(
                "INSERT INTO fixture_events SELECT fixture_id, minute, extra, team_name, player_name, assist_name, event_type, detail, comments, updated_at FROM fixture_events_tmp"
            )
            con.unregister("fixture_events_tmp")

        if lineup_records:
            df = pd.DataFrame(lineup_records)
            con.register("fixture_lineups_tmp", df)
            con.execute(
                "DELETE FROM fixture_lineups WHERE fixture_id IN (SELECT DISTINCT fixture_id FROM fixture_lineups_tmp)"
            )
            con.execute(
                "INSERT INTO fixture_lineups SELECT fixture_id, team_id, team_name, formation, coach_name, player_id, player_name, player_number, position, grid, is_starting, lineup_type, updated_at FROM fixture_lineups_tmp"
            )
            con.unregister("fixture_lineups_tmp")

        status_df = pd.DataFrame(
            {
                "fixture_id": list(dict.fromkeys(processed_ids)),
                "fetched_at": updated_at,
            }
        )
        con.register("fixture_status_tmp", status_df)
        con.execute(
            "DELETE FROM fixture_details_status WHERE fixture_id IN (SELECT DISTINCT fixture_id FROM fixture_status_tmp)"
        )
        con.execute(
            "INSERT INTO fixture_details_status SELECT fixture_id, fetched_at FROM fixture_status_tmp"
        )
        con.unregister("fixture_status_tmp")
    finally:
        con.close()

    return {
        "fixtures": len(processed_ids),
        "api_calls": call_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch fixture statistics/events/lineups")
    parser.add_argument("--season", type=int, default=2025, help="Season start year (e.g. 2025 for 2025/26)")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK)
    parser.add_argument("--max-calls", type=int, default=None)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = fetch_fixture_details(
        args.season,
        chunk_size=args.chunk_size,
        max_calls=args.max_calls,
        db_path=args.db,
    )
    print(f"Processed fixtures: {stats['fixtures']}")
    print(f"API calls used: {stats['api_calls']}")


if __name__ == "__main__":
    main()
