from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd

from src.integrations.api_football import ApiFootballClient, ApiFootballError

BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "database" / "betting.duckdb"
DEFAULT_SEASON = date.today().year

LEAGUE_TO_DIV = {
    2: "UCL",    # UEFA Champions League
    3: "UEL",    # UEFA Europa League
    848: "UECL",  # UEFA Europa Conference League
    39: "E0",    # England 1 - Premier League
    40: "E1",    # England 2 - Championship
    41: "E2",    # England 3 - League One
    42: "E3",    # England 4 - League Two
    45: "EC",    # England 5 - National League
    61: "F1",    # France 1 - Ligue 1
    62: "F2",    # France 2 - Ligue 2
    78: "D1",    # Germany 1 - Bundesliga
    79: "D2",    # Germany 2 - 2. Bundesliga
    88: "N1",    # Netherlands 1 - Eredivisie
    94: "P1",    # Portugal 1 - Primeira Liga
    135: "I1",   # Italy 1 - Serie A
    136: "I2",   # Italy 2 - Serie B
    140: "SP1",  # Spain 1 - La Liga
    141: "SP2",  # Spain 2 - Segunda
    144: "B1",   # Belgium 1 - Pro League
    179: "SC0",  # Scotland 1 - Premiership
    180: "SC1",  # Scotland 2 - Championship
    181: "SC2",  # Scotland 3 - League One
    182: "SC3",  # Scotland 4 - League Two
    197: "G1",   # Greece 1 - Super League
    203: "T1",   # Turkey 1 - Super Lig
    71: "BR1",   # Brazil 1 - Serie A
    113: "SW1",  # Sweden 1 - Allsvenskan
    244: "FI1",  # Finland 1 - Veikkausliiga
    329: "EE1",  # Estonia 1 - Meistriliiga

}

COMPLETED_STATUSES = {"FT", "AET", "PEN", "AWD", "WO"}
UPCOMING_STATUSES = {"NS", "TBD", "PST", "SUSP"}


def _to_timestamp(value: str) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    return ts.tz_convert("UTC").tz_localize(None)


def _result_from_score(home: int | None, away: int | None) -> str | None:
    if home is None or away is None:
        return None
    if home > away:
        return "H"
    if away > home:
        return "A"
    return "D"


def fetch_fixtures(leagues: Iterable[int], season: int, *, db_path: Path = DB_PATH) -> None:
    client = ApiFootballClient()
    con = duckdb.connect(str(db_path))
    try:
        all_match_records: list[dict] = []
        all_fixture_records: list[dict] = []

        for league_id in leagues:
            div = LEAGUE_TO_DIV.get(league_id)
            if not div:
                raise ApiFootballError(f"No div mapping configured for league id {league_id}")

            response = client.get(
                "/fixtures",
                params={
                    "league": league_id,
                    "season": season,
                    "from": f"{season}-07-01",
                    "to": f"{season + 1}-06-30",
                },
            )

            fixtures = response.get("response", [])
            for entry in fixtures:
                fixture_info = entry.get("fixture", {})
                teams_info = entry.get("teams", {})
                goals_info = entry.get("goals", {})

                fixture_id = int(fixture_info.get("id"))
                kickoff = _to_timestamp(fixture_info.get("date"))
                status_short = (fixture_info.get("status") or {}).get("short")

                home_team = (teams_info.get("home") or {}).get("name")
                away_team = (teams_info.get("away") or {}).get("name")

                home_goals = goals_info.get("home")
                away_goals = goals_info.get("away")

                result = _result_from_score(home_goals, away_goals)

                record = {
                    "match_id": fixture_id,
                    "div": div,
                    "season": f"{season}/{season + 1}",
                    "match_date": kickoff,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "result": result,
                    "source_file": f"api_football_{league_id}",
                }

                if status_short in COMPLETED_STATUSES and result is not None:
                    all_match_records.append(record)
                elif status_short in UPCOMING_STATUSES:
                    fixture_record = {
                        "fixt_id": fixture_id,
                        "div": div,
                        "match_date": kickoff,
                        "home_team": home_team,
                        "away_team": away_team,
                        "source_file": f"api_football_{league_id}",
                    }
                    all_fixture_records.append(fixture_record)

        if all_match_records:
            matches_df = pd.DataFrame(all_match_records)
            matches_df = matches_df.dropna(subset=["match_date", "home_team", "away_team"])
            con.register("api_matches_tmp", matches_df)
            con.execute("DELETE FROM matches WHERE match_id IN (SELECT match_id FROM api_matches_tmp)")
            con.execute(
                "INSERT INTO matches SELECT match_id, div, season, match_date, home_team, away_team, home_goals, away_goals, result, source_file FROM api_matches_tmp"
            )
            con.unregister("api_matches_tmp")

        if all_fixture_records:
            fixtures_df = pd.DataFrame(all_fixture_records)
            fixtures_df = fixtures_df.dropna(subset=["match_date", "home_team", "away_team"])
            con.register("api_fixtures_tmp", fixtures_df)
            con.execute("DELETE FROM fixtures_queue WHERE fixt_id IN (SELECT fixt_id FROM api_fixtures_tmp)")
            con.execute(
                "INSERT INTO fixtures_queue SELECT fixt_id, div, match_date, home_team, away_team, source_file FROM api_fixtures_tmp"
            )
            con.unregister("api_fixtures_tmp")

        if all_match_records:
            con.execute(
                "DELETE FROM fixtures_queue WHERE fixt_id IN (SELECT match_id FROM matches)"
            )
    finally:
        con.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch fixtures from API-Football")
    parser.add_argument(
        "--leagues",
        type=int,
        nargs="+",
        default=sorted(LEAGUE_TO_DIV.keys()),
        help="League IDs to fetch",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=DEFAULT_SEASON,
        help="Season year (start year, e.g. 2024)",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DB_PATH,
        help="Path to DuckDB database",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fetch_fixtures(args.leagues, args.season, db_path=args.db)


if __name__ == "__main__":
    main()
