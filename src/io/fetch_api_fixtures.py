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
    2: "UCL",
    3: "UEL",
    39: "E0",
    40: "E1",
    41: "E2",
    42: "E3",
    45: "EC",
    61: "F1",
    62: "F2",
    71: "BR1",
    78: "D1",
    79: "D2",
    81: "D1",
    88: "N1",
    94: "P1",
    103: "NO1",
    106: "PL1",
    113: "SW1",
    119: "DK1",
    121: "DK1",
    135: "I1",
    136: "I2",
    137: "I1",
    140: "SP1",
    141: "SP2",
    144: "B1",
    164: "IS1",
    172: "BG1",
    179: "SC0",
    180: "SC1",
    181: "SC2",
    182: "SC3",
    197: "G1",
    199: "G1",
    203: "T1",
    207: "CH1",
    209: "CH1",
    210: "HR1",
    212: "HR1",
    218: "AT1",
    220: "AT1",
    244: "FI1",
    271: "HU1",
    273: "HU1",
    283: "RO1",
    286: "RS1",
    315: "BA1",
    318: "CY1",
    329: "EE1",
    332: "SK1",
    333: "UA1",
    335: "UA1",
    342: "AM1",
    345: "CZ1",
    347: "CZ1",
    357: "IE1",
    359: "IE1",
    371: "MK1",
    373: "SI1",
    383: "IL1",
    385: "IL1",
    389: "KZ1",
    393: "MT1",
    419: "AZ1",
    529: "D1",
    555: "RO1",
    659: "IL1",
    664: "XK1",
    667: "G1",
    680: "SK1",
    727: "PL1",
    756: "MK1",
    758: "GI1",
    848: "UECL",
    1042: "GI1",
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
