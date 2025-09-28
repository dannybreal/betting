from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import duckdb
import pandas as pd

from src.integrations.api_football import ApiFootballClient, ApiFootballError

BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "database" / "betting.duckdb"
DATA_DIR = BASE_DIR / "data"
DEFAULT_FROM = "2025-08-25"
DEFAULT_TO = "2026-06-30"
SEASON = 2025


def load_league_map() -> Dict[str, Dict[str, str]]:
    path = DATA_DIR / "league_api_map.csv"
    if not path.exists():
        raise FileNotFoundError("league_api_map.csv not found; run fetch_team_strength once to create it")
    df = pd.read_csv(path)
    return {
        row.div: {"league_id": int(row.league_id), "name": row.name}
        for row in df.itertuples(index=False)
    }


def load_divisions(con: duckdb.DuckDBPyConnection) -> List[str]:
    query = "SELECT DISTINCT div FROM matches WHERE season = ?"
    divs = con.execute(query, ["2025/2026"]).fetchdf()["div"].tolist()
    return divs


def fetch_fixtures(client: ApiFootballClient, league_id: int, from_date: str, to_date: str) -> List[dict]:
    params = {
        "league": league_id,
        "season": SEASON,
        "from": from_date,
        "to": to_date,
    }
    try:
        payload = client.get("/fixtures", params=params)
    except ApiFootballError as exc:
        print(f"[WARN] fixtures fetch failed league {league_id}: {exc}")
        return []
    fixtures = []
    for entry in payload.get("response", []):
        fixture = entry.get("fixture", {})
        teams = entry.get("teams", {})
        home = teams.get("home", {})
        away = teams.get("away", {})
        if not fixture.get("id"):
            continue
        fixtures.append(
            {
                "fixt_id": fixture["id"],
                "match_date": fixture.get("date"),
                "home_team": home.get("name"),
                "away_team": away.get("name"),
            }
        )
    return fixtures


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch upcoming fixtures for additional leagues")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--div", action="append", help="Divisions to fetch (repeat). Defaults to new leagues.")
    parser.add_argument("--from", dest="from_date", default=DEFAULT_FROM)
    parser.add_argument("--to", dest="to_date", default=DEFAULT_TO)
    args = parser.parse_args()

    from_date = args.from_date
    to_date = args.to_date

    client = ApiFootballClient()
    league_map = load_league_map()

    con = duckdb.connect(str(args.db))
    try:
        existing_divs = load_divisions(con)
    finally:
        con.close()

    target_divs = args.div if args.div else ["NO1", "SW1", "FI1", "EE1", "BR1"]

    rows = []
    for div in target_divs:
        info = league_map.get(div)
        if not info:
            print(f"[WARN] No league map for {div}")
            continue
        league_id = info["league_id"]
        print(f"[INFO] Fetching fixtures for {div} (league {league_id})")
        fixtures = fetch_fixtures(client, league_id, from_date, to_date)
        for fx in fixtures:
            if not fx["home_team"] or not fx["away_team"]:
                continue
            rows.append(
                {
                    "fixt_id": fx["fixt_id"],
                    "div": div,
                    "match_date": fx["match_date"],
                    "home_team": fx["home_team"],
                    "away_team": fx["away_team"],
                    "source_file": f"api_football_{league_id}",
                }
            )
        print(f"[INFO] {div}: fetched {len(fixtures)} fixtures")

    if not rows:
        print("[INFO] No fixtures fetched")
        return

    df = pd.DataFrame(rows)
    df["match_date"] = pd.to_datetime(df["match_date"])
    df = df.drop_duplicates(subset=["fixt_id"])

    con = duckdb.connect(str(args.db))
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS fixtures_queue (
                fixt_id BIGINT,
                div TEXT,
                match_date TIMESTAMP,
                home_team TEXT,
                away_team TEXT,
                source_file TEXT
            )
            """
        )
        for div in target_divs:
            con.execute("DELETE FROM fixtures_queue WHERE div = ?", [div])
        con.register("fixtures_tmp", df)
        con.execute(
            "INSERT INTO fixtures_queue SELECT fixt_id, div, match_date, home_team, away_team, source_file FROM fixtures_tmp"
        )
        con.unregister("fixtures_tmp")
    finally:
        con.close()

    print(f"stored {len(df)} fixtures")


if __name__ == "__main__":
    main()
