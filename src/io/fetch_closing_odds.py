from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import duckdb
import pandas as pd

from src.integrations.api_football import ApiFootballClient, ApiFootballError

BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "database" / "betting.duckdb"
DATA_DIR = BASE_DIR / "data"
DEFAULT_FROM = pd.Timestamp.utcnow() - timedelta(hours=24)
DEFAULT_TO = pd.Timestamp.utcnow() + timedelta(hours=24)
DEFAULT_MARKET = 1  # Match winner (1X2)


def load_league_map() -> Dict[str, Dict[str, str]]:
    path = DATA_DIR / "league_api_map.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {
        row.div: {"league_id": int(row.league_id), "name": row.name}
        for row in df.itertuples(index=False)
    }


def fixtures_to_query(
    con: duckdb.DuckDBPyConnection,
    from_ts: pd.Timestamp,
    to_ts: pd.Timestamp,
    divisions: Optional[List[str]] = None,
) -> pd.DataFrame:
    query = """
        SELECT fixt_id, div, match_date
        FROM fixtures_queue
        WHERE match_date BETWEEN ? AND ?
    """
    params: List[object] = [from_ts.to_pydatetime(), to_ts.to_pydatetime()]
    if divisions:
        placeholders = ",".join("?" for _ in divisions)
        query += f" AND div IN ({placeholders})"
        params.extend(divisions)
    return con.execute(query, params).fetchdf()


def fetch_odds_for_fixture(
    client: ApiFootballClient,
    fixture_id: int,
    market: int,
    bookmakers: Optional[List[int]] = None,
) -> Iterable[dict]:
    params = {"fixture": fixture_id, "bet": market}
    try:
        payload = client.get("/odds", params=params)
    except ApiFootballError as exc:
        print(f"[WARN] odds fetch failed for fixture {fixture_id}: {exc}")
        return []

    rows: List[dict] = []
    for entry in payload.get("response", []):
        bookmaker = entry.get("bookmaker", {})
        bookmaker_id = bookmaker.get("id")
        if bookmakers and bookmaker_id not in bookmakers:
            continue
        bets = entry.get("bets", [])
        for bet in bets:
            if bet.get("id") != market:
                continue
            odds_map = {value.get("value"): value.get("odd") for value in bet.get("values", [])}
            rows.append(
                {
                    "fixt_id": fixture_id,
                    "bookmaker_id": bookmaker_id,
                    "bookmaker_name": bookmaker.get("name"),
                    "market": bet.get("name", "1X2"),
                    "odds_home": _to_float(odds_map.get("Home")),
                    "odds_draw": _to_float(odds_map.get("Draw")),
                    "odds_away": _to_float(odds_map.get("Away")),
                    "fetched_at": pd.Timestamp.utcnow(),
                }
            )
    return rows


def _to_float(value: Optional[object]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except ValueError:
        return None


def store_odds(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> None:
    if df.empty:
        return
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS odds_history (
            fixt_id BIGINT,
            bookmaker_id INTEGER,
            bookmaker_name TEXT,
            market TEXT,
            odds_home DOUBLE,
            odds_draw DOUBLE,
            odds_away DOUBLE,
            fetched_at TIMESTAMP
        )
        """
    )
    con.register("odds_tmp", df)
    con.execute(
        "INSERT INTO odds_history SELECT fixt_id, bookmaker_id, bookmaker_name, market, odds_home, odds_draw, odds_away, fetched_at FROM odds_tmp"
    )
    con.unregister("odds_tmp")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch closing odds for fixtures")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--from", dest="from_ts", type=str, default=None, help="ISO datetime start")
    parser.add_argument("--to", dest="to_ts", type=str, default=None, help="ISO datetime end")
    parser.add_argument("--div", action="append", dest="divs", help="Divisions to include")
    parser.add_argument("--bookmaker", action="append", dest="bookmakers", type=int, help="Bookmaker IDs to keep")
    parser.add_argument("--market", dest="market", type=int, default=DEFAULT_MARKET)
    args = parser.parse_args()

    from_ts = pd.Timestamp(args.from_ts) if args.from_ts else DEFAULT_FROM
    to_ts = pd.Timestamp(args.to_ts) if args.to_ts else DEFAULT_TO

    client = ApiFootballClient()

    con = duckdb.connect(str(args.db))
    try:
        fixtures = fixtures_to_query(con, from_ts, to_ts, args.divs)
    finally:
        con.close()

    if fixtures.empty:
        print("[INFO] No fixtures found in the requested window")
        return

    all_rows: List[dict] = []
    for row in fixtures.itertuples(index=False):
        rows = fetch_odds_for_fixture(client, row.fixt_id, args.market, args.bookmakers)
        all_rows.extend(rows)

    if not all_rows:
        print("[INFO] No odds fetched")
        return

    df = pd.DataFrame(all_rows)
    con = duckdb.connect(str(args.db))
    try:
        store_odds(con, df)
    finally:
        con.close()

    print(f"stored {len(df)} odds rows")


if __name__ == "__main__":
    main()
