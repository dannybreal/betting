from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import duckdb
import numpy as np
import pandas as pd

from src.integrations.api_football import ApiFootballClient, ApiFootballError

BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "database" / "betting.duckdb"
DATA_DIR = BASE_DIR / "data"
DEFAULT_FUTURE_DAYS = 3
DEFAULT_PAST_DAYS = 3
SLEEP_SECONDS = 1.1
BET_MARKET = 1  # 1X2


def _season_from_date(dt: pd.Timestamp) -> int:
    year = int(dt.year)
    return year if dt.month >= 7 else year - 1


def _load_league_map() -> Dict[str, Dict[str, int]]:
    path = DATA_DIR / "league_api_map.csv"
    if not path.exists():
        raise FileNotFoundError("league_api_map.csv not found; run fetch_team_strength first")
    df = pd.read_csv(path)
    mapping: Dict[str, Dict[str, int]] = {}
    for row in df.itertuples(index=False):
        mapping[row.div] = {"league_id": int(row.league_id)}
    return mapping


def _collect_targets(
    con: duckdb.DuckDBPyConnection,
    future_days: int,
    past_days: int,
    divisions: Iterable[str] | None = None,
) -> Tuple[Dict[Tuple[str, int, str], set], set[int]]:
    now = pd.Timestamp.utcnow()
    future_end = now + pd.Timedelta(days=future_days)
    past_start = now - pd.Timedelta(days=past_days)

    def fetch(sql: str, params: List[object]) -> pd.DataFrame:
        return con.execute(sql, params).fetchdf()

    div_filter = ""
    div_params: List[object] = []
    if divisions:
        placeholders = ",".join("?" for _ in divisions)
        div_filter = f" AND div IN ({placeholders})"
        div_params.extend(divisions)

    future_sql = (
        "SELECT fixt_id, div, match_date FROM fixtures_queue "
        "WHERE match_date BETWEEN ? AND ?" + div_filter
    )
    future_df = fetch(
        future_sql,
        [now.to_pydatetime(), future_end.to_pydatetime(), *div_params],
    )

    past_sql = (
        "SELECT fixt_id, div, match_date FROM fixture_results "
        "WHERE match_date BETWEEN ? AND ?" + div_filter
    )
    past_df = fetch(
        past_sql,
        [past_start.to_pydatetime(), now.to_pydatetime(), *div_params],
    )

    requests: Dict[Tuple[str, int, str], set] = {}
    fixture_ids: set[int] = set()

    for source_df in (future_df, past_df):
        for row in source_df.itertuples(index=False):
            match_ts = pd.Timestamp(row.match_date)
            key = (row.div, _season_from_date(match_ts), match_ts.strftime("%Y-%m-%d"))
            fixture_ids.add(int(row.fixt_id))
            requests.setdefault(key, set()).add(int(row.fixt_id))

    return requests, fixture_ids


def _fetch_odds_for_request(client: ApiFootballClient, league_id: int, season: int, date_str: str) -> List[dict]:
    rows: List[dict] = []
    page = 1
    while True:
        params = {
            "league": league_id,
            "season": season,
            "date": date_str,
            "bet": BET_MARKET,
            "page": page,
        }
        try:
            payload = client.get("/odds", params=params)
        except ApiFootballError as exc:
            if "rateLimit" in str(exc):
                print(f"[WARN] rate limited on league {league_id} {date_str}; stopping page iteration")
                break
            print(f"[WARN] odds fetch failed league {league_id} {date_str} page {page}: {exc}")
            break

        response = payload.get("response", []) or []
        rows.extend(response)
        paging = payload.get("paging", {}) or {}
        total_pages = int(paging.get("total", 1) or 1)
        if page >= total_pages:
            break
        page += 1
        time.sleep(SLEEP_SECONDS)
    return rows


def _parse_odds_payload(payload: List[dict], target_fixture_ids: set[int]) -> List[dict]:
    rows: List[dict] = []
    fetched_at = pd.Timestamp.utcnow()
    for entry in payload:
        fixture = entry.get("fixture", {}) or {}
        fixture_id = fixture.get("id")
        if fixture_id is None or int(fixture_id) not in target_fixture_ids:
            continue
        for bookmaker in entry.get("bookmakers", []) or []:
            bookmaker_id = bookmaker.get("id")
            bookmaker_name = bookmaker.get("name")
            for bet in bookmaker.get("bets", []) or []:
                if bet.get("id") != BET_MARKET:
                    continue
                odds_map = {value.get("value"): value.get("odd") for value in bet.get("values", [])}
                rows.append(
                    {
                        "fixt_id": int(fixture_id),
                        "bookmaker_id": bookmaker_id,
                        "bookmaker_name": bookmaker_name,
                        "market": bet.get("name", "1X2"),
                        "odds_home": float(odds_map.get("Home")) if odds_map.get("Home") else None,
                        "odds_draw": float(odds_map.get("Draw")) if odds_map.get("Draw") else None,
                        "odds_away": float(odds_map.get("Away")) if odds_map.get("Away") else None,
                        "fetched_at": fetched_at,
                    }
                )
    return rows


def store_odds(df: pd.DataFrame, db_path: Path) -> None:
    if df.empty:
        return
    fixture_ids = [int(v) for v in df["fixt_id"].tolist()]
    unique_ids = sorted(set(fixture_ids))
    placeholders = ",".join("?" for _ in unique_ids)

    con = duckdb.connect(str(db_path))
    try:
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
        if unique_ids:
            con.execute(f"DELETE FROM odds_history WHERE fixt_id IN ({placeholders})", unique_ids)
        con.register("odds_tmp", df)
        con.execute(
            "INSERT INTO odds_history SELECT fixt_id, bookmaker_id, bookmaker_name, market, odds_home, odds_draw, odds_away, fetched_at FROM odds_tmp"
        )
        con.unregister("odds_tmp")
    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch odds for fixtures")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--future-days", type=int, default=DEFAULT_FUTURE_DAYS)
    parser.add_argument("--past-days", type=int, default=DEFAULT_PAST_DAYS)
    parser.add_argument("--div", action="append", dest="divisions", help="Limit to specific divisions")
    parser.add_argument("--sleep", type=float, default=SLEEP_SECONDS)
    args = parser.parse_args()

    mapping = _load_league_map()

    con = duckdb.connect(str(args.db))
    try:
        requests, fixture_ids = _collect_targets(
            con,
            future_days=args.future_days,
            past_days=args.past_days,
            divisions=args.divisions,
        )
    finally:
        con.close()

    if not requests:
        print("[INFO] No fixtures in requested window")
        return

    client = ApiFootballClient()
    all_rows: List[dict] = []
    for (div, season_year, date_str), fixtures in sorted(requests.items(), key=lambda x: x[0]):
        league_info = mapping.get(div)
        if not league_info:
            continue
        league_id = league_info["league_id"]
        print(f"[INFO] Fetching odds for {div} (league {league_id}) date {date_str} season {season_year}")
        payload = _fetch_odds_for_request(client, league_id, season_year, date_str)
        rows = _parse_odds_payload(payload, fixtures)
        if rows:
            all_rows.extend(rows)
        time.sleep(args.sleep)

    if not all_rows:
        print("[INFO] No odds data fetched")
        return

    df = pd.DataFrame(all_rows)
    store_odds(df, args.db)
    print(f"stored odds rows: {len(df)} across {df['fixt_id'].nunique()} fixtures")


if __name__ == "__main__":
    main()
