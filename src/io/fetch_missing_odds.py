from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable, Sequence

import duckdb
import numpy as np
import pandas as pd

from src.integrations.api_football import ApiFootballClient, ApiFootballError

BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "database" / "betting.duckdb"
DEFAULT_LOOKAHEAD_DAYS = 7
DEFAULT_LOOKBACK_DAYS = 2
DEFAULT_MAX_FIXTURES = 500
BET_MARKET = 1  # 1X2
TARGET_DIVS = ("UCL", "UEL", "UECL")
PREFERRED_BOOKMAKERS: Sequence[str] = (
    "Pinnacle",
    "Bet365",
    "Marathonbet",
    "William Hill",
    "Unibet",
    "Bwin",
)
SLEEP_SECONDS = 1.1


def _load_target_fixtures(
    con: duckdb.DuckDBPyConnection,
    lookahead_days: int,
    lookback_days: int,
    divisions: Iterable[str] | None,
    max_fixtures: int,
) -> pd.DataFrame:
    now = pd.Timestamp.utcnow()
    start = now - pd.Timedelta(days=lookback_days)
    end = now + pd.Timedelta(days=lookahead_days)

    divs = list(divisions) if divisions else list(TARGET_DIVS)

    query = """
        WITH preview_targets AS (
            SELECT fixt_id, div, match_date, home_team, away_team
            FROM fixture_previews
            WHERE div IN ({div_placeholders})
              AND match_date BETWEEN ? AND ?
        ),
        missing AS (
            SELECT p.*
            FROM preview_targets p
            LEFT JOIN (
                SELECT DISTINCT fixt_id
                FROM odds_history
            ) o ON p.fixt_id = o.fixt_id
            WHERE o.fixt_id IS NULL
        )
        SELECT *
        FROM missing
        ORDER BY match_date
        LIMIT {limit}
    """.format(
        div_placeholders=",".join("?" for _ in divs),
        limit=max_fixtures if max_fixtures else DEFAULT_MAX_FIXTURES,
    )

    params: list[object] = [*divs, start.to_pydatetime(), end.to_pydatetime()]
    return con.execute(query, params).fetchdf()


def _fetch_odds_for_fixture(client: ApiFootballClient, fixture_id: int) -> list[dict]:
    try:
        payload = client.get(
            "/odds",
            params={
                "fixture": fixture_id,
                "bet": BET_MARKET,
            },
        )
    except ApiFootballError as exc:
        print(f"[WARN] odds fetch failed for fixture {fixture_id}: {exc}")
        return []
    return payload.get("response", []) or []


def _parse_fixture_payload(entries: list[dict], fixture_id: int) -> list[dict]:
    rows: list[dict] = []
    fetched_at = pd.Timestamp.utcnow()
    for entry in entries:
        fixture = entry.get("fixture", {}) or {}
        if int(fixture.get("id", 0)) != fixture_id:
            continue
        for bookmaker in entry.get("bookmakers", []) or []:
            bookmaker_id = bookmaker.get("id")
            bookmaker_name = bookmaker.get("name")
            for bet in bookmaker.get("bets", []) or []:
                if bet.get("id") != BET_MARKET:
                    continue
                values = bet.get("values", []) or []
                odds_map = {item.get("value"): item.get("odd") for item in values}
                def _to_float(value: str | None) -> float | None:
                    if not value:
                        return None
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        return None
                rows.append(
                    {
                        "fixt_id": int(fixture_id),
                        "bookmaker_id": bookmaker_id,
                        "bookmaker_name": bookmaker_name,
                        "market": bet.get("name") or "1X2",
                        "odds_home": _to_float(odds_map.get("Home")),
                        "odds_draw": _to_float(odds_map.get("Draw")),
                        "odds_away": _to_float(odds_map.get("Away")),
                        "fetched_at": fetched_at,
                    }
                )
    return rows


def _store_odds(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> None:
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
    con.register("new_odds_tmp", df)
    con.execute(
        "DELETE FROM odds_history WHERE fixt_id IN (SELECT DISTINCT fixt_id FROM new_odds_tmp)"
    )
    con.execute(
        "INSERT INTO odds_history SELECT fixt_id, bookmaker_id, bookmaker_name, market, odds_home, odds_draw, odds_away, fetched_at FROM new_odds_tmp"
    )
    con.unregister("new_odds_tmp")


def _select_preview_updates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df_sorted = df.sort_values("fetched_at", ascending=False).copy()
    df_sorted["bookmaker_name"] = df_sorted["bookmaker_name"].fillna("")

    def _pick(rows: pd.DataFrame) -> pd.Series:
        for name in PREFERRED_BOOKMAKERS:
            subset = rows[rows["bookmaker_name"].str.lower() == name.lower()]
            if not subset.empty:
                row = subset.iloc[0]
                return row
        return rows.iloc[0]

    picked = df_sorted.groupby("fixt_id", as_index=False).apply(_pick)
    picked = picked.reset_index(drop=True)
    return picked[["fixt_id", "odds_home", "odds_draw", "odds_away", "fetched_at"]]


def _update_preview_table(con: duckdb.DuckDBPyConnection, updates: pd.DataFrame) -> None:
    if updates.empty:
        return
    con.register("odds_updates_tmp", updates)
    con.execute(
        """
        UPDATE fixture_previews AS fp
        SET odds_home = u.odds_home,
            odds_draw = u.odds_draw,
            odds_away = u.odds_away
        FROM odds_updates_tmp u
        WHERE fp.fixt_id = u.fixt_id
        """
    )
    con.unregister("odds_updates_tmp")


def fetch_missing_odds(
    *,
    db_path: Path = DB_PATH,
    lookahead_days: int = DEFAULT_LOOKAHEAD_DAYS,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    divisions: Iterable[str] | None = None,
    max_fixtures: int = DEFAULT_MAX_FIXTURES,
    sleep: float = SLEEP_SECONDS,
) -> dict:
    con = duckdb.connect(str(db_path))
    try:
        targets = _load_target_fixtures(
            con,
            lookahead_days=lookahead_days,
            lookback_days=lookback_days,
            divisions=divisions,
            max_fixtures=max_fixtures,
        )
    finally:
        con.close()

    if targets.empty:
        return {"fixtures": 0, "rows": 0, "api_calls": 0}

    client = ApiFootballClient()
    all_rows: list[dict] = []
    api_calls = 0

    for row in targets.itertuples(index=False):
        fixture_id = int(row.fixt_id)
        payload = _fetch_odds_for_fixture(client, fixture_id)
        api_calls += 1
        rows = _parse_fixture_payload(payload, fixture_id)
        if rows:
            all_rows.extend(rows)
        time.sleep(sleep)

    if not all_rows:
        return {"fixtures": len(targets), "rows": 0, "api_calls": api_calls}

    odds_df = pd.DataFrame(all_rows)
    con = duckdb.connect(str(db_path))
    try:
        _store_odds(con, odds_df)
        preview_updates = _select_preview_updates(odds_df)
        _update_preview_table(con, preview_updates)
    finally:
        con.close()

    return {
        "fixtures": len(targets),
        "rows": len(odds_df),
        "api_calls": api_calls,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch missing odds for European fixtures")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--lookahead", type=int, default=DEFAULT_LOOKAHEAD_DAYS)
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--div", action="append", dest="divisions", help="Limit to specific divisions (repeatable)")
    parser.add_argument("--max-fixtures", type=int, default=DEFAULT_MAX_FIXTURES)
    parser.add_argument("--sleep", type=float, default=SLEEP_SECONDS)
    args = parser.parse_args()

    stats = fetch_missing_odds(
        db_path=args.db,
        lookahead_days=args.lookahead,
        lookback_days=args.lookback,
        divisions=args.divisions,
        max_fixtures=args.max_fixtures,
        sleep=args.sleep,
    )
    print(f"Fixtures scanned: {stats['fixtures']}")
    print(f"Odds rows stored: {stats['rows']}")
    print(f"API calls used: {stats['api_calls']}")


if __name__ == "__main__":
    main()
