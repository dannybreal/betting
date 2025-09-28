from __future__ import annotations

import argparse
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import duckdb
import pandas as pd

from src.integrations.api_football import ApiFootballClient, ApiFootballError

BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "database" / "betting.duckdb"
DATA_DIR = BASE_DIR / "data"
SEASON_LABEL = "2025/2026"
SEASON_YEAR = 2025
SLEEP_SECONDS = 0.1
MAX_PLAYER_PAGES = 1
START_DATE = pd.Timestamp('2025-08-25')


def load_divisions(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    query = """
        SELECT DISTINCT div, name, region
        FROM (
            SELECT m.div, c.name, c.region
            FROM matches m
            LEFT JOIN competitions c USING(div)
            WHERE m.season = ?
              AND m.match_date >= ?
            UNION
            SELECT f.div, c.name, c.region
            FROM fixtures_queue f
            LEFT JOIN competitions c USING(div)
            WHERE f.match_date >= ?
        )
    """
    return con.execute(query, [SEASON_LABEL, START_DATE, START_DATE]).fetchdf()


def load_league_map() -> Dict[str, Dict[str, str]]:
    path = DATA_DIR / "league_api_map.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {
        row.div: {"league_id": int(row.league_id), "name": row.name}
        for row in df.itertuples(index=False)
    }


def load_division_teams(con: duckdb.DuckDBPyConnection) -> Dict[str, set[str]]:
    query = """
        SELECT div, team
        FROM (
            SELECT div, home_team AS team
            FROM matches
            WHERE season = ?
              AND match_date >= ?
            UNION
            SELECT div, away_team AS team
            FROM matches
            WHERE season = ?
              AND match_date >= ?
            UNION
            SELECT div, home_team AS team
            FROM fixtures_queue
            WHERE match_date >= ?
            UNION
            SELECT div, away_team AS team
            FROM fixtures_queue
            WHERE match_date >= ?
        )
    """
    df = con.execute(query, [SEASON_LABEL, START_DATE, SEASON_LABEL, START_DATE, START_DATE, START_DATE]).fetchdf()
    mapping: Dict[str, set[str]] = defaultdict(set)
    for row in df.itertuples(index=False):
        mapping[row.div].add(row.team)
    return mapping


def resolve_league_id(
    client: ApiFootballClient,
    div: str,
    name: Optional[str],
    region: Optional[str],
    cache: Dict[str, Dict[str, str]],
) -> Optional[int]:
    if div in cache:
        return cache[div]["league_id"]

    search_term = name or div
    try:
        payload = client.get("/leagues", params={"search": search_term})
    except ApiFootballError as exc:
        print(f"[WARN] Failed to resolve league {div}: {exc}")
        return None

    best_id: Optional[int] = None
    best_name: str = search_term
    target_region = (region or "").lower()
    for entry in payload.get("response", []):
        league = entry.get("league", {})
        country = entry.get("country", {})
        seasons = entry.get("seasons", [])
        season_years = {s.get("year") for s in seasons}
        if SEASON_YEAR not in season_years:
            continue
        league_name = league.get("name", "")
        country_name = (country.get("name") or "").lower()
        if target_region and target_region not in country_name and target_region not in league_name.lower():
            continue
        best_id = league.get("id")
        best_name = league_name
        break

    if best_id is None:
        print(f"[WARN] Could not match league id for {div} ({search_term})")
        return None

    cache[div] = {"league_id": best_id, "name": best_name}
    persist_league_map(cache)
    return best_id


def fetch_teams(client: ApiFootballClient, league_id: int) -> List[Dict[str, str]]:
    payload = client.get("/teams", params={"league": league_id, "season": SEASON_YEAR})
    teams = []
    for entry in payload.get("response", []):
        team_info = entry.get("team", {})
        team_id = team_info.get("id")
        name = team_info.get("name")
        if team_id and name:
            teams.append({"team_id": team_id, "name": name})
    return teams


def _float(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except ValueError:
        return None


def fetch_team_statistics(
    client: ApiFootballClient,
    league_id: int,
    team_id: int,
) -> Dict[str, Optional[float]]:
    payload = client.get(
        "/teams/statistics",
        params={"league": league_id, "season": SEASON_YEAR, "team": team_id},
    )
    data = payload.get("response", {})
    fixtures = data.get("fixtures", {})
    goals = data.get("goals", {})

    played = fixtures.get("played", {})
    wins = fixtures.get("wins", {})
    draws = fixtures.get("draws", {})
    loses = fixtures.get("loses", {})

    goals_for_avg = goals.get("for", {}).get("average", {})
    goals_against_avg = goals.get("against", {}).get("average", {})
    goals_for_total = goals.get("for", {}).get("total", {})
    goals_against_total = goals.get("against", {}).get("total", {})

    stats = {
        "played_total": played.get("total", 0),
        "wins_total": wins.get("total", 0),
        "draws_total": draws.get("total", 0),
        "losses_total": loses.get("total", 0),
        "goals_for_avg": _float(goals_for_avg.get("total")),
        "goals_against_avg": _float(goals_against_avg.get("total")),
        "goals_for_total": goals_for_total.get("total"),
        "goals_against_total": goals_against_total.get("total"),
        "form": data.get("form"),
        "clean_sheets": data.get("clean_sheet", {}).get("total", {}),
        "failed_to_score": data.get("failed_to_score", {}).get("total", {}),
    }

    wins_total = stats["wins_total"] or 0
    draws_total = stats["draws_total"] or 0
    losses_total = stats["losses_total"] or 0
    played_total = stats["played_total"] or 0
    if played_total:
        stats["win_pct"] = wins_total / played_total
        stats["draw_pct"] = draws_total / played_total
    else:
        stats["win_pct"] = stats["draw_pct"] = None

    if stats["goals_for_avg"] is not None and stats["goals_against_avg"] is not None:
        stats["avg_goal_diff"] = stats["goals_for_avg"] - stats["goals_against_avg"]
    else:
        stats["avg_goal_diff"] = None

    return stats


def fetch_player_strength(
    client: ApiFootballClient,
    team_id: int,
) -> Dict[str, Optional[float]]:
    page = 1
    total_pages = 1
    rating_sum = 0.0
    rating_count = 0
    weighted_sum = 0.0
    minutes_sum = 0.0

    while page <= total_pages and page <= MAX_PLAYER_PAGES:
        payload = client.get(
            "/players",
            params={"team": team_id, "season": SEASON_YEAR, "page": page},
        )
        paging = payload.get("paging", {})
        total_pages = paging.get("total", 1) or 1

        for entry in payload.get("response", []):
            for stat in entry.get("statistics", []):
                rating = _float(stat.get("games", {}).get("rating"))
                minutes = stat.get("games", {}).get("minutes") or 0
                if rating is not None:
                    rating_sum += rating
                    rating_count += 1
                    weighted_sum += rating * minutes
                    minutes_sum += minutes

        page += 1
        time.sleep(SLEEP_SECONDS / 2)

    avg_rating = rating_sum / rating_count if rating_count else None
    weighted_rating = weighted_sum / minutes_sum if minutes_sum else avg_rating
    return {
        "avg_player_rating": avg_rating,
        "weighted_player_rating": weighted_rating,
        "players_sample": rating_count,
        "minutes_total": minutes_sum,
    }


def build_strength_row(
    div: str,
    team_name: str,
    team_id: int,
    stats: Dict[str, Optional[float]],
    player: Dict[str, Optional[float]],
) -> Dict[str, Optional[float]]:
    win_pct = stats.get("win_pct")
    avg_goal_diff = stats.get("avg_goal_diff")
    player_rating = player.get("weighted_player_rating") or player.get("avg_player_rating")

    components = []
    if avg_goal_diff is not None:
        components.append(avg_goal_diff)
    if win_pct is not None:
        components.append(win_pct - 0.33)
    if player_rating is not None:
        components.append((player_rating - 6.5) / 2.0)

    strength_raw = sum(components) / len(components) if components else None

    return {
        "div": div,
        "team": team_name,
        "team_id": team_id,
        "season": SEASON_LABEL,
        "played": stats.get("played_total"),
        "wins": stats.get("wins_total"),
        "draws": stats.get("draws_total"),
        "losses": stats.get("losses_total"),
        "goals_for_avg": stats.get("goals_for_avg"),
        "goals_against_avg": stats.get("goals_against_avg"),
        "win_pct": win_pct,
        "draw_pct": stats.get("draw_pct"),
        "avg_goal_diff": avg_goal_diff,
        "avg_player_rating": player.get("avg_player_rating"),
        "weighted_player_rating": player.get("weighted_player_rating"),
        "players_sample": player.get("players_sample"),
        "minutes_total": player.get("minutes_total"),
        "strength_raw": strength_raw,
        "form": stats.get("form"),
        "fetched_at": pd.Timestamp.utcnow(),
    }


def normalise_strength(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        df["strength_index"] = None
        return df
    def _norm(group: pd.DataFrame) -> pd.DataFrame:
        values = group["strength_raw"].dropna()
        if len(values) >= 3 and values.std() > 1e-9:
            mean = values.mean()
            std = values.std()
            group["strength_index"] = (group["strength_raw"] - mean) / std
        else:
            group["strength_index"] = group["strength_raw"]
        return group
    return df.groupby("div", group_keys=False).apply(_norm)


def store_strength(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS team_strength (
            div TEXT,
            team TEXT,
            team_id INTEGER,
            season TEXT,
            played INTEGER,
            wins INTEGER,
            draws INTEGER,
            losses INTEGER,
            goals_for_avg DOUBLE,
            goals_against_avg DOUBLE,
            win_pct DOUBLE,
            draw_pct DOUBLE,
            avg_goal_diff DOUBLE,
            avg_player_rating DOUBLE,
            weighted_player_rating DOUBLE,
            players_sample INTEGER,
            minutes_total DOUBLE,
            strength_raw DOUBLE,
            strength_index DOUBLE,
            form TEXT,
            fetched_at TIMESTAMP
        )
        """
    )
    con.execute("DELETE FROM team_strength WHERE season = ?", [SEASON_LABEL])
    con.register("strength_tmp", df)
    con.execute("INSERT INTO team_strength SELECT * FROM strength_tmp")
    con.unregister("strength_tmp")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch team statistics and player ratings")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--div", action="append", dest="divs", help="Limit fetch to specific divisions")
    args = parser.parse_args()

    target_divs = args.divs

    client = ApiFootballClient()

    con = duckdb.connect(str(args.db))
    try:
        if target_divs:
            divs_df = pd.DataFrame({"div": target_divs})
            division_teams = {div: set() for div in target_divs}
            existing_strength = con.execute("SELECT div, team FROM team_strength WHERE season = ?", [SEASON_LABEL]).fetchdf()
            for row in existing_strength.itertuples(index=False):
                if row.div in division_teams:
                    division_teams[row.div].add(row.team)
            fixtures = con.execute("SELECT div, home_team, away_team FROM fixtures_queue WHERE match_date >= ?", [START_DATE]).fetchdf()
            for row in fixtures.itertuples(index=False):
                if row.div in division_teams:
                    if row.home_team:
                        division_teams[row.div].add(row.home_team)
                    if row.away_team:
                        division_teams[row.div].add(row.away_team)
        else:
            divs_df = load_divisions(con)
            division_teams = load_division_teams(con)
    finally:
        con.close()

    league_map = load_league_map()
    rows = []

    for row in divs_df.itertuples(index=False):
        div = row.div
        needed_names = division_teams.get(div, set())
        if not needed_names:
            print(f"[INFO] Skipping {div}: no fixtures on or after {START_DATE.date()}")
            continue
        name = getattr(row, 'name', None)
        region = getattr(row, 'region', None)
        league_id = resolve_league_id(client, div, name, region, league_map)
        if not league_id:
            continue
        teams = fetch_teams(client, league_id)
        print(f"[INFO] {div}: {len(teams)} teams (league {league_id})")
        if not teams:
            continue
        needed_lookup = {name.lower(): name for name in needed_names}
        matched = []
        for team in teams:
            key = team["name"].lower()
            if key in needed_lookup:
                team_copy = dict(team)
                team_copy["matched_name"] = needed_lookup[key]
                matched.append(team_copy)
        if not matched:
            print(f"[WARN] No matching teams for {div} among {len(needed_names)} names")
            continue
        matched_names = {t["matched_name"] for t in matched}
        missing = needed_names - matched_names
        if missing:
            print(f"[INFO] {div}: unmatched team names {sorted(list(missing))[:5]}")
        for team in matched:
            team_id = team["team_id"]
            team_name = team.get("matched_name", team["name"])
            try:
                stats = fetch_team_statistics(client, league_id, team_id)
            except ApiFootballError as exc:
                print(f"[WARN] stats failed {div} {team_name}: {exc}")
                continue
            time.sleep(SLEEP_SECONDS)
            try:
                player = fetch_player_strength(client, team_id)
            except ApiFootballError as exc:
                print(f"[WARN] players failed {div} {team_name}: {exc}")
                player = {"avg_player_rating": None, "weighted_player_rating": None, "players_sample": None, "minutes_total": None}
            row_data = build_strength_row(div, team_name, team_id, stats, player)
            rows.append(row_data)
            time.sleep(SLEEP_SECONDS)

    df = pd.DataFrame(rows)
    columns = ['div', 'team', 'team_id', 'season', 'played', 'wins', 'draws', 'losses', 'goals_for_avg', 'goals_against_avg', 'win_pct', 'draw_pct', 'avg_goal_diff', 'avg_player_rating', 'weighted_player_rating', 'players_sample', 'minutes_total', 'strength_raw', 'strength_index', 'form', 'fetched_at']
    if not df.empty:
        df = normalise_strength(df)
        df = df.reindex(columns=columns)
    else:
        df = pd.DataFrame(columns=columns)

    con = duckdb.connect(str(args.db))
    try:
        store_strength(con, df)
    finally:
        con.close()

    print(f"stored {len(df)} team strength rows")


if __name__ == "__main__":
    main()
