from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable

import duckdb
import pandas as pd

from src.integrations.api_football import ApiFootballClient, ApiFootballError

BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "database" / "betting.duckdb"
DEFAULT_SEASON = date.today().year
DOMESTIC_THRESHOLD = 5
EURO_LEAGUE_IDS = {2, 3, 848}
COMPLETED_STATUSES = {"FT", "AET", "PEN", "AWD", "WO"}
UPCOMING_STATUSES = {"NS", "TBD", "PST", "SUSP"}

COUNTRY_DIV_MAP: Dict[str, tuple[str, str]] = {
    "Austria": ("AT1", "Austria 1 - Bundesliga"),
    "Armenia": ("AM1", "Armenia 1 - Premier League"),
    "Azerbaijan": ("AZ1", "Azerbaijan 1 - Premyer Liqa"),
    "Bosnia and Herzegovina": ("BA1", "Bosnia 1 - Premijer Liga"),
    "Bosnia & Herzegovina": ("BA1", "Bosnia 1 - Premijer Liga"),
    "Bosnia": ("BA1", "Bosnia 1 - Premijer Liga"),
    "Bulgaria": ("BG1", "Bulgaria 1 - Parva Liga"),
    "Croatia": ("HR1", "Croatia 1 - HNL"),
    "Cyprus": ("CY1", "Cyprus 1 - First Division"),
    "Czech Republic": ("CZ1", "Czech 1 - Fortuna Liga"),
    "Czech-Republic": ("CZ1", "Czech 1 - Fortuna Liga"),
    "Czechia": ("CZ1", "Czech 1 - Fortuna Liga"),
    "Denmark": ("DK1", "Denmark 1 - Superliga"),
    "Gibraltar": ("GI1", "Gibraltar 1 - National League"),
    "Hungary": ("HU1", "Hungary 1 - NB I"),
    "Iceland": ("IS1", "Iceland 1 - Urvalsdeild"),
    "Ireland": ("IE1", "Ireland 1 - Premier Division"),
    "Israel": ("IL1", "Israel 1 - Ligat Ha-al"),
    "Kazakhstan": ("KZ1", "Kazakhstan 1 - Premier League"),
    "Kosovo": ("XK1", "Kosovo 1 - Superleague"),
    "Malta": ("MT1", "Malta 1 - Premier League"),
    "North Macedonia": ("MK1", "North Macedonia 1 - First League"),
    "Macedonia": ("MK1", "North Macedonia 1 - First League"),
    "Norway": ("NO1", "Norway 1 - Eliteserien"),
    "Poland": ("PL1", "Poland 1 - Ekstraklasa"),
    "Romania": ("RO1", "Romania 1 - Liga I"),
    "Serbia": ("RS1", "Serbia 1 - SuperLiga"),
    "Slovakia": ("SK1", "Slovakia 1 - Fortuna Liga"),
    "Slovenia": ("SI1", "Slovenia 1 - PrvaLiga"),
    "Switzerland": ("CH1", "Switzerland 1 - Super League"),
    "Ukraine": ("UA1", "Ukraine 1 - Premier League"),
    "England": ("E0", "England 1 - Premier League"),
    "Scotland": ("SC0", "Scotland 1 - Premiership"),
    "Greece": ("G1", "Greece 1 - Super League"),
    "Italy": ("I1", "Italy 1 - Serie A"),
    "Germany": ("D1", "Germany 1 - Bundesliga"),
    "Spain": ("SP1", "Spain 1 - La Liga"),
    "Portugal": ("P1", "Portugal 1 - Primeira Liga"),
    "Turkey": ("T1", "Turkey 1 - Super Lig"),
    "Brazil": ("BR1", "Brazil 1 - Serie A"),
    "Sweden": ("SW1", "Sweden 1 - Allsvenskan"),
    "Finland": ("FI1", "Finland 1 - Veikkausliiga"),
    "Estonia": ("EE1", "Estonia 1 - Meistriliiga"),
}


COUNTRY_BASELINE = {
    "AT1": (1500.0, 18.0, 45.0),
    "AM1": (1480.0, 20.0, 45.0),
    "AZ1": (1480.0, 20.0, 45.0),
    "BA1": (1480.0, 20.0, 45.0),
    "BG1": (1490.0, 18.0, 45.0),
    "CH1": (1510.0, 18.0, 45.0),
    "CY1": (1485.0, 20.0, 45.0),
    "CZ1": (1505.0, 18.0, 45.0),
    "DK1": (1510.0, 18.0, 45.0),
    "GI1": (1440.0, 24.0, 45.0),
    "HR1": (1500.0, 18.0, 45.0),
    "HU1": (1490.0, 18.0, 45.0),
    "IE1": (1470.0, 20.0, 45.0),
    "IL1": (1490.0, 18.0, 45.0),
    "IS1": (1460.0, 20.0, 45.0),
    "KZ1": (1475.0, 20.0, 45.0),
    "MK1": (1470.0, 20.0, 45.0),
    "MT1": (1450.0, 22.0, 45.0),
    "NO1": (1500.0, 18.0, 45.0),
    "PL1": (1505.0, 18.0, 45.0),
    "RO1": (1495.0, 18.0, 45.0),
    "RS1": (1495.0, 18.0, 45.0),
    "SI1": (1475.0, 20.0, 45.0),
    "SK1": (1480.0, 20.0, 45.0),
    "UA1": (1510.0, 18.0, 45.0),
    "XK1": (1460.0, 22.0, 45.0)
}


def _to_timestamp(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.tz_convert("UTC").tz_localize(None)


def _winner_symbol(home: int | None, away: int | None) -> str | None:
    if home is None or away is None:
        return None
    if home > away:
        return "H"
    if away > home:
        return "A"
    return "D"


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _searchable(name: str) -> str:
    normalized = unicodedata.normalize('NFKD', name)
    ascii_only = ''.join(ch for ch in normalized if ch.isascii())
    cleaned = re.sub(r'[^A-Za-z0-9 ]+', ' ', ascii_only).strip()
    if cleaned:
        return cleaned
    fallback = re.sub(r'[^A-Za-z0-9 ]+', ' ', name).strip()
    return fallback or name


def _safe_print(message: str) -> None:
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode('ascii', 'ignore').decode())


TEAM_OVERRIDES = {
    'bodoglimt': {'id': 327, 'name': 'Bodo/Glimt', 'country': 'Norway'},
    'drita': {'id': 14281, 'name': 'Drita', 'country': 'Kosovo'},
}


def load_low_sample_teams(con: duckdb.DuckDBPyConnection, threshold: int) -> pd.DataFrame:
    query = """
        WITH european AS (
            SELECT DISTINCT div, home_team AS team FROM fixture_previews WHERE div IN ('UCL','UEL','UECL')
            UNION
            SELECT DISTINCT div, away_team FROM fixture_previews WHERE div IN ('UCL','UEL','UECL')
        ),
        ratings AS (
            SELECT div AS rating_div, team AS rating_team, COALESCE(matches_played, 0) AS matches_played
            FROM team_ratings
        )
        SELECT e.div AS competition,
               e.team,
               MAX(r.matches_played) FILTER (WHERE r.rating_div NOT IN ('UCL','UEL','UECL')) AS domestic_matches
        FROM european e
        LEFT JOIN ratings r ON lower(trim(e.team)) = lower(trim(r.rating_team))
        GROUP BY e.div, e.team
        HAVING COALESCE(MAX(r.matches_played) FILTER (WHERE r.rating_div NOT IN ('UCL','UEL','UECL')), 0) < ?
        ORDER BY e.team
    """
    return con.execute(query, [threshold]).fetchdf()




def fetch_team_profile(client: ApiFootballClient, name: str) -> tuple[dict | None, int]:
    queries: list[str] = []
    primary = _searchable(name)
    queries.append(primary)
    if primary != name:
        queries.append(name)
    sanitized = name.replace('/', ' ')
    if sanitized not in queries:
        queries.append(sanitized)
    separators = re.split(r'[\s/\-]+', name)
    for part in separators:
        part_clean = part.strip()
        if part_clean and part_clean not in queries:
            queries.append(part_clean)

    calls = 0
    norm_target = _normalize(name)
    override = TEAM_OVERRIDES.get(norm_target)
    if override:
        return ({'id': override['id'], 'name': override['name'], 'country': override['country'], 'from': override}, calls)
    for query in queries:
        try:
            payload = client.get("/teams", params={"search": query})
            calls += 1
        except ApiFootballError as exc:
            _safe_print(f"[WARN] failed to fetch team profile for {name} ({query}): {exc}")
            continue
        candidates = payload.get("response", []) or []
        if not candidates:
            continue
        for entry in candidates:
            team = entry.get("team") or {}
            if _normalize(team.get("name", "")) == norm_target:
                return ({'id': team.get('id'), 'name': team.get('name'), 'country': team.get('country'), 'from': entry}, calls)
        team = (candidates[0] or {}).get('team') or {}
        return ({'id': team.get('id'), 'name': team.get('name'), 'country': team.get('country'), 'from': candidates[0]}, calls)
    _safe_print(f"[WARN] no team record found for {name}")
    return (None, calls)

def derive_div_for_league(league: dict, existing_codes: set[str]) -> tuple[str, str]:
    country = league.get("country") or ""
    league_name = league.get("name") or ""
    league_id = league.get("id")
    div_info = COUNTRY_DIV_MAP.get(country)
    if div_info:
        return div_info
    base = ''.join(word[0] for word in country.upper().split() if word)[:2] or 'XX'
    digits = ''.join(ch for ch in league_name if ch.isdigit()) or '1'
    candidate = f"{base}{digits[0]}"
    suffix_ord = ord('A')
    while candidate in existing_codes:
        candidate = f"{base}{digits[0]}{chr(suffix_ord)}"
        suffix_ord += 1
    return candidate, f"{country} - {league_name}"


def ensure_competition(con: duckdb.DuckDBPyConnection, div: str, name: str) -> None:
    existing = con.execute("SELECT COUNT(*) FROM competitions WHERE div = ?", [div]).fetchone()[0]
    if existing:
        return
    baseline, k_factor, home_field = COUNTRY_BASELINE.get(div, (1490.0, 18.0, 45.0))
    region = name.split(' - ')[0]
    con.execute(
        "INSERT INTO competitions (div, name, region, baseline_elo, k_factor, home_field) VALUES (?, ?, ?, ?, ?, ?)",
        [div, name, region, baseline, k_factor, home_field],
    )


def store_matches(con: duckdb.DuckDBPyConnection, matches: list[dict], fixtures: list[dict]) -> None:
    if matches:
        matches_df = pd.DataFrame(matches)
        matches_df = matches_df.dropna(subset=["match_date", "home_team", "away_team"])
        if not matches_df.empty:
            matches_df = matches_df.sort_values("match_date").drop_duplicates("match_id", keep="last")
            con.register("enrich_matches_tmp", matches_df)
            con.execute("DELETE FROM matches WHERE match_id IN (SELECT match_id FROM enrich_matches_tmp)")
            con.execute(
                "INSERT INTO matches SELECT match_id, div, season, match_date, home_team, away_team, home_goals, away_goals, result, source_file FROM enrich_matches_tmp"
            )
            con.unregister("enrich_matches_tmp")
    if fixtures:
        fixtures_df = pd.DataFrame(fixtures)
        fixtures_df = fixtures_df.dropna(subset=["match_date", "home_team", "away_team"])
        if not fixtures_df.empty:
            fixtures_df = fixtures_df.sort_values("match_date").drop_duplicates("fixt_id", keep="last")
            con.register("enrich_fixtures_tmp", fixtures_df)
            con.execute("DELETE FROM fixtures_queue WHERE fixt_id IN (SELECT fixt_id FROM enrich_fixtures_tmp)")
            con.execute(
                "INSERT INTO fixtures_queue SELECT fixt_id, div, match_date, home_team, away_team, source_file FROM enrich_fixtures_tmp"
            )
            con.unregister("enrich_fixtures_tmp")
    if matches:
        con.execute("DELETE FROM fixtures_queue WHERE fixt_id IN (SELECT match_id FROM matches)")


def enrich_domestic(season: int, threshold: int, max_calls: int | None = None) -> dict:
    con = duckdb.connect(str(DB_PATH))
    teams_df = load_low_sample_teams(con, threshold)
    existing_codes = set(con.execute("SELECT div FROM competitions").fetchnumpy()["div"])
    con.close()

    if teams_df.empty:
        return {"api_calls": 0, "teams_processed": 0, "matches": 0, "fixtures": 0, "leagues": {}}

    client = ApiFootballClient()
    api_calls = 0
    matches: list[dict] = []
    fixtures: list[dict] = []
    new_league_map: dict[int, tuple[str, str]] = {}
    processed = 0

    for row in teams_df.itertuples(index=False):
        team_name = row.team
        if max_calls is not None and api_calls >= max_calls:
            break

        profile, calls_used = fetch_team_profile(client, team_name)
        api_calls += calls_used
        if not profile or not profile.get('id'):
            continue
        team_id = profile['id']
        team_country = profile.get('country') or ""
        try:
            payload = client.get(
                "/fixtures",
                params={
                    "team": team_id,
                    "season": season,
                    "from": f"{season}-07-01",
                    "to": f"{season + 1}-06-30",
                },
            )
        except ApiFootballError as exc:
            _safe_print(f"[WARN] fixtures fetch failed for {team_name}: {exc}")
            continue
        api_calls += 1
        entries = payload.get("response", []) or []
        if not entries:
            continue

        for entry in entries:
            league = entry.get("league") or {}
            league_id = league.get("id")
            if not league_id or league_id in EURO_LEAGUE_IDS:
                continue
            if league.get("type") and league.get("type") != "League":
                continue
            league_country = league.get("country") or team_country
            if league_country in {"World", "International"}:
                league_country = team_country

            div_code, div_name = COUNTRY_DIV_MAP.get(league_country, (None, None))
            if not div_code:
                div_code, div_name = derive_div_for_league(league, existing_codes)
                COUNTRY_DIV_MAP[league_country] = (div_code, div_name)
            existing_codes.add(div_code)
            new_league_map.setdefault(int(league_id), (div_code, div_name))

            fixture = entry.get("fixture") or {}
            fixture_id = fixture.get("id")
            if not fixture_id:
                continue
            kickoff = _to_timestamp(fixture.get("date"))
            teams = entry.get("teams") or {}
            home = (teams.get("home") or {}).get("name")
            away = (teams.get("away") or {}).get("name")
            goals = entry.get("goals") or {}
            home_goals = goals.get("home")
            away_goals = goals.get("away")
            status_short = (fixture.get("status") or {}).get("short")
            if not home or not away or kickoff is None:
                continue

            season_label = f"{season}/{season + 1}"
            result_symbol = _winner_symbol(home_goals, away_goals)

            record = {
                "match_id": int(fixture_id),
                "div": div_code,
                "season": season_label,
                "match_date": kickoff,
                "home_team": home,
                "away_team": away,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "result": result_symbol,
                "source_file": f"api_football_{league_id}",
            }

            if status_short in COMPLETED_STATUSES and result_symbol is not None:
                matches.append(record)
            elif status_short in UPCOMING_STATUSES:
                fixtures.append(
                    {
                        "fixt_id": int(fixture_id),
                        "div": div_code,
                        "match_date": kickoff,
                        "home_team": home,
                        "away_team": away,
                        "source_file": f"api_football_{league_id}",
                    }
                )
        processed += 1

    con = duckdb.connect(str(DB_PATH))
    try:
        for league_id, (div_code, div_name) in new_league_map.items():
            ensure_competition(con, div_code, div_name)
        store_matches(con, matches, fixtures)
    finally:
        con.close()

    return {
        "api_calls": api_calls,
        "teams_processed": processed,
        "matches": len(matches),
        "fixtures": len(fixtures),
        "leagues": new_league_map,
    }


def update_league_mapping(leagues: dict[int, tuple[str, str]]) -> None:
    if not leagues:
        return
    target_path = BASE_DIR / "src" / "io" / "fetch_api_fixtures.py"
    source = target_path.read_text(encoding="utf-8")

    pattern = r"LEAGUE_TO_DIV\s*=\s*{([\s\S]*?)}\n\n"
    match = re.search(pattern, source)
    if not match:
        _safe_print("[WARN] Could not locate LEAGUE_TO_DIV mapping for update")
        return
    body = match.group(1)
    entries = re.findall(r"\s*(\d+):\s*\"([^\"]+)\"", body)
    mapping = {int(k): v for k, v in entries}
    changed = False
    for league_id, (div_code, _) in leagues.items():
        if league_id not in mapping:
            mapping[league_id] = div_code
            changed = True
    if not changed:
        return
    lines = ["LEAGUE_TO_DIV = {"]
    for league_id in sorted(mapping):
        lines.append(f"    {league_id}: \"{mapping[league_id]}\",")
    lines.append("}\n\n")
    new_body = "\n".join(lines)
    updated = source[:match.start()] + new_body + source[match.end():]
    target_path.write_text(updated, encoding="utf-8")


def append_competitions_yaml(leagues: dict[int, tuple[str, str]]) -> None:
    if not leagues:
        return
    yaml_path = BASE_DIR / "config" / "competitions.yml"
    existing_text = yaml_path.read_bytes().decode('utf-8', errors='ignore')
    additions: list[str] = []
    for _, (div_code, div_name) in leagues.items():
        block = f"  {div_code}:\n    name: \"{div_name}\"\n    region: \"{div_name.split(' - ')[0]}\"\n    baseline_elo: {COUNTRY_BASELINE.get(div_code, (1490.0, 18.0, 45.0))[0]}\n    k_factor: {COUNTRY_BASELINE.get(div_code, (1490.0, 18.0, 45.0))[1]}\n    home_field: {COUNTRY_BASELINE.get(div_code, (1490.0, 18.0, 45.0))[2]}\n"
        if block.strip() not in existing_text:
            additions.append(block)
    if not additions:
        return
    with yaml_path.open("a", encoding="utf-8") as handle:
        handle.write("".join(additions))


def run_ratings_update() -> None:
    from src.ratings.pipeline import RatingsPipeline

    pipeline = RatingsPipeline()
    try:
        pipeline.update_team_ratings()
    finally:
        pipeline.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich domestic data for European teams")
    parser.add_argument("--season", type=int, default=DEFAULT_SEASON, help="Season start year (e.g. 2025)")
    parser.add_argument("--threshold", type=int, default=DOMESTIC_THRESHOLD, help="Minimum domestic matches to skip enrichment")
    parser.add_argument("--max-calls", type=int, default=None, help="Optional cap on API calls")
    args = parser.parse_args()

    stats = enrich_domestic(args.season, args.threshold, args.max_calls)
    update_league_mapping(stats["leagues"])
    append_competitions_yaml(stats["leagues"])
    run_ratings_update()

    _safe_print(f"Teams processed: {stats['teams_processed']}")
    _safe_print(f"Matches inserted: {stats['matches']}")
    _safe_print(f"Upcoming fixtures inserted: {stats['fixtures']}")
    _safe_print(f"API calls used: {stats['api_calls']}")
    if stats['leagues']:
        league_lines = [f"    {league_id}: {div}" for league_id, (div, _) in stats["leagues"].items()]
        _safe_print("Leagues mapped:\n" + "\n".join(league_lines))


if __name__ == "__main__":
    main()
