from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from types import SimpleNamespace
from pathlib import Path
from typing import Dict, Tuple

import duckdb
import numpy as np
import pandas as pd

from . import elo, xg

BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "database" / "betting.duckdb"
REPORTS_DIR = BASE_DIR / "reports"
DATA_DIR = BASE_DIR / "data"

SCHEDULE_WEIGHT = 0.02
LUCK_WEIGHT = 4.0
ELO_SCALE = 50.0
HOME_FIELD_SCALE = 5000.0
XG_PROB_WEIGHT = 0.01
SOT_PROB_WEIGHT = 0.003
XG_DRAW_THRESHOLD = 0.4
XG_DRAW_WEIGHT = 0.035
SOT_DRAW_THRESHOLD = 1.5
SOT_DRAW_WEIGHT = 0.02
STRENGTH_WEIGHT = 30.0
STRENGTH_CLIP = 0.6
STRENGTH_HOME_ADVANTAGE = 0.05
MARKET_BLEND_WEIGHT = 0.80
MARKET_BLEND_FLOOR = 0.02
MARKET_BLEND_CEILING = 0.9
DRAW_OVERRIDE_GAP = 0.08
DRAW_OVERRIDE_MIN_PROB = 0.20
DRAW_OVERRIDE_MIN_RAW = 0.14
DRAW_OVERRIDE_SHIFT = 0.04
DEFAULT_DRAW_BASE = 0.21
EURO_DIVS = {'UCL', 'UEL', 'UEC'}
CROSSOVER_MIN_MATCHES = 20
EURO_FALLBACK_BASELINE = 1450.0
TRUSTED_MARKET_DIVS = {'E0', 'D1', 'I1', 'SP1', 'F1', 'BR1', 'UCL', 'UEL'}
MARKET_EDGE_WARN = 0.08
MARKET_EDGE_CAP = 0.15
HIGH_EDGE_MODEL_WEIGHT = 0.60
MIN_DRAW_FLOOR = 0.08
MAX_DRAW_CEILING = 0.55
DRAW_BAND = 0.12
DRAW_PRIOR_WEIGHT = 40
DRAW_SLOPE = 160.0
DRAW_CALIBRATION_BLEND = 0.4
EARLY_SEASON_THRESHOLD = 6
EARLY_SEASON_BLEND = 0.4
RECENT_DRAW_DAYS = 120
MIN_DRAW_SAMPLE = 20
STRENGTH_SEASON_FALLBACK = '2025/2026'

@dataclass
class TeamState:
    elo: float
    opp_elo_history: deque = field(default_factory=lambda: deque(maxlen=5))
    luck_history: deque = field(default_factory=lambda: deque(maxlen=5))
    schedule_adj: float = 0.0
    luck_adj: float = 0.0
    xg_for: float = 0.0
    xg_against: float = 0.0
    matches_played: int = 0
    season: str = ""
    last_match: pd.Timestamp | None = None
    form: deque = field(default_factory=lambda: deque(maxlen=5))


class RatingsPipeline:
    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self.con = duckdb.connect(str(db_path))
        self.params_by_div: Dict[str, elo.EloParameters] = {}
        self.draw_base_by_div: Dict[str, float] = {}
        self.draw_calibration: dict | None = None
        self.league_accuracy_info: Dict[str, dict] = {}
        self._load_competition_params()
        self._load_draw_rates()
        self.draw_calibration = self._load_draw_calibration()

    def _load_competition_params(self) -> None:
        df = self.con.execute("SELECT * FROM competitions").fetchdf()
        for row in df.itertuples(index=False):
            self.params_by_div[row.div] = elo.EloParameters(
                baseline=row.baseline_elo or 1500.0,
                k_factor=row.k_factor or 20.0,
                home_field=row.home_field or 50.0,
            )

    def _load_draw_rates(self) -> None:
        trend_query = f"""
            SELECT div,
                   SUM(CASE WHEN result = 'D' THEN 1 ELSE 0 END) AS draws,
                   COUNT(*) AS matches
            FROM matches
            WHERE result IS NOT NULL
              AND match_date >= current_timestamp - INTERVAL '{RECENT_DRAW_DAYS} days'
            GROUP BY div
        """
        try:
            df = self.con.execute(trend_query).fetchdf()
        except duckdb.Error:
            df = pd.DataFrame(columns=['div', 'draws', 'matches'])

        accuracy_query = """
            WITH results AS (
                SELECT fixt_id, div, result
                FROM fixture_results
                WHERE status IN ('FT', 'AET', 'PEN')
                  AND result IS NOT NULL
            ), preview_ranked AS (
                SELECT fph.*, ROW_NUMBER() OVER (PARTITION BY fph.fixt_id ORDER BY generated_at DESC) AS rn
                FROM fixture_previews_history fph
                JOIN results r ON r.fixt_id = fph.fixt_id
            ), latest_preview AS (
                SELECT *
                FROM preview_ranked
                WHERE rn = 1
            ), labelled AS (
                SELECT r.div,
                       r.result,
                       lp.prob_home,
                       lp.prob_draw,
                       lp.prob_away,
                       CASE
                           WHEN lp.prob_home >= lp.prob_draw AND lp.prob_home >= lp.prob_away THEN 'H'
                           WHEN lp.prob_away >= lp.prob_home AND lp.prob_away >= lp.prob_draw THEN 'A'
                           ELSE 'D'
                       END AS predicted
                FROM results r
                JOIN latest_preview lp ON lp.fixt_id = r.fixt_id
            )
            SELECT div,
                   COUNT(*) AS matches,
                   AVG(CASE WHEN predicted = result THEN 1 ELSE 0 END) AS accuracy,
                   AVG(CASE WHEN result = 'D' THEN 1 ELSE 0 END) AS draw_rate,
                   AVG(prob_draw) AS avg_pred_draw
            FROM labelled
            GROUP BY div
        """
        try:
            accuracy_df = self.con.execute(accuracy_query).fetchdf()
        except duckdb.Error:
            accuracy_df = pd.DataFrame(columns=['div', 'matches', 'accuracy', 'draw_rate', 'avg_pred_draw'])

        accuracy_map: Dict[str, dict] = {}
        for row in accuracy_df.itertuples(index=False):
            accuracy_map[row.div] = {
                'matches': int(row.matches or 0),
                'accuracy': float(row.accuracy or 0.0),
                'draw_rate': float(row.draw_rate or 0.0),
                'avg_pred_draw': float(row.avg_pred_draw or 0.0),
            }

        self.league_accuracy_info = accuracy_map

        for row in df.itertuples(index=False):
            if not row.matches or row.matches < MIN_DRAW_SAMPLE:
                continue
            draws = row.draws or 0
            rate = (draws + DEFAULT_DRAW_BASE * DRAW_PRIOR_WEIGHT) / (row.matches + DRAW_PRIOR_WEIGHT)
            rate = max(MIN_DRAW_FLOOR, min(MAX_DRAW_CEILING, rate))
            info = accuracy_map.get(row.div)
            if info and info['matches'] >= 30 and info['accuracy'] < 0.5:
                target = max(info['draw_rate'], info['avg_pred_draw']) + 0.03
                rate = min(MAX_DRAW_CEILING, max(rate, target))
            self.draw_base_by_div[row.div] = rate

        for div, info in accuracy_map.items():
            if div not in self.draw_base_by_div and info['matches'] >= MIN_DRAW_SAMPLE:
                fallback = max(info['draw_rate'], DEFAULT_DRAW_BASE)
                self.draw_base_by_div[div] = max(MIN_DRAW_FLOOR, min(MAX_DRAW_CEILING, fallback))

    def _load_draw_calibration(self) -> dict | None:
        try:
            row = self.con.execute("SELECT feature_names, coefficients FROM draw_calibration ORDER BY created_at DESC LIMIT 1").fetchone()
        except duckdb.Error:
            return None
        if not row:
            return None
        raw_features, raw_coeffs = row
        if isinstance(raw_features, list):
            feature_names = list(raw_features)
        else:
            try:
                feature_names = list(json.loads(raw_features))
            except (TypeError, json.JSONDecodeError):
                feature_names = []
        if isinstance(raw_coeffs, list):
            coeff_list = list(raw_coeffs)
        else:
            try:
                coeff_list = list(json.loads(raw_coeffs))
            except (TypeError, json.JSONDecodeError):
                coeff_list = []
        if not feature_names:
            return None
        coeffs = np.array(coeff_list, dtype=float) if coeff_list else None
        if coeffs is None or len(feature_names) != len(coeffs):
            return None
        return {"features": feature_names, "coefficients": coeffs}

    def _load_market_probabilities(self) -> dict[int, tuple[float, float, float]]:
        query = """
            WITH latest AS (
                SELECT fixt_id,
                       odds_home,
                       odds_draw,
                       odds_away,
                       ROW_NUMBER() OVER (PARTITION BY fixt_id ORDER BY fetched_at DESC) AS rn
                FROM odds_history
            )
            SELECT fixt_id,
                   odds_home,
                   odds_draw,
                   odds_away
            FROM latest
            WHERE rn = 1
        """
        try:
            df = self.con.execute(query).fetchdf()
        except duckdb.Error:
            return {}
        market: dict[int, tuple[float, float, float]] = {}
        for row in df.itertuples(index=False):
            odds_home = row.odds_home
            odds_draw = row.odds_draw
            odds_away = row.odds_away
            if odds_home is None or odds_draw is None or odds_away is None:
                continue
            if odds_home <= 1.0 or odds_draw <= 1.0 or odds_away <= 1.0:
                continue
            inv_home = 1.0 / float(odds_home)
            inv_draw = 1.0 / float(odds_draw)
            inv_away = 1.0 / float(odds_away)
            total = inv_home + inv_draw + inv_away
            if total <= 0:
                continue
            market[row.fixt_id] = (
                inv_home / total,
                inv_draw / total,
                inv_away / total,
            )
        return market

    def _blend_with_market(
        self,
        model_probs: tuple[float, float, float],
        market_probs: tuple[float, float, float],
        model_weight: float | None = None,
    ) -> tuple[float, float, float]:
        weight = MARKET_BLEND_WEIGHT if model_weight is None else float(model_weight)
        weight = max(0.0, min(1.0, weight))
        market_weight = 1.0 - weight
        mixed = [
            weight * m + market_weight * mk
            for m, mk in zip(model_probs, market_probs)
        ]
        total = sum(mixed)
        if total <= 0:
            return model_probs
        normalized = [val / total for val in mixed]
        normalized = [
            max(MARKET_BLEND_FLOOR, min(MARKET_BLEND_CEILING, val))
            for val in normalized
        ]
        norm_sum = sum(normalized)
        if norm_sum <= 0:
            return model_probs
        return tuple(val / norm_sum for val in normalized)

    def _calibrate_draw_probability(
        self,
        p_draw_base: float,
        base_draw_rate: float,
        elo_diff: float,
        home_strength: float | None,
        away_strength: float | None,
        home_xg: float,
        away_xg: float,
        home_sot: float,
        away_sot: float,
    ) -> float | None:
        calibration = self.draw_calibration
        if not calibration:
            return None
        vector: list[float] = []
        for name in calibration["features"]:
            if name == "intercept":
                vector.append(1.0)
            elif name == "abs_elo":
                vector.append(abs(elo_diff))
            elif name == "strength_gap":
                if home_strength is None or away_strength is None:
                    vector.append(0.0)
                else:
                    vector.append(abs(home_strength - away_strength))
            elif name == "xg_gap":
                vector.append(abs(home_xg - away_xg))
            elif name == "sot_gap":
                vector.append(abs(home_sot - away_sot))
            elif name in ("prob_draw_raw", "prob_draw"):
                vector.append(p_draw_base)
            elif name == "draw_rate":
                vector.append(base_draw_rate)
            else:
                vector.append(0.0)
        vec = np.asarray(vector, dtype=float)
        coeffs = calibration["coefficients"]
        if vec.shape[0] != coeffs.shape[0]:
            return None
        if not np.all(np.isfinite(vec)):
            return None
        z = float(np.dot(coeffs, vec))
        return 1.0 / (1.0 + np.exp(-z))

    def _matches_with_stats(self) -> pd.DataFrame:
        matches = self.con.execute(
            "SELECT match_id, div, season, match_date, home_team, away_team, home_goals, away_goals, result FROM matches ORDER BY match_date"
        ).fetchdf()
        if matches.empty:
            return matches
        stats = self.con.execute("SELECT match_id, stat_name, stat_value FROM match_stats").fetchdf()
        if not stats.empty:
            pivot = stats.pivot(index="match_id", columns="stat_name", values="stat_value")
            matches = matches.join(pivot, on="match_id")
        return matches

    def update_team_ratings(self) -> pd.DataFrame:
            matches = self._matches_with_stats()
            if matches.empty:
                return pd.DataFrame()

            def _apply_adjustments(state: TeamState, params: elo.EloParameters) -> None:
                prev_total = state.schedule_adj + state.luck_adj
                if prev_total:
                    state.elo -= prev_total
                schedule_adj = 0.0
                if state.opp_elo_history:
                    avg_opp = sum(state.opp_elo_history) / len(state.opp_elo_history)
                    schedule_adj = (avg_opp - params.baseline) * SCHEDULE_WEIGHT
                luck_adj = 0.0
                if state.luck_history:
                    avg_luck = sum(state.luck_history) / len(state.luck_history)
                    luck_adj = -avg_luck * LUCK_WEIGHT
                state.schedule_adj = schedule_adj
                state.luck_adj = luck_adj
                state.elo += schedule_adj + luck_adj

            team_states: Dict[Tuple[str, str], TeamState] = {}
            results_by_div: Dict[str, set[str]] = defaultdict(set)

            for row in matches.itertuples(index=False):
                params = self.params_by_div.get(row.div, elo.EloParameters())
                home_key = (row.div, row.home_team)
                away_key = (row.div, row.away_team)

                home_state = team_states.get(home_key)
                if home_state is None:
                    home_state = TeamState(elo=params.baseline, season=row.season)
                    team_states[home_key] = home_state
                elif home_state.season != row.season:
                    home_state.elo = elo.regress_towards_baseline(home_state.elo, params)
                    home_state.xg_for *= 0.6
                    home_state.xg_against *= 0.6
                    home_state.matches_played = max(home_state.matches_played // 2, 0)
                    home_state.season = row.season
                    home_state.form.clear()
                    home_state.opp_elo_history.clear()
                    home_state.luck_history.clear()
                    home_state.schedule_adj = 0.0
                    home_state.luck_adj = 0.0

                away_state = team_states.get(away_key)
                if away_state is None:
                    away_state = TeamState(elo=params.baseline, season=row.season)
                    team_states[away_key] = away_state
                elif away_state.season != row.season:
                    away_state.elo = elo.regress_towards_baseline(away_state.elo, params)
                    away_state.xg_for *= 0.6
                    away_state.xg_against *= 0.6
                    away_state.matches_played = max(away_state.matches_played // 2, 0)
                    away_state.season = row.season
                    away_state.form.clear()
                    away_state.opp_elo_history.clear()
                    away_state.luck_history.clear()
                    away_state.schedule_adj = 0.0
                    away_state.luck_adj = 0.0

                home_pre_elo = home_state.elo
                away_pre_elo = away_state.elo

                home_state.opp_elo_history.append(away_pre_elo)
                away_state.opp_elo_history.append(home_pre_elo)

                if row.home_goals > row.away_goals:
                    result_value = 1.0
                    home_result = "W"
                    away_result = "L"
                elif row.home_goals < row.away_goals:
                    result_value = 0.0
                    home_result = "L"
                    away_result = "W"
                else:
                    result_value = 0.5
                    home_result = away_result = "D"

                home_shots = getattr(row, "home_shots", None)
                home_sot = getattr(row, "home_shots_on_target", None)
                away_shots = getattr(row, "away_shots", None)
                away_sot = getattr(row, "away_shots_on_target", None)

                home_xg = xg.estimate_xg(home_shots, home_sot, row.home_goals)
                away_xg = xg.estimate_xg(away_shots, away_sot, row.away_goals)

                home_luck = (row.home_goals - row.away_goals) - (home_xg - away_xg)
                away_luck = (row.away_goals - row.home_goals) - (away_xg - home_xg)
                home_state.luck_history.append(home_luck)
                away_state.luck_history.append(away_luck)

                home_state.elo, away_state.elo = elo.update_pair(
                    home_state.elo,
                    away_state.elo,
                    result_value,
                    params,
                )

                home_state.matches_played += 1
                away_state.matches_played += 1
                home_state.xg_for = xg.update_mean(home_state.xg_for, home_state.matches_played - 1, home_xg)
                home_state.xg_against = xg.update_mean(home_state.xg_against, home_state.matches_played - 1, away_xg)
                away_state.xg_for = xg.update_mean(away_state.xg_for, away_state.matches_played - 1, away_xg)
                away_state.xg_against = xg.update_mean(away_state.xg_against, away_state.matches_played - 1, home_xg)

                if home_state.matches_played < EARLY_SEASON_THRESHOLD:
                    blend = EARLY_SEASON_BLEND * (EARLY_SEASON_THRESHOLD - home_state.matches_played) / EARLY_SEASON_THRESHOLD
                    if blend > 0:
                        home_state.elo = home_state.elo * (1 - blend) + params.baseline * blend
                if away_state.matches_played < EARLY_SEASON_THRESHOLD:
                    blend = EARLY_SEASON_BLEND * (EARLY_SEASON_THRESHOLD - away_state.matches_played) / EARLY_SEASON_THRESHOLD
                    if blend > 0:
                        away_state.elo = away_state.elo * (1 - blend) + params.baseline * blend

                match_time = pd.to_datetime(row.match_date)
                home_state.last_match = match_time
                away_state.last_match = match_time
                home_state.form.appendleft(
                    {
                        "date": match_time.strftime("%Y-%m-%d"),
                        "opponent": row.away_team,
                        "venue": "H",
                        "score": f"{row.home_goals}-{row.away_goals}",
                        "result": home_result,
                    }
                )
                away_state.form.appendleft(
                    {
                        "date": match_time.strftime("%Y-%m-%d"),
                        "opponent": row.home_team,
                        "venue": "A",
                        "score": f"{row.away_goals}-{row.home_goals}",
                        "result": away_result,
                    }
                )

                _apply_adjustments(home_state, params)
                _apply_adjustments(away_state, params)

                results_by_div[row.div].add(row.season)

            records = []
            for (div_code, team_name), state in team_states.items():
                updated = state.last_match if state.last_match is not None else pd.Timestamp.utcnow()
                records.append(
                    {
                        "div": div_code,
                        "team": team_name,
                        "season": state.season,
                        "elo": round(state.elo, 2),
                        "xg_for": round(state.xg_for, 2),
                        "xg_against": round(state.xg_against, 2),
                        "matches_played": int(state.matches_played),
                        "updated_at": updated,
                        "rolling_form": json.dumps(list(state.form)),
                        "schedule_adj": round(state.schedule_adj, 2),
                        "luck_adj": round(state.luck_adj, 2),
                    }
                )

            ratings_df = pd.DataFrame.from_records(records)
            if ratings_df.empty:
                return ratings_df

            self.con.register("ratings_tmp", ratings_df)
            divs = ",".join(f"'{d}'" for d in ratings_df["div"].unique())
            self.con.execute(f"DELETE FROM team_ratings WHERE div IN ({divs})")
            self.con.execute(
                "INSERT INTO team_ratings SELECT div, team, season, elo, xg_for, xg_against, matches_played, updated_at, rolling_form, schedule_adj, luck_adj FROM ratings_tmp"
            )
            self.con.unregister("ratings_tmp")

            REPORTS_DIR.mkdir(exist_ok=True)
            ratings_df.sort_values(["div", "elo"], ascending=[True, False]).to_csv(
                REPORTS_DIR / "team_ratings.csv", index=False
            )

            return ratings_df


    def generate_previews(self) -> pd.DataFrame:
        fixtures = self.con.execute(
            "SELECT fixt_id, div, match_date, home_team, away_team, source_file FROM fixtures_queue ORDER BY match_date"
        ).fetchdf()
        if fixtures.empty:
            return pd.DataFrame()

        ratings = self.con.execute(
            """
            SELECT div, team, season, elo, xg_for, xg_against, matches_played, updated_at
            FROM (
                SELECT *,
                       ROW_NUMBER() OVER (PARTITION BY div, team ORDER BY updated_at DESC) AS rn
                FROM team_ratings
            )
            WHERE rn = 1
            """
        ).fetchdf()
        rating_map: dict[tuple[str, str], SimpleNamespace] = {}
        team_best_map: dict[str, SimpleNamespace] = {}
        for row in ratings.itertuples(index=False):
            data = row._asdict()
            state = SimpleNamespace(**data)
            rating_map[(row.div, row.team)] = state
            best = team_best_map.get(row.team)
            matches = state.matches_played or 0
            if best is None or matches > (best.matches_played or 0):
                team_best_map[row.team] = state

        season_row = self.con.execute("SELECT season FROM matches ORDER BY match_date DESC LIMIT 1").fetchone()
        season_str = season_row[0] if season_row else None
        metrics_map: Dict[Tuple[str, str], pd.Series] = {}
        market_prob_map = self._load_market_probabilities()
        if season_str:
            metrics_df = self.con.execute(
                """
                WITH xg AS (
                    SELECT m.div,
                           fs.team_name AS team,
                           m.match_date,
                           fs.stat_value AS xg_for,
                           opp.stat_value AS xg_against
                    FROM fixture_statistics fs
                    JOIN matches m ON fs.fixture_id = m.match_id
                    JOIN fixture_statistics opp
                      ON opp.fixture_id = fs.fixture_id
                     AND opp.team_name <> fs.team_name
                     AND LOWER(opp.stat_type) = 'expected_goals'
                    WHERE m.season = ?
                      AND LOWER(fs.stat_type) = 'expected_goals'
                ), sot AS (
                    SELECT m.div,
                           fs.team_name AS team,
                           m.match_date,
                           fs.stat_value AS sot
                    FROM fixture_statistics fs
                    JOIN matches m ON fs.fixture_id = m.match_id
                    WHERE m.season = ?
                      AND LOWER(fs.stat_type) = 'shots on goal'
                ), merged AS (
                    SELECT xg.div,
                           xg.team,
                           xg.match_date,
                           xg.xg_for,
                           xg.xg_against,
                           sot.sot
                    FROM xg
                    LEFT JOIN sot
                      ON sot.div = xg.div
                     AND sot.team = xg.team
                     AND sot.match_date = xg.match_date
                ), ranked AS (
                    SELECT *,
                           ROW_NUMBER() OVER (PARTITION BY div, team ORDER BY match_date DESC) AS rn
                    FROM merged
                )
                SELECT div, team,
                       AVG(xg_for) AS xg_for_avg,
                       AVG(xg_against) AS xg_against_avg,
                       AVG(sot) AS sot_avg
                FROM ranked
                WHERE rn <= 5
                GROUP BY 1,2
                """,
                [season_str, season_str],
            ).fetchdf()
            metrics_map = {
                (row.div, row.team): row for row in metrics_df.itertuples(index=False)
            }

        strength_season = season_str or STRENGTH_SEASON_FALLBACK
        strength_df = self.con.execute(
            "SELECT div, team, strength_index FROM team_strength WHERE season = ?",
            [strength_season],
        ).fetchdf()
        strength_map = {
            (row.div, row.team): row.strength_index for row in strength_df.itertuples(index=False)
        }

        strength_team_map: dict[str, float] = {}
        for row in strength_df.itertuples(index=False):
            strength_team_map.setdefault(row.team, row.strength_index)

        baseline_path = DATA_DIR / "team_strength_baselines.csv"
        if baseline_path.exists():
            baseline_df = pd.read_csv(baseline_path)
            for base_row in baseline_df.itertuples(index=False):
                value = getattr(base_row, "baseline_strength", None)
                if value is None or pd.isna(value):
                    continue
                strength_team_map.setdefault(base_row.team, float(value))

        ratings_strength_df = self.con.execute(
            "SELECT team, elo FROM team_ratings WHERE season = ?",
            [strength_season],
        ).fetchdf()
        if not ratings_strength_df.empty:
            for rating_row in ratings_strength_df.itertuples(index=False):
                elo_val = getattr(rating_row, "elo", None)
                if elo_val is None or pd.isna(elo_val):
                    continue
                strength_team_map.setdefault(
                    rating_row.team,
                    float(1.0 / (1.0 + np.exp(-(float(elo_val) - 1500.0) / 75.0)))
                )


        def _clone_state(state: SimpleNamespace | None) -> SimpleNamespace | None:
            if state is None:
                return None
            return SimpleNamespace(**state.__dict__)

        def _resolve_team_state(team: str, div: str, params: elo.EloParameters) -> SimpleNamespace:
            state = _clone_state(rating_map.get((div, team)))
            best = _clone_state(team_best_map.get(team))
            candidate = state or best
            if div in EURO_DIVS:
                base_elo = EURO_FALLBACK_BASELINE
                if best and best.div != div and (state is None or (state.matches_played or 0) < CROSSOVER_MIN_MATCHES):
                    candidate = best
                if candidate is not None and candidate.div == div:
                    matches = candidate.matches_played or 0
                    if matches < CROSSOVER_MIN_MATCHES:
                        reference = best if best and best.div != div and (best.matches_played or 0) >= CROSSOVER_MIN_MATCHES else None
                        ref_elo = reference.elo if reference else base_elo
                        blend = (CROSSOVER_MIN_MATCHES - matches) / CROSSOVER_MIN_MATCHES
                        candidate.elo = candidate.elo * (1 - blend) + ref_elo * blend
                if candidate is None:
                    candidate = SimpleNamespace(
                        div=div,
                        team=team,
                        elo=float(base_elo),
                        xg_for=None,
                        xg_against=None,
                        matches_played=0,
                        season='',
                        updated_at=None,
                    )
                return candidate
            if candidate is None:
                candidate = SimpleNamespace(
                    div=div,
                    team=team,
                    elo=float(params.baseline),
                    xg_for=None,
                    xg_against=None,
                    matches_played=0,
                    season='',
                    updated_at=None,
                )
            return candidate

        generated_at = pd.Timestamp.utcnow()
        override_records: list[dict] = []
        records = []
        for row in fixtures.itertuples(index=False):
            params = self.params_by_div.get(row.div, elo.EloParameters())
            home_state = _resolve_team_state(row.home_team, row.div, params)
            away_state = _resolve_team_state(row.away_team, row.div, params)

            home_elo = float(home_state.elo)
            away_elo = float(away_state.elo)

            home_metrics = metrics_map.get((row.div, row.home_team))
            away_metrics = metrics_map.get((row.div, row.away_team))

            def _metric_value(metric_row, attr):
                if metric_row is None:
                    return None
                value = getattr(metric_row, attr, None)
                if value is None or pd.isna(value):
                    return None
                return float(value)

            home_xg_form = _metric_value(home_metrics, 'xg_for_avg')
            away_xg_form = _metric_value(away_metrics, 'xg_for_avg')
            home_sot_form = _metric_value(home_metrics, 'sot_avg')
            away_sot_form = _metric_value(away_metrics, 'sot_avg')

            home_xg_val = home_xg_form if home_xg_form is not None else (float(home_state.xg_for) if home_state.xg_for is not None else None)
            away_xg_val = away_xg_form if away_xg_form is not None else (float(away_state.xg_for) if away_state.xg_for is not None else None)
            if home_xg_val is None:
                home_xg_val = 1.2
            if away_xg_val is None:
                away_xg_val = 1.2

            home_xg_eval = home_xg_form if home_xg_form is not None else home_xg_val
            away_xg_eval = away_xg_form if away_xg_form is not None else away_xg_val
            home_sot_eval = home_sot_form if home_sot_form is not None else 3.0
            away_sot_eval = away_sot_form if away_sot_form is not None else 3.0

            home_strength = strength_map.get((row.div, row.home_team))
            if home_strength is None:
                home_strength = strength_team_map.get(row.home_team)
            if home_strength is None:
                home_strength = strength_team_map.get(row.home_team)
            away_strength = strength_map.get((row.div, row.away_team))
            if away_strength is None:
                away_strength = strength_team_map.get(row.away_team)

            if home_strength is not None and away_strength is not None:
                strength_gap = abs(home_strength - away_strength)
            else:
                strength_gap = None

            elo_diff = home_elo - away_elo
            if home_strength is not None and away_strength is not None:
                strength_diff = (home_strength - away_strength) + STRENGTH_HOME_ADVANTAGE
                strength_diff = max(min(strength_diff, STRENGTH_CLIP), -STRENGTH_CLIP)
                elo_diff += strength_diff * STRENGTH_WEIGHT

            p_home_base = 1.0 / (1.0 + 10 ** (-elo_diff / ELO_SCALE))
            p_home_adj = p_home_base
            if params.home_field:
                p_home_adj += params.home_field / HOME_FIELD_SCALE
            if home_xg_form is not None and away_xg_form is not None:
                p_home_adj += (home_xg_form - away_xg_form) * XG_PROB_WEIGHT
            if home_sot_form is not None and away_sot_form is not None:
                p_home_adj += (home_sot_form - away_sot_form) * SOT_PROB_WEIGHT

            p_home_adj = min(max(p_home_adj, 0.01), 0.99)
            p_away_adj = 1.0 - p_home_adj

            p_home_baseline = p_home_base
            delta_xg = abs(home_xg_eval - away_xg_eval) if home_xg_eval is not None and away_xg_eval is not None else None
            delta_sot = abs(home_sot_eval - away_sot_eval) if home_sot_eval is not None and away_sot_eval is not None else None

            flatten_elo = max(0.0, 1.0 - min(1.0, abs(elo_diff) / 50.0))
            flatten_xg = max(0.0, 1.0 - min(1.0, (delta_xg if delta_xg is not None else 0.6) / 0.6))
            flatten_sot = max(0.0, 1.0 - min(1.0, (delta_sot if delta_sot is not None else 4.0) / 4.0))
            flatten_prob = max(0.0, 1.0 - min(1.0, abs(p_home_adj - 0.5) / 0.16))
            flatten_score = (
                0.45 * flatten_elo
                + 0.30 * flatten_xg
                + 0.15 * flatten_sot
                + 0.10 * flatten_prob
            )
            if flatten_score > 0.0:
                shrink = max(0.10, 1.0 - 1.4 * flatten_score)
                p_home_adj = 0.5 + (p_home_adj - 0.5) * shrink
                p_home_adj = min(max(p_home_adj, 0.05), 0.95)
                p_away_adj = 1.0 - p_home_adj

            if p_home_baseline >= 0.58 and p_home_adj < p_home_baseline - 0.08:
                p_home_adj = p_home_baseline - 0.08
            elif p_home_baseline <= 0.42 and p_home_adj > p_home_baseline + 0.08:
                p_home_adj = p_home_baseline + 0.08
            p_home_adj = min(max(p_home_adj, 0.05), 0.95)
            p_away_adj = 1.0 - p_home_adj

            draw_bonus = 0.0
            if delta_xg is not None and delta_xg < XG_DRAW_THRESHOLD:
                draw_bonus += (XG_DRAW_THRESHOLD - delta_xg) * XG_DRAW_WEIGHT
            if delta_sot is not None and delta_sot < SOT_DRAW_THRESHOLD:
                draw_bonus += (SOT_DRAW_THRESHOLD - delta_sot) * SOT_DRAW_WEIGHT

            base_draw_rate = self.draw_base_by_div.get(row.div, DEFAULT_DRAW_BASE)
            base_draw_rate = max(MIN_DRAW_FLOOR, min(MAX_DRAW_CEILING, base_draw_rate))
            draw_raw = base_draw_rate - abs(elo_diff) / DRAW_SLOPE + draw_bonus
            floor = max(MIN_DRAW_FLOOR, base_draw_rate - DRAW_BAND)
            ceiling = min(MAX_DRAW_CEILING, base_draw_rate + DRAW_BAND)
            p_draw_base = max(floor, min(ceiling, draw_raw))

            draw_span = max(0.08, min(0.12, DRAW_BAND))
            elo_gap = abs(elo_diff)
            xg_gap = delta_xg if delta_xg is not None else 0.5
            sot_gap = delta_sot if delta_sot is not None else 2.0
            fav_margin = abs(p_home_adj - 0.5)

            draw_adjust = base_draw_rate
            draw_adjust -= 0.18 * min(1.0, (elo_gap / 90.0) ** 1.1)
            draw_adjust += 0.14 * (1.0 - min(1.0, xg_gap / 0.6))
            draw_adjust += 0.08 * (1.0 - min(1.0, sot_gap / 4.0))
            draw_adjust += 0.10 * (1.0 - min(1.0, fav_margin / 0.25))
            draw_adjust -= 0.12 * min(1.0, fav_margin / 0.28) ** 1.1
            draw_adjust -= 0.06 * min(1.0, abs(elo_diff) / 35.0)

            clamp_low = max(MIN_DRAW_FLOOR, base_draw_rate - draw_span)
            clamp_high = min(MAX_DRAW_CEILING, base_draw_rate + draw_span)
            p_draw = min(max(draw_adjust, clamp_low), clamp_high)

            parity_elo = max(0.0, 1.0 - min(1.0, elo_gap / 45.0))
            parity_xg = max(0.0, 1.0 - min(1.0, xg_gap / 0.55))
            parity_sot = max(0.0, 1.0 - min(1.0, sot_gap / 3.5))
            parity_prob = max(0.0, 1.0 - min(1.0, fav_margin / 0.15))
            parity_score = (
                0.45 * parity_elo
                + 0.30 * parity_xg
                + 0.15 * parity_sot
                + 0.10 * parity_prob
            )
            parity_score *= max(0.0, 1.0 - 0.6 * min(1.0, fav_margin / 0.25))
            draw_target = min(clamp_high, base_draw_rate + 0.33 * parity_score)
            if draw_target > p_draw:
                p_draw = draw_target
            balance_close = parity_score >= 0.18

            calibrated_draw = self._calibrate_draw_probability(
                p_draw,
                base_draw_rate,
                elo_diff,
                home_strength,
                away_strength,
                home_xg_val,
                away_xg_val,
                home_sot_eval,
                away_sot_eval,
            )
            if calibrated_draw is not None and fav_margin <= 0.2 and (balance_close or abs(elo_diff) <= 50.0 or 0.08 <= p_draw <= 0.45):
                blended_draw = DRAW_CALIBRATION_BLEND * p_draw + (1.0 - DRAW_CALIBRATION_BLEND) * calibrated_draw
                if blended_draw > p_draw:
                    p_draw = max(floor, min(ceiling, blended_draw))
            if abs(elo_diff) >= 6.0 and p_draw > p_draw_base + 0.05:
                p_draw = min(p_draw, p_draw_base + 0.05)
            scale = 1.0 - p_draw
            p_home = p_home_adj * scale
            p_away = p_away_adj * scale

            total = p_home + p_draw + p_away
            if total > 0:
                p_home /= total
                p_draw /= total
                p_away /= total

            elite_guard = (
                abs(elo_diff) < 20.0
                and (delta_xg if delta_xg is not None else 0.6) < 0.35
                and (delta_sot if delta_sot is not None else 4.0) < 0.35
                and strength_gap is not None
                and strength_gap < 0.6
            )
            if elite_guard and p_draw < 0.28:
                boost = 0.28 - p_draw
                fav_is_home = p_home >= p_away
                fav_prob = p_home if fav_is_home else p_away
                reducible = max(0.0, fav_prob - 0.15)
                transfer = min(boost, reducible)
                if transfer > 0:
                    p_draw += transfer
                    if fav_is_home:
                        p_home = max(0.0, p_home - transfer)
                    else:
                        p_away = max(0.0, p_away - transfer)
                    total = p_home + p_draw + p_away
                    if total > 0:
                        p_home /= total
                        p_draw /= total
                        p_away /= total

            market_probs = market_prob_map.get(int(row.fixt_id))
            market_edge = None
            if market_probs:
                p_home, p_draw, p_away = self._blend_with_market(
                    (p_home, p_draw, p_away),
                    market_probs,
                )
                fav_probs_tmp = np.asarray([p_home, p_draw, p_away])
                fav_idx_tmp = int(np.argmax(fav_probs_tmp))
                market_edge = float(fav_probs_tmp[fav_idx_tmp] - market_probs[fav_idx_tmp])
                if abs(market_edge) >= MARKET_EDGE_CAP and row.div not in TRUSTED_MARKET_DIVS:
                    p_home, p_draw, p_away = self._blend_with_market(
                        (p_home, p_draw, p_away),
                        market_probs,
                        model_weight=HIGH_EDGE_MODEL_WEIGHT,
                    )
                    fav_probs_tmp = np.asarray([p_home, p_draw, p_away])
                    fav_idx_tmp = int(np.argmax(fav_probs_tmp))
                    market_edge = float(fav_probs_tmp[fav_idx_tmp] - market_probs[fav_idx_tmp])

            market_used = bool(market_probs)
            before_override = (p_home, p_draw, p_away)
            if (
                abs(p_home - p_away) <= DRAW_OVERRIDE_GAP
                and p_draw >= DRAW_OVERRIDE_MIN_PROB
                and p_draw_base >= DRAW_OVERRIDE_MIN_RAW
            ):
                shift_home = min(DRAW_OVERRIDE_SHIFT, p_home)
                shift_away = min(DRAW_OVERRIDE_SHIFT, p_away)
                if shift_home > 0 and shift_away > 0:
                    p_home -= shift_home
                    p_away -= shift_away
                    p_draw += shift_home + shift_away
                    total = p_home + p_draw + p_away
                    if total > 0:
                        p_home /= total
                        p_draw /= total
                        p_away /= total
                    override_records.append(
                        {
                            "fixt_id": int(row.fixt_id),
                            "div": row.div,
                            "match_date": pd.to_datetime(row.match_date),
                            "home_team": row.home_team,
                            "away_team": row.away_team,
                            "market_used": market_used,
                            "prob_draw_raw": round(float(p_draw_base), 4),
                            "prob_home_before": round(before_override[0], 4),
                            "prob_draw_before": round(before_override[1], 4),
                            "prob_away_before": round(before_override[2], 4),
                            "prob_home_after": round(p_home, 4),
                            "prob_draw_after": round(p_draw, 4),
                            "prob_away_after": round(p_away, 4),
                        }
                    )
            max_side = max(p_home, p_away)
            fav_is_home = p_home >= p_away
            tight_match = max_side <= 0.55 and p_draw >= 0.28
            close_gap = (max_side - p_draw) <= 0.05
            if tight_match and close_gap and max_side > p_draw:
                transfer = min(max_side - p_draw + 0.015, max_side * 0.35)
                if transfer > 0:
                    p_draw += transfer
                    if fav_is_home:
                        p_home = max(0.0, p_home - transfer)
                    else:
                        p_away = max(0.0, p_away - transfer)
                    total = p_home + p_draw + p_away
                    if total > 0:
                        p_home /= total
                        p_draw /= total
                        p_away /= total

            fav_is_home_post = p_home >= p_away
            fav_prob_post = p_home if fav_is_home_post else p_away
            margin_post = abs(p_home - p_away)
            if fav_prob_post > 0.70 and not (margin_post > 0.25 and p_draw < 0.15):
                excess = fav_prob_post - 0.70
                if fav_is_home_post:
                    p_home = max(0.0, p_home - excess)
                else:
                    p_away = max(0.0, p_away - excess)
                p_draw += excess
                total = p_home + p_draw + p_away
                if total > 0:
                    p_home /= total
                    p_draw /= total
                    p_away /= total

            records.append(
                {
                    "fixt_id": row.fixt_id,
                    "div": row.div,
                    "match_date": pd.to_datetime(row.match_date),
                    "home_team": row.home_team,
                    "away_team": row.away_team,
                    "home_elo": round(home_elo, 2),
                    "away_elo": round(away_elo, 2),
                    "elo_edge": round(elo_diff, 2),
                    "home_xg": round(home_xg_val, 2),
                    "away_xg": round(away_xg_val, 2),
                    "home_sot": round(home_sot_eval, 2),
                    "away_sot": round(away_sot_eval, 2),
                    "prob_draw_raw": round(p_draw_base, 3),
                    "home_strength": round(home_strength, 3) if home_strength is not None else None,
                    "away_strength": round(away_strength, 3) if away_strength is not None else None,
                    "prob_home": round(p_home, 3),
                    "prob_draw": round(p_draw, 3),
                    "prob_away": round(p_away, 3),
                    "generated_at": generated_at,
                }
            )

        preview_df = pd.DataFrame.from_records(records)
        if override_records:
            override_df = pd.DataFrame.from_records(override_records)
            override_df.sort_values("match_date").to_csv(REPORTS_DIR / "draw_override_events.csv", index=False)
        self.con.register("previews_tmp", preview_df)
        self.con.execute("DELETE FROM fixture_previews")
        self.con.execute(
            "INSERT INTO fixture_previews (fixt_id, div, match_date, home_team, away_team, home_elo, away_elo, elo_edge, home_xg, away_xg, home_sot, away_sot, prob_draw_raw, prob_home, prob_draw, prob_away, generated_at, home_strength, away_strength) SELECT fixt_id, div, match_date, home_team, away_team, home_elo, away_elo, elo_edge, home_xg, away_xg, home_sot, away_sot, prob_draw_raw, prob_home, prob_draw, prob_away, generated_at, home_strength, away_strength FROM previews_tmp"
        )
        self.con.execute(
            "INSERT INTO fixture_previews_history (fixt_id, div, match_date, home_team, away_team, generated_at, home_elo, away_elo, elo_edge, home_xg, away_xg, home_sot, away_sot, prob_draw_raw, prob_home, prob_draw, prob_away, home_strength, away_strength) SELECT fixt_id, div, match_date, home_team, away_team, generated_at, home_elo, away_elo, elo_edge, home_xg, away_xg, home_sot, away_sot, prob_draw_raw, prob_home, prob_draw, prob_away, home_strength, away_strength FROM previews_tmp"
        )
        self.con.unregister("previews_tmp")

        REPORTS_DIR.mkdir(exist_ok=True)
        preview_df.sort_values("match_date").to_csv(REPORTS_DIR / "fixture_previews.csv", index=False)
        if self.league_accuracy_info:
            league_rows = [
                {"div": div, **info}
                for div, info in self.league_accuracy_info.items()
            ]
            league_df = pd.DataFrame.from_records(league_rows).sort_values("matches", ascending=False)
            league_df.to_csv(REPORTS_DIR / "league_accuracy.csv", index=False)
        return preview_df


    def close(self) -> None:
            self.con.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ratings pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    upd = sub.add_parser("update", help="update team ratings from matches")
    upd.add_argument("--db", type=Path, default=DB_PATH)

    prev = sub.add_parser("preview", help="generate fixture previews")
    prev.add_argument("--db", type=Path, default=DB_PATH)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    pipeline = RatingsPipeline(db_path=args.db)
    try:
        if args.command == "update":
            df = pipeline.update_team_ratings()
            print(f"updated teams: {len(df)}")
        elif args.command == "preview":
            df = pipeline.generate_previews()
            print(f"previews: {len(df)}")
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()






















