import json
import subprocess
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from pandas.api import types as ptypes
import streamlit as st
from sklearn.isotonic import IsotonicRegression

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "database" / "betting.duckdb"
DRAW_ALERT_MIN_PROB = 0.22
DRAW_ALERT_MAX_GAP = 0.08
DRAW_ALERT_XG_GAP = 0.4
DRAW_ALERT_SOT_GAP = 1.5
DRAW_ALERT_ELO_GAP = 35.0
TRUSTED_MARKET_DIVS = {'E0', 'D1', 'I1', 'SP1', 'F1', 'BR1', 'UCL', 'UEL'}
VISIBLE_DIVS = {'B1','BR1','E0','E1','E2','E3','EE1','UCL','UEL','FI1','F1','F2','D1','D2','G1','I1','I2','N1','NO1','P1','SC0','SP1','SP2','SW1','T1'}
MARKET_EDGE_WARN = 0.08
MARKET_EDGE_CAP = 0.15

CALIBRATION_WINDOW_DAYS = 60
CALIBRATION_MIN_SAMPLES = 40
LEAGUE_EDGE_MIN_SAMPLES = 25
GLOBAL_CALIBRATION_KEY = "__global__"
OUTCOME_PROB_COLUMNS = {'H': 'prob_home', 'D': 'prob_draw', 'A': 'prob_away'}



def _fit_isotonic_regression(probabilities: pd.Series, targets: pd.Series) -> IsotonicRegression | None:
    if probabilities is None or targets is None:
        return None
    numeric_probs = pd.to_numeric(probabilities, errors="coerce")
    numeric_targets = pd.to_numeric(targets, errors="coerce")
    mask = numeric_probs.notna() & numeric_targets.notna()
    if not mask.any():
        return None
    x_vals = numeric_probs[mask].astype(float).to_numpy()
    y_vals = numeric_targets[mask].astype(float).to_numpy()
    if len(x_vals) < CALIBRATION_MIN_SAMPLES or len(np.unique(y_vals)) < 2:
        return None
    if np.isclose(x_vals.max(), x_vals.min()):
        return None
    model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
    model.fit(x_vals, y_vals)
    return model


def build_isotonic_calibrators(df: pd.DataFrame) -> dict[str, dict[str, IsotonicRegression]]:
    calibrators: dict[str, dict[str, IsotonicRegression]] = {}
    if df.empty:
        return calibrators
    working = df.copy()
    if 'match_date' in working.columns:
        working['match_date'] = pd.to_datetime(working['match_date'])
        if working['match_date'].notna().any():
            cutoff = working['match_date'].max() - pd.Timedelta(days=CALIBRATION_WINDOW_DAYS)
            recent = working[working['match_date'] >= cutoff]
        else:
            recent = working
    else:
        recent = working
    if recent.empty:
        recent = working
    calibrators[GLOBAL_CALIBRATION_KEY] = {}
    for outcome, column in OUTCOME_PROB_COLUMNS.items():
        target_recent = (recent['result'] == outcome).astype(float)
        model = _fit_isotonic_regression(recent[column], target_recent)
        if model is None:
            target_full = (working['result'] == outcome).astype(float)
            model = _fit_isotonic_regression(working[column], target_full)
        if model is not None:
            calibrators[GLOBAL_CALIBRATION_KEY][column] = model
    for div, group in recent.groupby('div'):
        div_models: dict[str, IsotonicRegression] = {}
        source_full = working[working['div'] == div]
        for outcome, column in OUTCOME_PROB_COLUMNS.items():
            target_group = (group['result'] == outcome).astype(float)
            model = _fit_isotonic_regression(group[column], target_group)
            if model is None and not source_full.empty:
                target_full = (source_full['result'] == outcome).astype(float)
                model = _fit_isotonic_regression(source_full[column], target_full)
            if model is not None:
                div_models[column] = model
        if div_models:
            calibrators[div] = div_models
    if not calibrators.get(GLOBAL_CALIBRATION_KEY):
        calibrators.pop(GLOBAL_CALIBRATION_KEY, None)
    return calibrators



def apply_isotonic_calibration(df: pd.DataFrame, calibrators: dict[str, dict[str, IsotonicRegression]]) -> pd.DataFrame:
    result = df.copy()
    if not calibrators:
        for column in OUTCOME_PROB_COLUMNS.values():
            result[f'{column}_cal'] = result[column]
        return result

    def _calibrate_value(row: pd.Series, column: str) -> float | None:
        raw_value = row.get(column)
        if pd.isna(raw_value):
            return raw_value
        div_key = row.get('div')
        calibrator = calibrators.get(div_key, {}).get(column)
        if calibrator is None:
            calibrator = calibrators.get(GLOBAL_CALIBRATION_KEY, {}).get(column)
        if calibrator is None:
            return raw_value
        try:
            calibrated = calibrator.transform([float(raw_value)])[0]
        except ValueError:
            return raw_value
        return float(np.clip(calibrated, 0.0, 1.0))

    for column in OUTCOME_PROB_COLUMNS.values():
        result[f'{column}_cal'] = result.apply(lambda row, col=column: _calibrate_value(row, col), axis=1)

    cal_cols = [f'{column}_cal' for column in OUTCOME_PROB_COLUMNS.values() if f'{column}_cal' in result.columns]
    if cal_cols:
        row_sums = result[cal_cols].sum(axis=1)
        mask = row_sums.replace([np.inf, -np.inf], np.nan).notna() & (row_sums > 0)
        if mask.any():
            result.loc[mask, cal_cols] = result.loc[mask, cal_cols].div(row_sums[mask], axis=0)

    return result

def compute_league_edge_lookup(df: pd.DataFrame, calibrators: dict[str, dict[str, IsotonicRegression]]) -> tuple[dict[str, float], dict[str, int]]:
    if df.empty:
        return {}, {}
    working = apply_isotonic_calibration(df, calibrators)
    working = working.copy()

    def _implied_probs(row: pd.Series) -> pd.Series:
        odds = [row.get('odds_home'), row.get('odds_draw'), row.get('odds_away')]
        inv = []
        for value in odds:
            if value is None or pd.isna(value) or value <= 0:
                inv.append(np.nan)
            else:
                inv.append(1.0 / value)
        inv_sum = np.nansum(inv)
        if not np.isfinite(inv_sum) or inv_sum <= 0:
            return pd.Series([np.nan, np.nan, np.nan])
        return pd.Series([(v / inv_sum) if np.isfinite(v) else np.nan for v in inv])

    working[['imp_home', 'imp_draw', 'imp_away']] = working.apply(_implied_probs, axis=1)

    prob_lookup: dict[str, str] = {}
    for outcome, base_col in OUTCOME_PROB_COLUMNS.items():
        cal_col = f'{base_col}_cal'
        prob_lookup[outcome] = cal_col if cal_col in working.columns else base_col

    implied_lookup = {'H': 'imp_home', 'D': 'imp_draw', 'A': 'imp_away'}

    def _edge_actual(row: pd.Series) -> float | None:
        key = row.get('result')
        if key not in prob_lookup:
            return None
        prob_val = row.get(prob_lookup[key])
        imp_val = row.get(implied_lookup[key])
        if pd.isna(prob_val) or pd.isna(imp_val):
            return None
        return float(prob_val - imp_val)

    working['edge_actual'] = working.apply(_edge_actual, axis=1)

    summary = (
        working.dropna(subset=['edge_actual'])
        .groupby('div')['edge_actual']
        .agg(['mean', 'count'])
        .reset_index()
    )

    edge_lookup = {row['div']: float(row['mean']) for _, row in summary.iterrows()}
    count_lookup = {row['div']: int(row['count']) for _, row in summary.iterrows()}
    return edge_lookup, count_lookup


    def _calibrate_value(row: pd.Series, column: str) -> float | None:
        raw_value = row.get(column)
        if pd.isna(raw_value):
            return raw_value
        div_key = row.get('div')
        calibrator = calibrators.get(div_key, {}).get(column)
        if calibrator is None:
            calibrator = calibrators.get(GLOBAL_CALIBRATION_KEY, {}).get(column)
        if calibrator is None:
            return raw_value
        try:
            calibrated = calibrator.transform([float(raw_value)])[0]
        except ValueError:
            return raw_value
        return float(np.clip(calibrated, 0.0, 1.0))

    for column in OUTCOME_PROB_COLUMNS.values():
        result[f'{column}_cal'] = result.apply(lambda row, col=column: _calibrate_value(row, col), axis=1)

    cal_cols = [f'{column}_cal' for column in OUTCOME_PROB_COLUMNS.values() if f'{column}_cal' in result.columns]
    if cal_cols:
        row_sums = result[cal_cols].sum(axis=1)
        mask = row_sums.replace([np.inf, -np.inf], np.nan).notna() & (row_sums > 0)
        if mask.any():
            result.loc[mask, cal_cols] = result.loc[mask, cal_cols].div(row_sums[mask], axis=0)

    return result

def _connect(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(DB_PATH), read_only=read_only)


@st.cache_data(show_spinner=False)
def load_competitions() -> pd.DataFrame:
    con = _connect()
    try:
        return con.execute("SELECT * FROM competitions ORDER BY div").fetchdf()
    finally:
        con.close()


@st.cache_data(show_spinner=False)
def load_fixture_previews() -> pd.DataFrame:
    con = _connect()
    try:
        return con.execute("SELECT * FROM fixture_previews ORDER BY match_date").fetchdf()
    finally:
        con.close()


@st.cache_data(show_spinner=False)
def load_team_ratings() -> pd.DataFrame:
    con = _connect()
    try:
        return con.execute(
            "SELECT div, team, season, elo, xg_for, xg_against, matches_played, updated_at, rolling_form FROM team_ratings"
        ).fetchdf()
    finally:
        con.close()


@st.cache_data(show_spinner=False)
def load_post_match_data() -> pd.DataFrame:
    con = _connect()
    try:
        query = """
            WITH latest_previews AS (
                SELECT *
                FROM (
                    SELECT fp.*, ROW_NUMBER() OVER (PARTITION BY fixt_id ORDER BY generated_at DESC) AS rn
                    FROM fixture_previews_history fp
                )
                WHERE rn = 1
            ),
            latest_odds AS (
                SELECT *
                FROM (
                    SELECT oh.*, ROW_NUMBER() OVER (PARTITION BY fixt_id ORDER BY fetched_at DESC) AS rn
                    FROM odds_history oh
                )
                WHERE rn = 1
            )
            SELECT r.fixt_id,
                   COALESCE(p.div, r.div) AS div,
                   COALESCE(p.match_date, r.match_date) AS match_date,
                   COALESCE(p.home_team, r.home_team) AS home_team,
                   COALESCE(p.away_team, r.away_team) AS away_team,
                   r.home_goals,
                   r.away_goals,
                   r.result,
                   r.status,
                   r.fetched_at AS result_fetched_at,
                   p.generated_at AS preview_generated_at,
                   p.home_elo,
                   p.away_elo,
                   p.elo_edge,
                   p.home_strength,
                   p.away_strength,
                   p.prob_home,
                   p.prob_draw,
                   p.prob_away,
                   p.prob_draw_raw,
                   o.bookmaker_name,
                   o.odds_home,
                   o.odds_draw,
                   o.odds_away,
                   o.fetched_at AS odds_fetched_at
            FROM fixture_results r
            LEFT JOIN latest_previews p ON r.fixt_id = p.fixt_id
            LEFT JOIN latest_odds o ON r.fixt_id = o.fixt_id
            WHERE r.status IN ('FT', 'AET', 'PEN')
        """
        return con.execute(query).fetchdf()
    finally:
        con.close()




@st.cache_data(show_spinner=False)
def load_latest_odds() -> pd.DataFrame:
    con = _connect()
    try:
        query = """
            WITH ranked AS (
                SELECT fixt_id,
                       bookmaker_name,
                       odds_home,
                       odds_draw,
                       odds_away,
                       ROW_NUMBER() OVER (PARTITION BY fixt_id ORDER BY fetched_at DESC) AS rn
                FROM odds_history
            )
            SELECT fixt_id,
                   bookmaker_name,
                   odds_home,
                   odds_draw,
                   odds_away
            FROM ranked
            WHERE rn = 1
        """
        return con.execute(query).fetchdf()
    finally:
        con.close()

st.set_page_config(page_title="Fixture Insights", layout="wide")
st.title("Fixture Insights")

if st.sidebar.button("Refresh data"):
    load_competitions.clear()
    load_fixture_previews.clear()
    load_team_ratings.clear()
    load_post_match_data.clear()
    load_latest_odds.clear()
    st.sidebar.success("Cache cleared; data will reload below.")

if st.sidebar.button("Rebuild previews"):
    with st.spinner("Rebuilding fixture previews..."):
        result = subprocess.run([sys.executable, "-m", "src.ratings.pipeline", "preview"], cwd=str(BASE_DIR))
    if result.returncode == 0:
        load_competitions.clear()
        load_fixture_previews.clear()
        load_team_ratings.clear()
        load_post_match_data.clear()
        load_latest_odds.clear()
        st.sidebar.success("Previews regenerated.")
    else:
        st.sidebar.error(f"Preview pipeline failed (exit code {result.returncode}).")

if st.sidebar.button("Fetch missing odds"):
    with st.spinner("Fetching odds for European fixtures..."):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.io.fetch_missing_odds",
                "--lookahead",
                "7",
                "--lookback",
                "2",
                "--max-fixtures",
                "400",
            ],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
        )
    if result.returncode == 0:
        load_post_match_data.clear()
        load_latest_odds.clear()
        st.sidebar.success("Odds updated.")
        stdout = (result.stdout or "").strip()
        if stdout:
            for line in stdout.splitlines():
                st.sidebar.caption(line)
    else:
        st.sidebar.error(f"Odds fetch failed (exit code {result.returncode}).")
        stderr = (result.stderr or "").strip()
        if stderr:
            st.sidebar.caption(stderr.splitlines()[-1])

if st.sidebar.button("Refresh results"):
    with st.spinner("Fetching completed fixtures..."):
        result = subprocess.run(
            [sys.executable, "-m", "src.io.fetch_results"],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
        )
    if result.returncode == 0:
        load_fixture_previews.clear()
        load_post_match_data.clear()
        load_latest_odds.clear()
        st.sidebar.success("Results updated.")
        summary = (result.stdout or "").strip().splitlines()
        if summary:
            st.sidebar.caption(summary[-1])
    else:
        st.sidebar.error(f"Results fetch failed (exit code {result.returncode}).")
        err_summary = (result.stderr or "").strip().splitlines()
        if err_summary:
            st.sidebar.caption(err_summary[-1])

competitions = load_competitions()
if not competitions.empty:
    competitions = competitions[competitions['div'].isin(VISIBLE_DIVS)].copy()
fixture_df = load_fixture_previews()
if not fixture_df.empty:
    fixture_df = fixture_df[fixture_df['div'].isin(VISIBLE_DIVS)].copy()
ratings_df = load_team_ratings()
if not ratings_df.empty:
    ratings_df = ratings_df[ratings_df['div'].isin(VISIBLE_DIVS)].copy()
odds_df = load_latest_odds()
if not odds_df.empty and 'div' in odds_df.columns:
    odds_df = odds_df[odds_df['div'].isin(VISIBLE_DIVS)].copy()
post_match_full_df = load_post_match_data()
if post_match_full_df is None:
    post_match_full_df = pd.DataFrame()
else:
    post_match_full_df = post_match_full_df[post_match_full_df['div'].isin(VISIBLE_DIVS)].copy()
calibrators_shared = build_isotonic_calibrators(post_match_full_df)
league_edge_lookup_global, league_edge_counts_global = compute_league_edge_lookup(post_match_full_df, calibrators_shared)

if competitions.empty:
    st.warning("Competition metadata not loaded. Populate competitions table first.")

comp_lookup = dict(zip(competitions["div"], competitions["name"])) if not competitions.empty else {}

previews_tab, ratings_tab, analysis_tab = st.tabs(["Fixture Previews", "Team Ratings", "Post-Match Analysis"])

with previews_tab:
    if fixture_df.empty:
        st.info("No fixtures loaded. Import fixtures and run previews.")
    else:
        previews_df = apply_isotonic_calibration(fixture_df.copy(), calibrators_shared)
        previews_df["match_date"] = pd.to_datetime(previews_df["match_date"])
        previews_df["kickoff"] = previews_df["match_date"].dt.strftime("%Y-%m-%d %H:%M")
        previews_df["competition"] = previews_df["div"].map(comp_lookup).fillna(previews_df["div"])

        has_calibration_previews = any(calibrators_shared.values())
        use_calibrated_previews = st.toggle(
            "Use calibrated probabilities in previews",
            value=has_calibration_previews,
            help="Apply isotonic regression (per competition when samples allow, otherwise global) before ranking favourites and edges.",
        )
        st.caption(
            f"Calibration derives from the last {CALIBRATION_WINDOW_DAYS} days (min {CALIBRATION_MIN_SAMPLES} matches per competition when available, otherwise global history)."
        )
        if use_calibrated_previews and not has_calibration_previews:
            st.info("Not enough completed fixtures to fit competition calibrators yet; using original model probabilities.")

        leagues = sorted(previews_df["competition"].unique())
        selected = st.multiselect("Competitions", leagues, default=leagues, key="previews_competitions")
        lookahead_hours = st.slider("Lookahead (hours)", min_value=6, max_value=240, value=72, step=6)
        now = pd.Timestamp.utcnow().tz_localize(None)
        horizon = now + pd.Timedelta(hours=lookahead_hours)

        filtered = previews_df[previews_df["competition"].isin(selected)].copy()
        filtered = filtered[(filtered["match_date"] >= now) & (filtered["match_date"] <= horizon)]
        if filtered.empty:
            st.info("No fixtures in the selected window.")
        else:
            active_prob_cols: dict[str, str] = {}
            for outcome, base_col in OUTCOME_PROB_COLUMNS.items():
                cal_col = f"{base_col}_cal"
                if use_calibrated_previews and cal_col in filtered.columns:
                    active_prob_cols[outcome] = cal_col
                else:
                    active_prob_cols[outcome] = base_col

            filtered["prob_home_model"] = filtered[active_prob_cols["H"]]
            filtered["prob_draw_model"] = filtered[active_prob_cols["D"]]
            filtered["prob_away_model"] = filtered[active_prob_cols["A"]]

            prob_matrix = filtered[["prob_home_model", "prob_draw_model", "prob_away_model"]].fillna(0.0).to_numpy()
            fav_idx = prob_matrix.argmax(axis=1)
            fav_prob = prob_matrix.max(axis=1)
            second_prob = np.partition(prob_matrix, -2, axis=1)[:, -2]
            fav_gap = fav_prob - second_prob

            code_lookup = np.array(["H", "D", "A"])
            label_lookup = {"H": "Home", "D": "Draw", "A": "Away"}
            filtered["fav_code"] = pd.Series(code_lookup[fav_idx], index=filtered.index)
            filtered["fav_label"] = filtered["fav_code"].map(label_lookup)
            filtered["fav_prob"] = fav_prob
            filtered["fav_gap"] = fav_gap

            draw_lean_mask = (
                (filtered.get("prob_draw_raw", filtered["prob_draw_model"]).fillna(0.0) >= DRAW_ALERT_MIN_PROB)
                & (filtered["prob_draw_model"] >= DRAW_ALERT_MIN_PROB)
                & (fav_gap <= DRAW_ALERT_MAX_GAP)
            )
            filtered["draw_alert"] = np.where(draw_lean_mask, "Draw lean", "")

            if not odds_df.empty:
                latest_odds = odds_df.copy()
                filtered = filtered.merge(latest_odds, on="fixt_id", how="left")
            else:
                for col in ["bookmaker_name", "odds_home", "odds_draw", "odds_away"]:
                    if col not in filtered.columns:
                        filtered[col] = np.nan

            for side in ("home", "draw", "away"):
                odds_col = f"odds_{side}"
                filtered[odds_col] = pd.to_numeric(filtered.get(odds_col), errors="coerce")

            for side in ("home", "draw", "away"):
                odds_col = f"odds_{side}"
                imp_col = f"imp_{side}"
                filtered[imp_col] = np.where(filtered[odds_col] > 0, 1.0 / filtered[odds_col], np.nan)

            filtered["imp_total"] = filtered[["imp_home", "imp_draw", "imp_away"]].sum(axis=1)
            valid_implied = filtered["imp_total"] > 0
            for side in ("home", "draw", "away"):
                imp_col = f"imp_{side}"
                filtered.loc[valid_implied, imp_col] = filtered.loc[valid_implied, imp_col] / filtered.loc[valid_implied, "imp_total"]
                filtered.loc[~valid_implied, imp_col] = np.nan
            filtered.drop(columns=["imp_total"], inplace=True)

            implied_lookup_labels = {"Home": "imp_home", "Draw": "imp_draw", "Away": "imp_away"}
            implied_lookup_codes = {"H": "imp_home", "D": "imp_draw", "A": "imp_away"}

            def _fav_implied(row: pd.Series) -> float | None:
                label = row.get("fav_label")
                if not label:
                    return np.nan
                imp_col = implied_lookup_labels.get(label)
                if not imp_col:
                    return np.nan
                val = row.get(imp_col)
                return float(val) if pd.notna(val) else np.nan

            filtered["fav_implied"] = filtered.apply(_fav_implied, axis=1)
            filtered["edge_pred"] = filtered["fav_prob"] - filtered["fav_implied"]
            filtered["market_flag"] = np.where(
                (filtered["fav_implied"].notna()) & (np.abs(filtered["edge_pred"]) >= MARKET_EDGE_WARN),
                "Market disagree",
                ""
            )

            hide_high_edges = st.checkbox("Hide >10pp market gaps (non-trusted leagues)", value=True)
            if hide_high_edges:
                high_gap_mask = (
                    (filtered["fav_implied"].notna())
                    & (np.abs(filtered["edge_pred"]) >= MARKET_EDGE_CAP)
                    & (~filtered["div"].isin(TRUSTED_MARKET_DIVS))
                )
                filtered = filtered[~high_gap_mask]

            shortlist_enabled = st.checkbox("Apply edge shortlist filters", value=False)
            if shortlist_enabled:
                edge_threshold = st.slider("Minimum edge (%)", min_value=2.0, max_value=15.0, value=6.0, step=0.5)
                confidence_threshold = st.slider("Minimum favourite probability (%)", min_value=50.0, max_value=75.0, value=55.0, step=1.0)
                exclude_draw = st.checkbox("Exclude draw-lean fixtures", value=True)
                shortlist_mask = (
                    filtered["edge_pred"].notna()
                    & (filtered["edge_pred"] >= edge_threshold / 100.0)
                    & (filtered["fav_prob"] >= confidence_threshold / 100.0)
                )
                filtered = filtered[shortlist_mask]
                if exclude_draw:
                    filtered = filtered[filtered["draw_alert"] == ""]

            if filtered.empty:
                st.info("No fixtures match the current filters.")
            else:
                if {"odds_home", "odds_draw", "odds_away"}.issubset(filtered.columns):
                    filtered[["odds_home", "odds_draw", "odds_away"]] = filtered[["odds_home", "odds_draw", "odds_away"]].round(2)

                filtered["fav_prob_pct"] = filtered["fav_prob"].apply(lambda v: round(v * 100, 2) if pd.notna(v) else np.nan)
                filtered["fav_implied_pct"] = filtered["fav_implied"].apply(lambda v: round(v * 100, 2) if pd.notna(v) else np.nan)
                filtered["edge_pred_pct"] = filtered["edge_pred"].apply(lambda v: round(v * 100, 2) if pd.notna(v) else np.nan)

                raw_lookup = {"H": OUTCOME_PROB_COLUMNS["H"], "D": OUTCOME_PROB_COLUMNS["D"], "A": OUTCOME_PROB_COLUMNS["A"]}

                def _fav_raw_prob(row: pd.Series) -> float | None:
                    code = row.get("fav_code")
                    if code not in raw_lookup:
                        return None
                    val = row.get(raw_lookup[code])
                    return float(val) if pd.notna(val) else None

                filtered["fav_prob_raw"] = filtered.apply(_fav_raw_prob, axis=1)
                filtered["fav_prob_raw_pct"] = filtered["fav_prob_raw"].apply(lambda v: round(v * 100, 2) if v is not None else np.nan)
                filtered["fav_shift_pct"] = filtered.apply(
                    lambda row: round((row.get("fav_prob") - row.get("fav_prob_raw")) * 100, 2)
                    if use_calibrated_previews and row.get("fav_prob") is not None and row.get("fav_prob_raw") is not None
                    else np.nan,
                    axis=1,
                )

                filtered["home_pct"] = filtered["prob_home_model"].apply(lambda v: round(v * 100, 2) if pd.notna(v) else np.nan)
                filtered["draw_pct"] = filtered["prob_draw_model"].apply(lambda v: round(v * 100, 2) if pd.notna(v) else np.nan)
                filtered["away_pct"] = filtered["prob_away_model"].apply(lambda v: round(v * 100, 2) if pd.notna(v) else np.nan)

                filtered["league_edge_pct"] = filtered["div"].map(lambda d: league_edge_lookup_global.get(d) * 100 if d in league_edge_lookup_global else np.nan)
                filtered["league_edge_sample"] = filtered["div"].map(lambda d: league_edge_counts_global.get(d, 0))

                model_prob_lookup = {"H": "prob_home_model", "D": "prob_draw_model", "A": "prob_away_model"}
                outcome_name_lookup = {"H": "Home", "D": "Draw", "A": "Away"}

                def _preview_alerts(row: pd.Series) -> str:
                    notes: list[str] = []
                    for outcome_code, label in outcome_name_lookup.items():
                        prob_val = row.get(model_prob_lookup[outcome_code])
                        imp_val = row.get(implied_lookup_codes[outcome_code])
                        if pd.isna(prob_val) or pd.isna(imp_val):
                            continue
                        diff = float(prob_val - imp_val)
                        if abs(diff) >= MARKET_EDGE_WARN:
                            direction = "above" if diff > 0 else "below"
                            notes.append(f"{label} prob {abs(diff) * 100:.2f}% {direction} market")
                    hist_edge = league_edge_lookup_global.get(row.get("div"))
                    if hist_edge is not None and hist_edge < 0:
                        pred_edge = row.get("edge_pred")
                        if pd.notna(pred_edge) and pred_edge > 0:
                            sample = league_edge_counts_global.get(row.get("div"), 0)
                            msg = f"League avg edge {hist_edge * 100:.2f}%"
                            if sample:
                                msg += f" over {sample} matches"
                            notes.append(msg)
                    return "; ".join(notes)

                filtered["preview_alerts"] = filtered.apply(_preview_alerts, axis=1)

                display_cols = [
                    "competition",
                    "kickoff",
                    "home_team",
                    "away_team",
                    "home_elo",
                    "away_elo",
                    "elo_edge",
                    "home_xg",
                    "away_xg",
                    "home_sot",
                    "away_sot",
                    "home_strength",
                    "away_strength",
                    "home_pct",
                    "draw_pct",
                    "away_pct",
                    "fav_label",
                    "fav_prob_pct",
                    "fav_prob_raw_pct",
                    "fav_shift_pct",
                    "fav_implied_pct",
                    "edge_pred_pct",
                    "league_edge_pct",
                    "league_edge_sample",
                    "odds_home",
                    "odds_draw",
                    "odds_away",
                    "bookmaker_name",
                    "draw_alert",
                    "market_flag",
                    "preview_alerts",
                ]

                if not use_calibrated_previews:
                    for column in ["fav_prob_raw_pct", "fav_shift_pct"]:
                        if column in display_cols:
                            display_cols.remove(column)

                missing = [col for col in display_cols if col not in filtered.columns]
                for col in missing:
                    filtered[col] = np.nan

                display = filtered[display_cols].copy()
                rename_map = {
                    "competition": "Competition",
                    "kickoff": "Kick-off",
                    "home_team": "Home",
                    "away_team": "Away",
                    "home_elo": "Home Elo",
                    "away_elo": "Away Elo",
                    "elo_edge": "Elo Edge",
                    "home_xg": "Home xG",
                    "away_xg": "Away xG",
                    "home_sot": "Home SoT",
                    "away_sot": "Away SoT",
                    "home_strength": "Home Str",
                    "away_strength": "Away Str",
                    "home_pct": "Home %",
                    "draw_pct": "Draw %",
                    "away_pct": "Away %",
                    "fav_label": "Fav",
                    "fav_prob_pct": "Fav %",
                    "fav_prob_raw_pct": "Fav Raw %",
                    "fav_shift_pct": "Fav Shift %",
                    "fav_implied_pct": "Fav Implied %",
                    "edge_pred_pct": "Edge %",
                    "league_edge_pct": "League Edge %",
                    "league_edge_sample": "League Edge N",
                    "odds_home": "Odds H",
                    "odds_draw": "Odds D",
                    "odds_away": "Odds A",
                    "bookmaker_name": "Bookmaker",
                    "draw_alert": "Draw Lean",
                    "market_flag": "Market Alert",
                    "preview_alerts": "Alerts",
                }
                display.rename(columns={k: v for k, v in rename_map.items() if k in display.columns}, inplace=True)

                def _alert_style(row: pd.Series) -> list[str]:
                    styles = ["" for _ in row]
                    alerts = row.get("Alerts")
                    if isinstance(alerts, str) and alerts:
                        try:
                            idx = list(row.index).index("Alerts")
                        except ValueError:
                            idx = None
                        if idx is not None:
                            styles[idx] = "background-color: rgba(251, 191, 36, 0.25)"
                    return styles

                styled = display.style.apply(_alert_style, axis=1)
                formatters = {col: '{:.2f}'.format for col in display.columns if ptypes.is_float_dtype(display[col]) and col not in {"League Edge N"}}
                if formatters:
                    styled = styled.format(formatters)
                st.dataframe(styled, width='stretch')

with ratings_tab:
    if ratings_df.empty:
        st.info("Run the ratings update to populate team data.")
    else:
        ratings_df = ratings_df.copy()
        ratings_df["competition"] = ratings_df["div"].map(comp_lookup).fillna(ratings_df["div"])
        leagues = sorted(ratings_df["competition"].unique())
        selected_league = st.selectbox("Competition", leagues, index=0 if leagues else None)
        comp_filtered = ratings_df[ratings_df["competition"] == selected_league] if selected_league else ratings_df
        comp_filtered = comp_filtered.sort_values("elo", ascending=False).copy()
        comp_filtered["updated_at"] = pd.to_datetime(comp_filtered["updated_at"]).dt.strftime("%Y-%m-%d %H:%M")
        comp_filtered["xg_balance"] = comp_filtered["xg_for"] - comp_filtered["xg_against"]
        display = comp_filtered[[
            "team",
            "season",
            "elo",
            "matches_played",
            "xg_for",
            "xg_against",
            "xg_balance",
            "updated_at",
        ]].rename(
            columns={
                "team": "Team",
                "season": "Season",
                "elo": "Elo",
                "matches_played": "Matches",
                "xg_for": "xG For",
                "xg_against": "xG Against",
                "xg_balance": "xG Balance",
                "updated_at": "Last Match",
            }
        )
        st.dataframe(display, width='stretch')

        if comp_filtered.empty:
            st.caption("No teams available for this competition yet.")
        else:
            team_options = comp_filtered["team"].tolist()
            selected_team = st.selectbox("Team details", team_options)
            row = comp_filtered[comp_filtered["team"] == selected_team].iloc[0]
            col1, col2, col3 = st.columns(3)
            col1.metric("Elo", f"{row['elo']:.2f}")
            col2.metric("xG For", f"{row['xg_for']:.2f}")
            col3.metric("xG Against", f"{row['xg_against']:.2f}")
            form = json.loads(row["rolling_form"]) if isinstance(row["rolling_form"], str) else []
            st.subheader("Recent Form")
            if form:
                form_df = pd.DataFrame(form)
                st.dataframe(form_df, width='stretch')
            else:
                st.caption("No recent matches recorded.")


with analysis_tab:
    post_df = post_match_full_df.copy() if post_match_full_df is not None else pd.DataFrame()
    if post_df.empty:
        st.info("No completed fixtures with previews recorded yet.")
    else:
        post_df = post_df.copy()
        post_df["match_date"] = pd.to_datetime(post_df["match_date"])
        post_df["kickoff"] = post_df["match_date"].dt.strftime("%Y-%m-%d %H:%M")
        post_df["competition"] = post_df["div"].map(comp_lookup).fillna(post_df["div"])
        post_df["scoreline"] = post_df.apply(
            lambda row: f"{int(row['home_goals'])}-{int(row['away_goals'])}" if pd.notna(row["home_goals"]) and pd.notna(row["away_goals"]) else "",
            axis=1,
        )

        calibrators = calibrators_shared
        post_df = apply_isotonic_calibration(post_df, calibrators)
        has_calibration = any(calibrators.values())

        use_calibrated = st.toggle(
            "Use calibrated probabilities",
            value=True,
            help="Apply isotonic regression (per competition when sample sizes allow, otherwise global) so probabilities better match observed frequencies.",
        )
        st.caption(
            f"Calibration uses the last {CALIBRATION_WINDOW_DAYS} days (min {CALIBRATION_MIN_SAMPLES} matches per competition when available, otherwise global history)."
        )
        if use_calibrated and not has_calibration:
            st.info("Not enough completed fixtures to fit competition-specific calibrators yet; using original model probabilities.")

        base_prob_cols = {"H": "prob_home", "D": "prob_draw", "A": "prob_away"}
        active_prob_cols = {key: (f"{value}_cal" if use_calibrated else value) for key, value in base_prob_cols.items()}
        prob_col_list = [active_prob_cols["H"], active_prob_cols["D"], active_prob_cols["A"]]

        outcome_map = {0: "H", 1: "D", 2: "A"}

        def _pick_prediction(row: pd.Series) -> pd.Series:
            probs = [row.get(col) for col in prob_col_list]
            if any(pd.isna(p) for p in probs):
                return pd.Series([None, None])
            idx = int(np.argmax(probs))
            return pd.Series([outcome_map[idx], probs[idx]])

        post_df[["pred_choice", "pred_confidence"]] = post_df.apply(_pick_prediction, axis=1)

        def _actual_prob(row: pd.Series) -> float | None:
            key = row.get("result")
            col = active_prob_cols.get(key or "")
            if not col or pd.isna(row.get(col)):
                return None
            return float(row.get(col))

        post_df["actual_prob"] = post_df.apply(_actual_prob, axis=1)

        def _implied_probs(row: pd.Series) -> pd.Series:
            odds = [row.get("odds_home"), row.get("odds_draw"), row.get("odds_away")]
            inv = []
            for value in odds:
                if value is None or pd.isna(value) or value <= 0:
                    inv.append(np.nan)
                else:
                    inv.append(1.0 / value)
            inv_sum = np.nansum(inv)
            if not np.isfinite(inv_sum) or inv_sum <= 0:
                return pd.Series([np.nan, np.nan, np.nan])
            return pd.Series([(v / inv_sum) if np.isfinite(v) else np.nan for v in inv])

        post_df[["imp_home", "imp_draw", "imp_away"]] = post_df.apply(_implied_probs, axis=1)

        def _brier(row: pd.Series) -> float:
            result = row.get("result")
            if result not in {"H", "D", "A"}:
                return float("nan")
            probs = [row.get(active_prob_cols["H"]), row.get(active_prob_cols["D"]), row.get(active_prob_cols["A"])]
            if any(pd.isna(p) for p in probs):
                return float("nan")
            actuals = [1.0 if result == outcome else 0.0 for outcome in ["H", "D", "A"]]
            return float(np.sum((np.array(probs) - np.array(actuals)) ** 2))

        post_df["brier"] = post_df.apply(_brier, axis=1)

        def _log_loss(row: pd.Series) -> float:
            prob = row.get("actual_prob")
            if prob is None or pd.isna(prob):
                return float("nan")
            clipped = float(np.clip(prob, 1e-9, 1 - 1e-9))
            return float(-np.log(clipped))

        post_df["log_loss"] = post_df.apply(_log_loss, axis=1)

        implied_lookup = {"H": "imp_home", "D": "imp_draw", "A": "imp_away"}

        def _edge(row: pd.Series, actual: bool = True) -> float | None:
            key = row.get("result") if actual else row.get("pred_choice")
            prob_col = active_prob_cols.get(key or "")
            imp_col = implied_lookup.get(key or "")
            if not prob_col or not imp_col:
                return None
            prob_val = row.get(prob_col)
            imp_val = row.get(imp_col)
            if pd.isna(prob_val) or pd.isna(imp_val):
                return None
            return float(prob_val - imp_val)

        post_df["edge_actual"] = post_df.apply(_edge, axis=1)
        post_df["edge_pred"] = post_df.apply(lambda row: _edge(row, actual=False), axis=1)

        league_edge_df = (
            post_df.dropna(subset=["edge_actual"])
            .groupby("div")
            .agg(edge_mean=("edge_actual", "mean"), sample_size=("edge_actual", "count"))
            .reset_index()
        )
        historical_edge_lookup = {}
        historical_edge_counts = {}
        for _, row in league_edge_df.iterrows():
            if row["sample_size"] >= LEAGUE_EDGE_MIN_SAMPLES:
                historical_edge_lookup[row["div"]] = float(row["edge_mean"])
                historical_edge_counts[row["div"]] = int(row["sample_size"])

        min_date = post_df["match_date"].min()
        max_date = post_df["match_date"].max()
        default_start = max_date - pd.Timedelta(days=14) if pd.notna(max_date) else min_date
        start_default = default_start if pd.notna(default_start) else min_date
        min_slider = min_date if pd.notna(min_date) else pd.Timestamp.utcnow()
        max_slider = max_date if pd.notna(max_date) else pd.Timestamp.utcnow()

        date_range = st.slider(
            "Date range",
            min_value=min_slider.to_pydatetime(),
            max_value=max_slider.to_pydatetime(),
            value=(start_default.to_pydatetime(), max_slider.to_pydatetime()),
            format="YYYY-MM-DD",
        )

        leagues = sorted(post_df["competition"].unique())
        selected_leagues = st.multiselect("Competitions", leagues, default=leagues, key="analysis_competitions")

        start_ts = pd.Timestamp(date_range[0])
        end_ts = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)

        filtered = post_df[
            (post_df["match_date"] >= start_ts)
            & (post_df["match_date"] < end_ts)
            & (post_df["competition"].isin(selected_leagues))
        ].copy()

        outcome_name_lookup = {"H": "Home", "D": "Draw", "A": "Away"}

        def _alert_flags(row: pd.Series) -> str:
            messages: list[str] = []
            for outcome_key, prob_col in active_prob_cols.items():
                imp_col = implied_lookup[outcome_key]
                prob_val = row.get(prob_col)
                imp_val = row.get(imp_col)
                if pd.isna(prob_val) or pd.isna(imp_val):
                    continue
                diff = float(prob_val - imp_val)
                if abs(diff) >= MARKET_EDGE_WARN:
                    direction = "above" if diff > 0 else "below"
                    messages.append(
                        f"{outcome_name_lookup[outcome_key]} prob {abs(diff) * 100:.2f}% {direction} market"
                    )
            hist_edge = historical_edge_lookup.get(row.get("div"))
            pred_edge = row.get("edge_pred")
            if hist_edge is not None and hist_edge < 0 and pd.notna(pred_edge) and pred_edge > 0:
                sample = historical_edge_counts.get(row.get("div"))
                if sample:
                    messages.append(
                        f"League avg edge {hist_edge * 100:.2f}% over {sample} matches"
                    )
                else:
                    messages.append(
                        f"League avg edge {hist_edge * 100:.2f}% (insufficient history)"
                    )
            return "; ".join(messages)

        filtered["alert_flags"] = filtered.apply(_alert_flags, axis=1)

        if filtered.empty:
            st.info("No completed fixtures in the selected window.")
        else:
            valid = filtered.dropna(subset=["brier"])
            sample_size = int(len(valid))
            hit_rate = float((valid["pred_choice"] == valid["result"]).mean()) if sample_size else float("nan")
            avg_brier = float(valid["brier"].mean()) if sample_size else float("nan")
            avg_log_loss = float(valid["log_loss"].mean()) if sample_size else float("nan")
            avg_edge = float(valid["edge_actual"].mean()) if sample_size else float("nan")
            avg_actual_prob = float(valid["actual_prob"].mean()) if sample_size else float("nan")

            st.subheader("Performance Summary")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Matches", f"{sample_size}")
            col2.metric("Hit Rate", f"{hit_rate * 100:.2f}%" if sample_size else "-")
            col3.metric("Brier Score", f"{avg_brier:.2f}" if sample_size else "-")
            col4.metric("Log Loss", f"{avg_log_loss:.2f}" if sample_size else "-")
            col5.metric("Avg Actual Prob", f"{avg_actual_prob * 100:.2f}%" if sample_size else "-")

            if sample_size:
                grouped = (
                    valid.groupby("competition")
                    .apply(
                        lambda g: pd.Series(
                            {
                                "Matches": len(g),
                                "Hit Rate": (g["pred_choice"] == g["result"]).mean(),
                                "Brier": g["brier"].mean(),
                                "Log Loss": g["log_loss"].mean(),
                                "Avg Edge": g["edge_actual"].mean(),
                            }
                        )
                    )
                    .reset_index()
                )
                grouped["Hit Rate"] = (grouped["Hit Rate"] * 100).round(2)
                grouped["Brier"] = grouped["Brier"].round(2)
                grouped["Log Loss"] = grouped["Log Loss"].round(2)
                grouped["Avg Edge"] = (grouped["Avg Edge"] * 100).round(2)
                st.caption("Edges use calibrated probabilities minus the implied market probabilities (closing odds).")
                st.dataframe(grouped.rename(columns={"competition": "Competition"}), width='stretch')

            display = filtered.copy()
            actual_map = {"H": "Home", "D": "Draw", "A": "Away"}
            display["Actual"] = display["result"].map(actual_map).fillna("")
            display["Actual Code"] = display["result"]
            display = display.drop(columns=["result"], errors="ignore")
            display["Home %"] = (display[active_prob_cols["H"]] * 100).round(2)
            display["Draw %"] = (display[active_prob_cols["D"]] * 100).round(2)
            display["Away %"] = (display[active_prob_cols["A"]] * 100).round(2)
            display["Implied Home %"] = (display["imp_home"] * 100).round(2)
            display["Implied Draw %"] = (display["imp_draw"] * 100).round(2)
            display["Implied Away %"] = (display["imp_away"] * 100).round(2)
            display["Predicted Winner"] = display["pred_choice"].map({"H": "Home", "D": "Draw", "A": "Away"})
            display["Predicted %"] = (display["pred_confidence"] * 100).round(2)
            display["Actual Prob %"] = display["actual_prob"].apply(lambda v: round(v * 100, 2) if pd.notna(v) else np.nan)
            display["Edge (Actual) %"] = display["edge_actual"].apply(lambda v: round(v * 100, 2) if pd.notna(v) else np.nan)
            display["Edge (Pred) %"] = display["edge_pred"].apply(lambda v: round(v * 100, 2) if pd.notna(v) else np.nan)
            display["League Edge %"] = display["div"].map(lambda d: historical_edge_lookup.get(d) * 100 if d in historical_edge_lookup else np.nan)
            display["League Edge N"] = display["div"].map(lambda d: historical_edge_counts.get(d))

            table_cols = [
                "competition",
                "kickoff",
                "home_team",
                "away_team",
                "scoreline",
                "Actual",
                "Predicted Winner",
                "Predicted %",
                "Actual Prob %",
                "Home %",
                "Draw %",
                "Away %",
                "Implied Home %",
                "Implied Draw %",
                "Implied Away %",
                "Edge (Actual) %",
                "Edge (Pred) %",
                "League Edge %",
                "League Edge N",
                "alert_flags",
                "odds_home",
                "odds_draw",
                "odds_away",
                "bookmaker_name",
            ]

            present = [col for col in table_cols if col in display.columns]
            display_table = display[present].rename(
                columns={
                    "competition": "Competition",
                    "kickoff": "Kick-off",
                    "home_team": "Home",
                    "away_team": "Away",
                    "scoreline": "Score",
                    "Actual": "Actual",
                    "alert_flags": "Alerts",
                    "odds_home": "Odds H",
                    "odds_draw": "Odds D",
                    "odds_away": "Odds A",
                    "bookmaker_name": "Bookmaker",
                }
            )

            def _result_row_style(row: pd.Series) -> list[str]:
                actual = row.get("Actual")
                predicted = row.get("Predicted Winner")
                if pd.isna(actual) or pd.isna(predicted):
                    return ["" for _ in row]
                if actual == predicted:
                    return ["background-color: rgba(34, 197, 94, 0.18)"] * len(row)
                return ["background-color: rgba(239, 68, 68, 0.18)"] * len(row)

            def _alert_style(row: pd.Series) -> list[str]:
                styles = ["" for _ in row]
                alerts = row.get("Alerts")
                if isinstance(alerts, str) and alerts:
                    try:
                        alert_idx = list(row.index).index("Alerts")
                    except ValueError:
                        alert_idx = None
                    if alert_idx is not None:
                        styles[alert_idx] = "background-color: rgba(251, 191, 36, 0.25)"
                return styles

            styled_table = display_table.style.apply(_result_row_style, axis=1)
            styled_table = styled_table.apply(_alert_style, axis=1)
            formatters = {col: '{:.2f}'.format for col in display_table.columns if ptypes.is_float_dtype(display_table[col]) and col not in {"League Edge N"}}
            if formatters:
                styled_table = styled_table.format(formatters)
            st.subheader("Match Detail")
            st.dataframe(styled_table, width='stretch')
