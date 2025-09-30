import json
import subprocess
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "database" / "betting.duckdb"
DRAW_ALERT_MIN_PROB = 0.22
DRAW_ALERT_MAX_GAP = 0.08
DRAW_ALERT_XG_GAP = 0.4
DRAW_ALERT_SOT_GAP = 1.5
DRAW_ALERT_ELO_GAP = 35.0
TRUSTED_MARKET_DIVS = {'E0', 'D1', 'I1', 'SP1', 'F1', 'BR1', 'UCL', 'UEL'}
MARKET_EDGE_WARN = 0.08
MARKET_EDGE_CAP = 0.15


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

if st.sidebar.button("Fetch 3-day odds"):
    with st.spinner("Fetching odds (+/-3 days)..."):
        result = subprocess.run([sys.executable, "-m", "src.io.fetch_odds"], cwd=str(BASE_DIR))
    if result.returncode == 0:
        load_post_match_data.clear()
        load_latest_odds.clear()
        st.sidebar.success("Odds updated.")
    else:
        st.sidebar.error(f"Odds fetch failed (exit code {result.returncode}).")

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
fixture_df = load_fixture_previews()
ratings_df = load_team_ratings()
odds_df = load_latest_odds()

if competitions.empty:
    st.warning("Competition metadata not loaded. Populate competitions table first.")

comp_lookup = dict(zip(competitions["div"], competitions["name"])) if not competitions.empty else {}

previews_tab, ratings_tab, analysis_tab = st.tabs(["Fixture Previews", "Team Ratings", "Post-Match Analysis"])

with previews_tab:
    if fixture_df.empty:
        st.info("No fixtures loaded. Import fixtures and run previews.")
    else:
        fixture_df = fixture_df.copy()
        fixture_df["match_date"] = pd.to_datetime(fixture_df["match_date"])
        fixture_df["kickoff"] = fixture_df["match_date"].dt.strftime("%Y-%m-%d %H:%M")
        fixture_df["competition"] = fixture_df["div"].map(comp_lookup).fillna(fixture_df["div"])
        leagues = sorted(fixture_df["competition"].unique())
        selected = st.multiselect("Competitions", leagues, default=leagues)
        lookahead_hours = st.slider("Lookahead (hours)", min_value=6, max_value=240, value=72, step=6)
        now = pd.Timestamp.utcnow().tz_localize(None)
        horizon = now + pd.Timedelta(hours=lookahead_hours)
        filtered = fixture_df[fixture_df["competition"].isin(selected)].copy()
        filtered = filtered[(filtered["match_date"] >= now) & (filtered["match_date"] <= horizon)]
        if filtered.empty:
            st.info("No fixtures in the selected window.")
        else:
            filtered["elo_edge"] = filtered["home_elo"] - filtered["away_elo"]
            prob_matrix = filtered[["prob_home", "prob_draw", "prob_away"]].fillna(0.0).to_numpy()
            fav_idx = prob_matrix.argmax(axis=1)
            fav_prob = prob_matrix.max(axis=1)
            second_prob = np.partition(prob_matrix, -2, axis=1)[:, -2]
            fav_gap = fav_prob - second_prob
            key_lookup = np.array(["prob_home", "prob_draw", "prob_away"])
            label_lookup = np.array(["Home", "Draw", "Away"])
            fav_side = pd.Series(key_lookup[fav_idx], index=filtered.index)
            filtered["fav_label"] = pd.Series(label_lookup[fav_idx], index=filtered.index)
            filtered["fav_prob"] = fav_prob
            filtered["fav_gap"] = fav_gap
            draw_lean_mask = (filtered["prob_draw_raw"] >= 0.14) & (filtered["prob_draw"] >= 0.20) & (fav_gap <= 0.08)
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
            implied_lookup = {"Home": "imp_home", "Draw": "imp_draw", "Away": "imp_away"}
            def _fav_implied(row: pd.Series) -> float | None:
                label = row.get("fav_label")
                if not label:
                    return np.nan
                key = implied_lookup.get(label)
                if not key:
                    return np.nan
                val = row.get(key)
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
                filtered["fav_prob_pct"] = filtered["fav_prob"].apply(lambda v: round(v * 100, 1) if pd.notna(v) else np.nan)
                filtered["fav_implied_pct"] = filtered["fav_implied"].apply(lambda v: round(v * 100, 1) if pd.notna(v) else np.nan)
                filtered["edge_pred_pct"] = filtered["edge_pred"].apply(lambda v: round(v * 100, 1) if pd.notna(v) else np.nan)
                percent_cols = ["prob_home", "prob_draw", "prob_away"]
                filtered[percent_cols] = (filtered[percent_cols] * 100).round(1)
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
                    "prob_home",
                    "prob_draw",
                    "prob_away",
                    "fav_label",
                    "fav_prob_pct",
                    "fav_implied_pct",
                    "edge_pred_pct",
                    "odds_home",
                    "odds_draw",
                    "odds_away",
                    "bookmaker_name",
                    "draw_alert",
                    "market_flag",
                ]
                missing = [col for col in display_cols if col not in filtered.columns]
                for col in missing:
                    filtered[col] = np.nan
                display = filtered[display_cols].copy()
                display.rename(
                    columns={
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
                        "prob_home": "Home %",
                        "prob_draw": "Draw %",
                        "prob_away": "Away %",
                        "fav_label": "Fav",
                        "fav_prob_pct": "Fav %",
                        "fav_implied_pct": "Fav Implied %",
                        "edge_pred_pct": "Edge %",
                        "odds_home": "Odds H",
                        "odds_draw": "Odds D",
                        "odds_away": "Odds A",
                        "bookmaker_name": "Bookmaker",
                        "draw_alert": "Draw Lean",
                        "market_flag": "Market Alert",
                    },
                    inplace=True,
                )
                st.dataframe(display, use_container_width=True)

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
        st.dataframe(display, use_container_width=True)

        if comp_filtered.empty:
            st.caption("No teams available for this competition yet.")
        else:
            team_options = comp_filtered["team"].tolist()
            selected_team = st.selectbox("Team details", team_options)
            row = comp_filtered[comp_filtered["team"] == selected_team].iloc[0]
            col1, col2, col3 = st.columns(3)
            col1.metric("Elo", f"{row['elo']:.1f}")
            col2.metric("xG For", f"{row['xg_for']:.2f}")
            col3.metric("xG Against", f"{row['xg_against']:.2f}")
            form = json.loads(row["rolling_form"]) if isinstance(row["rolling_form"], str) else []
            st.subheader("Recent Form")
            if form:
                form_df = pd.DataFrame(form)
                st.dataframe(form_df, use_container_width=True)
            else:
                st.caption("No recent matches recorded.")


with analysis_tab:
    post_df = load_post_match_data()
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

        outcome_map = {0: "H", 1: "D", 2: "A"}

        def _pick_prediction(row: pd.Series) -> pd.Series:
            probs = [row.get("prob_home"), row.get("prob_draw"), row.get("prob_away")]
            if any(pd.isna(p) for p in probs):
                return pd.Series([None, None])
            idx = int(np.argmax(probs))
            return pd.Series([outcome_map[idx], probs[idx]])

        post_df[["pred_choice", "pred_confidence"]] = post_df.apply(_pick_prediction, axis=1)

        def _actual_prob(row: pd.Series) -> float | None:
            mapping = {"H": "prob_home", "D": "prob_draw", "A": "prob_away"}
            key = mapping.get(row.get("result"))
            if not key or pd.isna(row.get(key)):
                return None
            return float(row.get(key))

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

        post_df["brier"] = post_df.apply(
            lambda row: np.sum(
                [
                    (row.get("prob_home", np.nan) - (1.0 if row.get("result") == "H" else 0.0)) ** 2,
                    (row.get("prob_draw", np.nan) - (1.0 if row.get("result") == "D" else 0.0)) ** 2,
                    (row.get("prob_away", np.nan) - (1.0 if row.get("result") == "A" else 0.0)) ** 2,
                ]
            )
            if row.get("result") in {"H", "D", "A"} and not any(pd.isna([row.get("prob_home"), row.get("prob_draw"), row.get("prob_away")]))
            else np.nan,
            axis=1,
        )

        post_df["log_loss"] = post_df.apply(
            lambda row: -np.log(max(min(row.get("actual_prob", np.nan), 1 - 1e-9), 1e-9))
            if row.get("actual_prob") is not None
            else np.nan,
            axis=1,
        )

        def _edge(row: pd.Series, actual: bool = True) -> float | None:
            mapping = {"H": ("prob_home", "imp_home"), "D": ("prob_draw", "imp_draw"), "A": ("prob_away", "imp_away")}
            key = row.get("result") if actual else row.get("pred_choice")
            if key not in mapping:
                return None
            prob_key, imp_key = mapping[key]
            prob_val = row.get(prob_key)
            imp_val = row.get(imp_key)
            if pd.isna(prob_val) or pd.isna(imp_val):
                return None
            return float(prob_val - imp_val)

        post_df["edge_actual"] = post_df.apply(_edge, axis=1)
        post_df["edge_pred"] = post_df.apply(lambda row: _edge(row, actual=False), axis=1)

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
        selected_leagues = st.multiselect("Competitions", leagues, default=leagues)

        start_ts = pd.Timestamp(date_range[0])
        end_ts = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)

        filtered = post_df[
            (post_df["match_date"] >= start_ts)
            & (post_df["match_date"] < end_ts)
            & (post_df["competition"].isin(selected_leagues))
        ].copy()

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
            col2.metric("Hit Rate", f"{hit_rate * 100:.1f}%" if sample_size else "-")
            col3.metric("Brier Score", f"{avg_brier:.3f}" if sample_size else "-")
            col4.metric("Log Loss", f"{avg_log_loss:.3f}" if sample_size else "-")
            col5.metric("Avg Actual Prob", f"{avg_actual_prob * 100:.1f}%" if sample_size else "-")

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
                grouped["Hit Rate"] = (grouped["Hit Rate"] * 100).round(1)
                grouped["Brier"] = grouped["Brier"].round(3)
                grouped["Log Loss"] = grouped["Log Loss"].round(3)
                grouped["Avg Edge"] = (grouped["Avg Edge"] * 100).round(2)
                st.caption("Edges use our probabilities minus the implied market probabilities (closing odds).")
                st.dataframe(grouped.rename(columns={"competition": "Competition"}), use_container_width=True)

            display = filtered.copy()
            actual_map = {"H": "Home", "D": "Draw", "A": "Away"}
            display["Actual"] = display["result"].map(actual_map).fillna("")
            display["Actual Code"] = display["result"]
            display = display.drop(columns=["result"], errors="ignore")
            display["Home %"] = (display["prob_home"] * 100).round(1)
            display["Draw %"] = (display["prob_draw"] * 100).round(1)
            display["Away %"] = (display["prob_away"] * 100).round(1)
            display["Implied Home %"] = (display["imp_home"] * 100).round(1)
            display["Implied Draw %"] = (display["imp_draw"] * 100).round(1)
            display["Implied Away %"] = (display["imp_away"] * 100).round(1)
            display["Predicted Winner"] = display["pred_choice"].map({"H": "Home", "D": "Draw", "A": "Away"})
            display["Predicted %"] = (display["pred_confidence"] * 100).round(1)
            display["Actual Prob %"] = display["actual_prob"].apply(lambda v: round(v * 100, 1) if pd.notna(v) else np.nan)
            display["Edge (Actual) %"] = display["edge_actual"].apply(lambda v: round(v * 100, 1) if pd.notna(v) else np.nan)
            display["Edge (Pred) %"] = display["edge_pred"].apply(lambda v: round(v * 100, 1) if pd.notna(v) else np.nan)

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

            styled_table = display_table.style.apply(_result_row_style, axis=1)
            st.subheader("Match Detail")
            st.dataframe(styled_table, use_container_width=True)
