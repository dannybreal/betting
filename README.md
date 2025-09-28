# Fixture Insights

Compact workspace for maintaining offline Elo/xG ratings, fixture probabilities, and odds overlays for a Streamlit dashboard.

## Prerequisites

- Python 3.10+
- Install dependencies via `pip install -e .` (uses `pyproject.toml`)

## Workflow

1. **Configure competitions** – edit `config/competitions.yml` for baseline Elo, K-factor, and home-field advantage per division.
2. **Import results** – load historical CSVs with `python -m src.io.loaders results Latest_Results.csv` (repeat per league file as needed).
3. **Import fixtures** – add upcoming matches via `python -m src.io.loaders fixtures fixtures.csv`.
4. **Update ratings** – run `python -m src.ratings.pipeline update` to refresh team Elo/xG state in `database/betting.duckdb`.
5. **Generate previews** – run `python -m src.ratings.pipeline preview` to produce win/draw probabilities and snapshots.
6. **Dashboard** – start `streamlit run streamlit_app.py` to browse upcoming fixtures, odds edges, and post-match analysis.

## Key Outputs

- `database/betting.duckdb` – canonical storage for fixtures, odds, ratings, and results.
- `reports/fixture_previews.csv` – latest preview export (also logged in history tables).
- `reports/draw_override_events.csv` – audit trail whenever draw overrides are applied.
- `reports/team_ratings.csv` – most recent Elo/xG metrics per club.

## Tests

Run `pytest` from the repo root to validate Elo update helpers and calibration utilities.

## Automation Tips

- `scripts/run_preview.py` (or the PowerShell wrapper) regenerates previews and logs history; schedule it ahead of kick-offs for automated refreshes.
- Use the Streamlit sidebar actions (`Refresh data`, `Rebuild previews`, `Fetch odds`) for manual updates without leaving the UI.

## Notes

- Inter-league strength is blended when teams cross into Europe to keep probabilities realistic even without fresh strength indices.
- Draw probability logic is calibrated and capped to target league-level draw rates while respecting xG/SoT parity.
- Odds fetching works within a ±3-day window; rerun when new markets are posted to update implied probabilities and edge calculations.
