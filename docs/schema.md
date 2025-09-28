# Data Model Overview

## Tables

- competitions(div TEXT PRIMARY KEY, name TEXT, region TEXT, baseline_elo DOUBLE, k_factor DOUBLE, home_field DOUBLE)
- matches(match_id INTEGER PRIMARY KEY, div TEXT, season TEXT, match_date TIMESTAMP, home_team TEXT, away_team TEXT,
          home_goals INTEGER, away_goals INTEGER, result TEXT)
- match_stats(match_id INTEGER, stat_name TEXT, stat_value DOUBLE)
- fixtures_queue(fixt_id INTEGER PRIMARY KEY, div TEXT, match_date TIMESTAMP, home_team TEXT, away_team TEXT, source_file TEXT)
- team_ratings(div TEXT, team TEXT, season TEXT, elo DOUBLE, xg_for DOUBLE, xg_against DOUBLE, updated_at TIMESTAMP, matches_played INTEGER,
              rolling_form JSON)

## Views

- v_team_form: combines team_ratings with last 5 match summary from matches
- v_fixture_previews: fixtures_queue joined to team_ratings for quick preview output

Season keys derive from year in Date column (split on /).
