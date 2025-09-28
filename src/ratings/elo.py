from __future__ import annotations

from dataclasses import dataclass


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


@dataclass
class EloParameters:
    baseline: float = 1500.0
    k_factor: float = 20.0
    home_field: float = 50.0
    season_regression: float = 0.25  # weight to baseline at season reset


def update_pair(
    home_rating: float,
    away_rating: float,
    result: float,
    params: EloParameters,
) -> tuple[float, float]:
    """Update Elo pair; result=1 home win, 0 draw? we use 1,0.5,0."""
    expected_home = expected_score(home_rating + params.home_field, away_rating)
    expected_away = expected_score(away_rating, home_rating + params.home_field)

    delta_home = params.k_factor * (result - expected_home)
    delta_away = params.k_factor * ((1.0 - result) - expected_away)

    return home_rating + delta_home, away_rating + delta_away


def regress_towards_baseline(current: float, params: EloParameters) -> float:
    return current * (1.0 - params.season_regression) + params.baseline * params.season_regression
