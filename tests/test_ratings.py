from __future__ import annotations

from src.ratings import elo, xg


def test_expected_score_symmetry():
    assert elo.expected_score(1500, 1500) == elo.expected_score(1500, 1500)
    assert 0.48 < elo.expected_score(1500, 1510) < 0.52


def test_update_pair_home_advantage():
    params = elo.EloParameters(baseline=1500, k_factor=20, home_field=50)
    home_new, away_new = elo.update_pair(1500, 1500, 1.0, params)
    assert home_new > 1500
    assert away_new < 1500


def test_xg_estimate_scales_with_shots():
    base = xg.estimate_xg(10, 4, 0)
    higher = xg.estimate_xg(12, 5, 1)
    assert higher > base


def test_xg_running_mean_updates():
    mean = xg.update_mean(0.0, 0, 1.0)
    mean = xg.update_mean(mean, 1, 2.0)
    assert mean > 1.0
