from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math


@dataclass
class XGWeights:
    shots_on_target: float = 0.15
    shots: float = 0.05
    goals: float = 0.6
    min_floor: float = 0.2


def estimate_xg(
    shots_total: Optional[float],
    shots_on_target: Optional[float],
    goals: Optional[float],
    weights: XGWeights | None = None,
) -> float:
    weights = weights or XGWeights()
    s_total = max(float(shots_total or 0.0), 0.0)
    s_target = max(min(float(shots_on_target or 0.0), s_total), 0.0)
    goals_val = max(float(goals or 0.0), 0.0)
    raw = (
        weights.shots_on_target * s_target
        + weights.shots * max(s_total - s_target, 0.0)
        + weights.goals * goals_val
    )
    return max(raw, weights.min_floor if raw > 0 else 0.0)


def update_mean(previous_mean: float, count: int, new_value: float, max_matches: int = 20) -> float:
    """Update running mean with optional cap on match history."""
    if count <= 0:
        return new_value
    span = min(count, max_matches)
    decay = (span - 1) / span
    return previous_mean * decay + new_value / span
