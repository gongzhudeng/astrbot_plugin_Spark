"""Pure heat and delay policies for proactive conversations."""

from __future__ import annotations

import math
from collections.abc import Iterable


def decayed_heat(
    timestamps: Iterable[float],
    now_ts: float,
    window_minutes: float,
    messages_for_full_score: float,
) -> float:
    """Return a normalized density score with exponentially fading message weight."""
    window_seconds = max(float(window_minutes or 1), 1.0) * 60.0
    full_score = max(float(messages_for_full_score or 1), 1.0)
    total = sum(
        math.exp(-3.0 * (now_ts - timestamp) / window_seconds)
        for timestamp in timestamps
        if 0 <= now_ts - timestamp <= window_seconds
    )
    return min(total / full_score, 1.0)


def dual_scale_heat(
    timestamps: Iterable[float],
    now_ts: float,
    *,
    short_window_minutes: float,
    long_window_minutes: float,
    messages_for_full_score: float,
    short_weight: float,
) -> float:
    """Blend quick conversation momentum with slower same-day residual warmth."""
    values = list(timestamps)
    short_score = decayed_heat(
        values,
        now_ts,
        short_window_minutes,
        messages_for_full_score,
    )
    long_score = decayed_heat(
        values,
        now_ts,
        long_window_minutes,
        messages_for_full_score,
    )
    short_weight = min(max(float(short_weight), 0.0), 1.0)
    return min(short_score * short_weight + long_score * (1.0 - short_weight), 1.0)


def geometric_delay(
    hot_delay: float,
    cold_delay: float,
    heat: float,
) -> float:
    """Map heat to delay with multiplicative, rather than linear, spacing."""
    hot = max(float(hot_delay), 1.0)
    cold = max(float(cold_delay), hot)
    normalized_heat = min(max(float(heat), 0.0), 1.0)
    return hot * (cold / hot) ** (1.0 - normalized_heat)


def heat_scaled_delay_seconds(
    hot_delay_seconds: float,
    cold_delay_seconds: float,
    heat: float,
) -> int:
    """Return a rounded non-negative delay for short-lived enhancement tasks."""
    return max(0, round(geometric_delay(hot_delay_seconds, cold_delay_seconds, heat)))
