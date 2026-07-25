from core.heat_policy import (
    decayed_heat,
    dual_scale_heat,
    geometric_delay,
    heat_scaled_delay_seconds,
)


def test_geometric_delay_keeps_configured_endpoints():
    assert geometric_delay(30, 200, 0.0) == 200
    assert geometric_delay(30, 200, 1.0) == 30


def test_geometric_delay_is_nonlinear_and_monotonic():
    delays = [geometric_delay(30, 200, heat) for heat in (0.0, 0.2, 0.5, 0.8, 1.0)]

    assert delays == sorted(delays, reverse=True)
    assert 76 < delays[2] < 78
    assert delays[2] < 115


def test_dual_scale_heat_preserves_residual_warmth_after_short_window():
    now = 100_000.0
    messages = [now - 120 * 60, now - 180 * 60]

    short_only = decayed_heat(messages, now, 45, 6)
    blended = dual_scale_heat(
        messages,
        now,
        short_window_minutes=45,
        long_window_minutes=720,
        messages_for_full_score=6,
        short_weight=0.7,
    )

    assert short_only == 0.0
    assert blended > 0.0


def test_dense_recent_chat_scores_higher_than_old_chat():
    now = 100_000.0
    recent = [now - index * 60 for index in range(6)]
    old = [now - (240 + index) * 60 for index in range(6)]
    options = {
        "short_window_minutes": 45,
        "long_window_minutes": 720,
        "messages_for_full_score": 6,
        "short_weight": 0.7,
    }

    assert dual_scale_heat(recent, now, **options) > dual_scale_heat(
        old, now, **options
    )


def test_heat_scaled_enhancement_delay_uses_same_mapping():
    assert heat_scaled_delay_seconds(45, 600, 0.0) == 600
    assert heat_scaled_delay_seconds(45, 600, 1.0) == 45
    assert 160 < heat_scaled_delay_seconds(45, 600, 0.5) < 165
