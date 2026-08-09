from datetime import datetime

from core.time_policy import (
    OffsetMode,
    TimePolicy,
    apply_datetime_policy,
    apply_delay_policy,
    choose_offset_minutes,
    cooldown_deadline,
    migrate_compact_policy_values,
    parse_policy,
)


def test_legacy_jitter_maps_to_random_both():
    policy = parse_policy({"jitter_minutes": 15}, legacy_jitter_key="jitter_minutes")
    assert policy.mode == OffsetMode.RANDOM_BOTH
    assert policy.minutes == 15


def test_compact_policy_supports_fixed_and_random_ranges():
    fixed_delay = parse_policy(
        {"jitter_minutes": "+5"}, legacy_jitter_key="jitter_minutes"
    )
    fixed_advance = parse_policy(
        {"jitter_minutes": "-5"}, legacy_jitter_key="jitter_minutes"
    )
    random_range = parse_policy(
        {"jitter_minutes": "+15~+25"}, legacy_jitter_key="jitter_minutes"
    )

    assert fixed_delay == TimePolicy(OffsetMode.FIXED_DELAY, minutes=5)
    assert fixed_advance == TimePolicy(OffsetMode.FIXED_ADVANCE, minutes=5)
    assert random_range == TimePolicy(
        OffsetMode.CUSTOM_RANGE,
        minimum_minutes=15,
        maximum_minutes=25,
    )


def test_compact_policy_accepts_full_width_separator_and_reversed_range():
    policy = parse_policy(
        {"jitter_minutes": "+2～-1"}, legacy_jitter_key="jitter_minutes"
    )
    assert choose_offset_minutes(policy, seed="range") in {-1, 0, 1, 2}


def test_explicit_legacy_fields_still_take_priority():
    policy = parse_policy(
        {
            "jitter_minutes": "+1~+2",
            "offset_mode": "fixed_advance",
            "offset_minutes": 8,
        },
        legacy_jitter_key="jitter_minutes",
    )
    assert policy == TimePolicy(OffsetMode.FIXED_ADVANCE, minutes=8)


def test_migrate_compact_policy_values_converts_all_legacy_integers():
    config = {
        "idle_greetings": {"idle_random_fluctuation_minutes": 2},
        "daily_prompts": {
            "daily_greetings": [
                {"enable": False, "jitter_minutes": 13},
                {"enable": True, "jitter_minutes": "+1~+2"},
            ]
        },
    }

    assert migrate_compact_policy_values(config) is True
    assert config["idle_greetings"]["idle_random_fluctuation_minutes"] == "2"
    assert config["daily_prompts"]["daily_greetings"][0]["jitter_minutes"] == "13"
    assert migrate_compact_policy_values(config) is False


def test_cooldown_deadline_uses_latest_activity_plus_minutes():
    assert cooldown_deadline(1_000.0, 10) == 1_600.0
    assert cooldown_deadline(1_000.0, 0) == 0.0
    assert cooldown_deadline(0.0, 10) == 0.0


def test_fixed_offsets_cross_day_naturally():
    base = datetime(2026, 7, 16, 0, 10)
    result = apply_datetime_policy(
        base,
        TimePolicy(OffsetMode.FIXED_ADVANCE, minutes=30),
        seed="fixed",
    )
    assert result == datetime(2026, 7, 15, 23, 40)


def test_random_policy_is_stable_for_same_seed():
    policy = TimePolicy(OffsetMode.CUSTOM_RANGE, minimum_minutes=-20, maximum_minutes=5)
    base = datetime(2026, 7, 16, 8, 0)
    assert apply_datetime_policy(base, policy, seed="same") == apply_datetime_policy(
        base, policy, seed="same"
    )


def test_invalid_random_delay_is_retryable():
    result = apply_delay_policy(
        1,
        TimePolicy(OffsetMode.CUSTOM_RANGE, minimum_minutes=-10, maximum_minutes=-10),
        seed="retry",
    )
    assert result.minutes is None
    assert result.retryable is True


def test_invalid_fixed_delay_ends_current_round():
    result = apply_delay_policy(
        10,
        TimePolicy(OffsetMode.FIXED_ADVANCE, minutes=20),
        seed="fixed",
    )
    assert result.minutes is None
    assert result.retryable is False
