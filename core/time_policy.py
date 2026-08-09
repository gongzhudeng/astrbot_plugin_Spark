"""Reusable time policies for proactive scheduling."""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class OffsetMode(str, Enum):
    NONE = "none"
    RANDOM_BOTH = "random_both"
    RANDOM_ADVANCE = "random_advance"
    RANDOM_DELAY = "random_delay"
    FIXED_ADVANCE = "fixed_advance"
    FIXED_DELAY = "fixed_delay"
    CUSTOM_RANGE = "custom_range"


RANDOM_MODES = {
    OffsetMode.RANDOM_BOTH,
    OffsetMode.RANDOM_ADVANCE,
    OffsetMode.RANDOM_DELAY,
    OffsetMode.CUSTOM_RANGE,
}


@dataclass(frozen=True)
class TimePolicy:
    mode: OffsetMode
    minutes: int = 0
    minimum_minutes: int = 0
    maximum_minutes: int = 0

    @property
    def is_random(self) -> bool:
        return self.mode in RANDOM_MODES


@dataclass(frozen=True)
class DelayResult:
    minutes: float | None
    retryable: bool = False

    @property
    def is_valid(self) -> bool:
        return self.minutes is not None and self.minutes > 0


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_compact_policy(value: object) -> TimePolicy | None:
    if isinstance(value, bool) or value is None:
        return None
    text = str(value).strip().replace("～", "~")
    if not text:
        return None

    range_match = re.fullmatch(r"([+-]?\d+)\s*~\s*([+-]?\d+)", text)
    if range_match:
        minimum, maximum = (int(part) for part in range_match.groups())
        if minimum == maximum == 0:
            return TimePolicy(OffsetMode.NONE)
        return TimePolicy(
            OffsetMode.CUSTOM_RANGE,
            minimum_minutes=minimum,
            maximum_minutes=maximum,
        )

    if not re.fullmatch(r"[+-]?\d+", text):
        return None
    minutes = int(text)
    if minutes == 0:
        return TimePolicy(OffsetMode.NONE)
    if text.startswith("+"):
        return TimePolicy(OffsetMode.FIXED_DELAY, minutes=minutes)
    if text.startswith("-"):
        return TimePolicy(OffsetMode.FIXED_ADVANCE, minutes=abs(minutes))
    return TimePolicy(OffsetMode.RANDOM_BOTH, minutes=minutes)


def parse_policy(
    config: Mapping[str, object],
    *,
    prefix: str = "offset",
    legacy_jitter_key: str | None = None,
) -> TimePolicy:
    """Normalize compact and legacy policy fields without changing old behavior."""
    raw_mode = config.get(f"{prefix}_mode")
    if raw_mode in (None, ""):
        compact = (
            _parse_compact_policy(config.get(legacy_jitter_key))
            if legacy_jitter_key
            else None
        )
        return compact or TimePolicy(OffsetMode.NONE)

    try:
        mode = OffsetMode(str(raw_mode))
    except ValueError:
        mode = OffsetMode.NONE
    return TimePolicy(
        mode=mode,
        minutes=max(0, _as_int(config.get(f"{prefix}_minutes"))),
        minimum_minutes=_as_int(config.get(f"{prefix}_min_minutes")),
        maximum_minutes=_as_int(config.get(f"{prefix}_max_minutes")),
    )


def stable_rng(seed: str) -> random.Random:
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def choose_offset_minutes(policy: TimePolicy, *, seed: str) -> int:
    minutes = max(0, policy.minutes)
    rng = stable_rng(seed)
    if policy.mode == OffsetMode.NONE:
        return 0
    if policy.mode == OffsetMode.RANDOM_BOTH:
        return rng.randint(-minutes, minutes)
    if policy.mode == OffsetMode.RANDOM_ADVANCE:
        return rng.randint(-minutes, 0)
    if policy.mode == OffsetMode.RANDOM_DELAY:
        return rng.randint(0, minutes)
    if policy.mode == OffsetMode.FIXED_ADVANCE:
        return -minutes
    if policy.mode == OffsetMode.FIXED_DELAY:
        return minutes
    minimum = min(policy.minimum_minutes, policy.maximum_minutes)
    maximum = max(policy.minimum_minutes, policy.maximum_minutes)
    return rng.randint(minimum, maximum)


def apply_datetime_policy(base: datetime, policy: TimePolicy, *, seed: str) -> datetime:
    return base + timedelta(minutes=choose_offset_minutes(policy, seed=seed))


def apply_delay_policy(
    base_minutes: float, policy: TimePolicy, *, seed: str
) -> DelayResult:
    result = float(base_minutes) + choose_offset_minutes(policy, seed=seed)
    if result > 0:
        return DelayResult(result)
    return DelayResult(None, retryable=policy.is_random)


def migrate_compact_policy_values(config: dict) -> bool:
    """Convert legacy numeric policy values to the schema's compact strings."""
    changed = False
    idle = config.get("idle_greetings")
    if isinstance(idle, dict):
        value = idle.get("idle_random_fluctuation_minutes")
        if isinstance(value, int) and not isinstance(value, bool):
            idle["idle_random_fluctuation_minutes"] = str(value)
            changed = True

    daily = config.get("daily_prompts")
    greetings = daily.get("daily_greetings") if isinstance(daily, dict) else None
    if isinstance(greetings, list):
        for greeting in greetings:
            if not isinstance(greeting, dict):
                continue
            value = greeting.get("jitter_minutes")
            if isinstance(value, int) and not isinstance(value, bool):
                greeting["jitter_minutes"] = str(value)
                changed = True
    return changed


def cooldown_deadline(latest_activity_ts: float, cooldown_minutes: int) -> float:
    if latest_activity_ts <= 0 or cooldown_minutes <= 0:
        return 0.0
    return latest_activity_ts + cooldown_minutes * 60
