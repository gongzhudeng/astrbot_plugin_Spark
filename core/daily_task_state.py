"""Pure durable state transitions for concrete daily greeting tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

DAILY_STATE_SCHEMA_VERSION = 2
DEFAULT_MAX_RECORDS = 96
DEFAULT_MAX_SUCCESS_RECORDS = 64

DailyStatus = Literal[
    "pending",
    "sending",
    "retrying",
    "sent",
    "skipped_cooldown",
    "skipped_dnd",
    "skipped_busy",
    "skipped_judge",
    "skipped_interval",
    "failed",
    "missed",
    "legacy_processed",
]
TERMINAL_STATUSES = {
    "sent",
    "skipped_cooldown",
    "skipped_dnd",
    "skipped_busy",
    "skipped_judge",
    "skipped_interval",
    "failed",
    "missed",
    "legacy_processed",
}
VALID_STATUSES = TERMINAL_STATUSES | {"pending", "sending", "retrying"}


@dataclass(frozen=True)
class DailyTaskState:
    tag: str
    target_at: float
    source_date: str
    status: DailyStatus = "pending"
    attempts: int = 0
    first_attempt_at: float = 0.0
    last_attempt_at: float = 0.0
    next_retry_at: float = 0.0
    retry_deadline_at: float = 0.0
    completed_at: float = 0.0
    reason: str = ""

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, tag: str = "") -> DailyTaskState | None:
        normalized_tag = str(data.get("tag") or tag).strip()[:180]
        status = str(data.get("status") or "pending")
        if not normalized_tag or status not in VALID_STATUSES:
            return None
        try:
            return cls(
                tag=normalized_tag,
                target_at=max(0.0, float(data.get("target_at", 0.0) or 0.0)),
                source_date=str(data.get("source_date") or "")[:10],
                status=status,
                attempts=max(0, int(data.get("attempts", 0) or 0)),
                first_attempt_at=max(
                    0.0, float(data.get("first_attempt_at", 0.0) or 0.0)
                ),
                last_attempt_at=max(
                    0.0, float(data.get("last_attempt_at", 0.0) or 0.0)
                ),
                next_retry_at=max(0.0, float(data.get("next_retry_at", 0.0) or 0.0)),
                retry_deadline_at=max(
                    0.0, float(data.get("retry_deadline_at", 0.0) or 0.0)
                ),
                completed_at=max(0.0, float(data.get("completed_at", 0.0) or 0.0)),
                reason=str(data.get("reason") or "")[:120],
            )
        except (TypeError, ValueError):
            return None


def new_task_state(tag: str, *, target_at: float, source_date: str) -> DailyTaskState:
    return DailyTaskState(
        tag=str(tag).strip()[:180],
        target_at=max(0.0, float(target_at)),
        source_date=str(source_date)[:10],
    )


def terminal_state(
    state: DailyTaskState,
    status: DailyStatus,
    *,
    now_ts: float,
    reason: str = "",
) -> DailyTaskState:
    if status not in TERMINAL_STATUSES:
        raise ValueError(f"non-terminal daily status: {status}")
    data = state.to_dict()
    data.update(
        status=status,
        next_retry_at=0.0,
        completed_at=max(0.0, float(now_ts)),
        reason=str(reason or "")[:120],
    )
    return DailyTaskState(**data)


def recover_interrupted_state(
    state: DailyTaskState, *, now_ts: float
) -> DailyTaskState:
    """Avoid duplicate delivery when a process stopped during an ambiguous send."""
    if state.status != "sending":
        return state
    return terminal_state(
        state,
        "failed",
        now_ts=now_ts,
        reason="interrupted_in_flight",
    )


def plan_task(
    state: DailyTaskState,
    *,
    now_ts: float,
    initial_grace_seconds: int,
    max_attempts: int,
) -> tuple[Literal["wait", "attempt", "terminal"], DailyTaskState]:
    """Decide whether a projected task should wait, execute, or stop."""
    now_ts = max(0.0, float(now_ts))
    grace = max(30, int(initial_grace_seconds))
    max_attempts = max(1, int(max_attempts))
    recovered = recover_interrupted_state(state, now_ts=now_ts)
    if recovered.terminal:
        return "terminal", recovered
    if now_ts < recovered.target_at:
        return "wait", recovered
    if recovered.status == "pending":
        if now_ts > recovered.target_at + grace:
            return "terminal", terminal_state(
                recovered,
                "missed",
                now_ts=now_ts,
                reason="initial_window_expired",
            )
        return "attempt", recovered
    if recovered.status == "retrying":
        if recovered.attempts >= max_attempts:
            return "terminal", terminal_state(
                recovered,
                "failed",
                now_ts=now_ts,
                reason="max_attempts_reached",
            )
        if recovered.retry_deadline_at and now_ts > recovered.retry_deadline_at:
            return "terminal", terminal_state(
                recovered,
                "failed",
                now_ts=now_ts,
                reason="retry_window_expired",
            )
        if now_ts < recovered.next_retry_at:
            return "wait", recovered
        return "attempt", recovered
    return "wait", recovered


def begin_attempt(
    state: DailyTaskState,
    *,
    now_ts: float,
    retry_window_seconds: int,
) -> DailyTaskState:
    now_ts = max(0.0, float(now_ts))
    first_attempt = state.first_attempt_at or now_ts
    deadline = state.retry_deadline_at or (
        state.target_at + max(60, int(retry_window_seconds))
    )
    data = state.to_dict()
    data.update(
        status="sending",
        attempts=state.attempts + 1,
        first_attempt_at=first_attempt,
        last_attempt_at=now_ts,
        next_retry_at=0.0,
        retry_deadline_at=deadline,
        reason="",
    )
    return DailyTaskState(**data)


def technical_failure(
    state: DailyTaskState,
    *,
    now_ts: float,
    max_attempts: int,
    retry_interval_seconds: int,
) -> DailyTaskState:
    now_ts = max(0.0, float(now_ts))
    max_attempts = max(1, int(max_attempts))
    interval = max(30, int(retry_interval_seconds))
    if state.attempts >= max_attempts:
        return terminal_state(
            state,
            "failed",
            now_ts=now_ts,
            reason="max_attempts_reached",
        )
    delay = interval * (2 ** max(0, state.attempts - 1))
    next_retry = now_ts + delay
    if state.retry_deadline_at and next_retry > state.retry_deadline_at:
        return terminal_state(
            state,
            "failed",
            now_ts=now_ts,
            reason="retry_window_expired",
        )
    data = state.to_dict()
    data.update(
        status="retrying",
        next_retry_at=next_retry,
        reason="technical_failure",
    )
    return DailyTaskState(**data)


def legacy_daily_state(
    tag: str,
    result: object,
    *,
    fired_at: float = 0.0,
) -> DailyTaskState:
    legacy = str(result or "")
    status_by_result: dict[str, DailyStatus] = {
        "sent": "sent",
        "cooldown_skipped": "skipped_cooldown",
        "skipped_cooldown": "skipped_cooldown",
        "skipped_dnd": "skipped_dnd",
        "skipped_busy": "skipped_busy",
        "skipped_judge": "skipped_judge",
        "skipped_interval": "skipped_interval",
        "failed": "failed",
        "missed": "missed",
    }
    status = status_by_result.get(legacy, "legacy_processed")
    timestamp = max(0.0, float(fired_at or 0.0))
    return DailyTaskState(
        tag=str(tag).strip()[:180],
        target_at=0.0,
        source_date="",
        status=status,
        completed_at=timestamp,
        reason="legacy_state",
    )


def normalize_daily_states(
    records: object,
    *,
    legacy_results: object = None,
    last_fired_tags: object = None,
    max_records: int = DEFAULT_MAX_RECORDS,
) -> dict[str, dict[str, Any]]:
    normalized: dict[str, DailyTaskState] = {}
    if isinstance(records, dict):
        for tag, raw in records.items():
            if not isinstance(raw, dict):
                continue
            state = DailyTaskState.from_dict(raw, tag=str(tag))
            if state is not None:
                normalized[state.tag] = state

    results = legacy_results if isinstance(legacy_results, dict) else {}
    fired = last_fired_tags if isinstance(last_fired_tags, dict) else {}
    for tag, fired_at in fired.items():
        tag = str(tag)
        if not tag.startswith("daily_") or tag in normalized:
            continue
        normalized[tag] = legacy_daily_state(
            tag,
            results.get(tag),
            fired_at=fired_at,
        )

    ordered = sorted(
        normalized.values(),
        key=lambda item: (
            item.target_at or item.completed_at or item.last_attempt_at,
            item.tag,
        ),
    )[-max(1, int(max_records)) :]
    return {item.tag: item.to_dict() for item in ordered}


def store_daily_state(
    records: object,
    state: DailyTaskState,
    *,
    max_records: int = DEFAULT_MAX_RECORDS,
) -> dict[str, dict[str, Any]]:
    normalized = normalize_daily_states(records, max_records=max_records)
    normalized[state.tag] = state.to_dict()
    return normalize_daily_states(normalized, max_records=max_records)


def normalize_success_times(
    records: object,
    *,
    max_records: int = DEFAULT_MAX_SUCCESS_RECORDS,
) -> dict[str, float]:
    normalized: dict[str, float] = {}
    if isinstance(records, dict):
        for raw_key, raw_timestamp in records.items():
            key = str(raw_key or "").strip()[:64]
            if not key:
                continue
            try:
                timestamp = float(raw_timestamp or 0.0)
            except (TypeError, ValueError):
                continue
            if timestamp > 0:
                normalized[key] = timestamp
    ordered = sorted(normalized.items(), key=lambda item: (item[1], item[0]))[
        -max(1, int(max_records)) :
    ]
    return dict(ordered)


def success_interval_deadline(
    records: object,
    greeting_id: str,
    interval_minutes: int,
) -> float:
    if interval_minutes <= 0:
        return 0.0
    last_success = normalize_success_times(records).get(str(greeting_id), 0.0)
    return last_success + interval_minutes * 60 if last_success > 0 else 0.0


def record_success_time(
    records: object,
    greeting_id: str,
    delivered_at: float,
    *,
    max_records: int = DEFAULT_MAX_SUCCESS_RECORDS,
) -> dict[str, float]:
    normalized = normalize_success_times(records, max_records=max_records)
    key = str(greeting_id or "").strip()[:64]
    timestamp = max(0.0, float(delivered_at or 0.0))
    if key and timestamp > 0:
        normalized[key] = timestamp
    return normalize_success_times(normalized, max_records=max_records)
