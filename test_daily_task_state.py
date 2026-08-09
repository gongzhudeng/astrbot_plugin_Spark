from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from data.plugins.astrbot_plugin_Spark.core.daily_task_state import (
    begin_attempt,
    new_task_state,
    plan_task,
    technical_failure,
    terminal_state,
)
from data.plugins.astrbot_plugin_Spark.main import (
    DailyGreetingTask,
    SessionState,
    Spark,
    UserProfile,
)


def _task(
    target: datetime,
    tag: str = "daily_0_0@2026-08-04->2026-08-04 08:00",
    *,
    ignore_judge: bool = True,
    source_type: str = "fixed",
    interval_minutes: int = 0,
    greeting_id: str = "0",
):
    return DailyGreetingTask(
        slot_num=0,
        greeting_id=greeting_id,
        target=target,
        tag=tag,
        prompt="早上好",
        ignore_dnd=False,
        ignore_judge=ignore_judge,
        cooldown_minutes=0,
        activity_trigger_interval_minutes=interval_minutes,
        source_date=target.date(),
        source_type=source_type,
    )


def _plugin(results: list[bool]) -> Spark:
    plugin = Spark.__new__(Spark)
    plugin.cfg = {
        "enable_daily_greetings": True,
        "daily_prompts": {
            "initial_trigger_grace_seconds": 90,
            "technical_retry_window_minutes": 15,
            "technical_retry_interval_seconds": 60,
            "technical_retry_max_attempts": 3,
        },
    }
    plugin.context = object()
    plugin._proactive_reply = AsyncMock(side_effect=results)
    plugin._judge_should_reply = AsyncMock(return_value=True)
    plugin._save_session_data = lambda: None
    return plugin


class SaveableConfig(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_calls = 0

    def save_config(self):
        self.save_calls += 1


def _provider(response: str = "是") -> SimpleNamespace:
    provider = SimpleNamespace()
    provider.text_chat = AsyncMock(
        return_value=SimpleNamespace(completion_text=response)
    )
    provider.meta = lambda: SimpleNamespace(id="judge")
    return provider


def test_technical_failure_retries_then_success_is_terminal():
    target = 1_000.0
    pending = new_task_state("daily-test", target_at=target, source_date="2026-08-04")
    decision, planned = plan_task(
        pending,
        now_ts=target + 10,
        initial_grace_seconds=90,
        max_attempts=3,
    )
    assert decision == "attempt"

    sending = begin_attempt(planned, now_ts=target + 10, retry_window_seconds=900)
    retrying = technical_failure(
        sending,
        now_ts=target + 20,
        max_attempts=3,
        retry_interval_seconds=60,
    )
    assert retrying.status == "retrying"
    assert retrying.attempts == 1

    decision, _ = plan_task(
        retrying,
        now_ts=retrying.next_retry_at - 1,
        initial_grace_seconds=90,
        max_attempts=3,
    )
    assert decision == "wait"
    decision, second = plan_task(
        retrying,
        now_ts=retrying.next_retry_at,
        initial_grace_seconds=90,
        max_attempts=3,
    )
    assert decision == "attempt"
    sent = terminal_state(
        begin_attempt(second, now_ts=retrying.next_retry_at, retry_window_seconds=900),
        "sent",
        now_ts=retrying.next_retry_at + 1,
    )
    assert sent.terminal is True
    assert sent.attempts == 2


def test_policy_skip_and_missed_are_terminal_without_retry():
    state = new_task_state("daily-test", target_at=1_000, source_date="2026-08-04")
    skipped = terminal_state(
        state,
        "skipped_dnd",
        now_ts=1_010,
        reason="dnd",
    )
    decision, same = plan_task(
        skipped,
        now_ts=2_000,
        initial_grace_seconds=90,
        max_attempts=3,
    )
    assert decision == "terminal"
    assert same.status == "skipped_dnd"

    decision, missed = plan_task(
        state,
        now_ts=1_091,
        initial_grace_seconds=90,
        max_attempts=3,
    )
    assert decision == "terminal"
    assert missed.status == "missed"


def test_retry_window_expiry_and_interrupted_send_do_not_replay():
    state = new_task_state("daily-test", target_at=1_000, source_date="2026-08-04")
    sending = begin_attempt(state, now_ts=1_010, retry_window_seconds=120)
    retrying = technical_failure(
        sending,
        now_ts=1_020,
        max_attempts=3,
        retry_interval_seconds=60,
    )
    decision, expired = plan_task(
        retrying,
        now_ts=1_121,
        initial_grace_seconds=90,
        max_attempts=3,
    )
    assert decision == "terminal"
    assert expired.status == "failed"
    assert expired.reason == "retry_window_expired"

    decision, interrupted = plan_task(
        sending,
        now_ts=1_030,
        initial_grace_seconds=90,
        max_attempts=3,
    )
    assert decision == "terminal"
    assert interrupted.reason == "interrupted_in_flight"


def test_old_last_fired_tag_migrates_and_deduplicates_after_restart():
    tag = "daily_0_0@2026-08-04->2026-08-04 08:00"
    restored = SessionState.from_dict(
        {
            "last_fired_tag": tag,
            "last_fired_tags": {},
            "daily_task_results": {tag: "sent"},
        }
    )

    assert restored.has_fired(tag) is True
    assert restored.daily_task_states[tag]["status"] == "sent"
    again = SessionState.from_dict(restored.to_dict())
    assert again.daily_task_states[tag]["status"] == "sent"


@pytest.mark.asyncio
async def test_scheduler_retries_only_technical_failure_then_sends():
    target = datetime(2026, 8, 4, 8, 0, tzinfo=timezone.utc)
    task = _task(target)
    state = SessionState()
    plugin = _plugin([False, True])
    profile = UserProfile(subscribed=True)

    await plugin._check_daily_greetings(
        "private:1",
        state,
        profile,
        target + timedelta(seconds=10),
        [task],
        "UTC",
        0,
    )
    retrying = state.daily_state(task)
    assert retrying.status == "retrying"
    assert retrying.attempts == 1

    retry_at = datetime.fromtimestamp(retrying.next_retry_at, tz=timezone.utc)
    await plugin._check_daily_greetings(
        "private:1",
        state,
        profile,
        retry_at,
        [task],
        "UTC",
        0,
    )

    assert state.daily_state(task).status == "sent"
    assert plugin._proactive_reply.await_count == 2

    restored = SessionState.from_dict(state.to_dict())
    await plugin._check_daily_greetings(
        "private:1",
        restored,
        profile,
        retry_at + timedelta(minutes=1),
        [task],
        "UTC",
        0,
    )
    assert plugin._proactive_reply.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("dnd", "busy", "expected"),
    [
        (True, False, "skipped_dnd"),
        (False, True, "skipped_busy"),
    ],
)
async def test_scheduler_policy_skip_is_terminal(dnd: bool, busy: bool, expected: str):
    target = datetime(2026, 8, 4, 8, 0, tzinfo=timezone.utc)
    task = _task(target)
    state = SessionState()
    plugin = _plugin([True])

    await plugin._check_daily_greetings(
        "private:1",
        state,
        UserProfile(subscribed=True),
        target + timedelta(seconds=10),
        [task],
        "UTC",
        0,
        is_in_dnd=dnd,
        is_busy=busy,
    )

    assert state.daily_state(task).status == expected
    plugin._proactive_reply.assert_not_awaited()


@pytest.mark.asyncio
async def test_scheduler_missed_task_does_not_send_late():
    target = datetime(2026, 8, 4, 8, 0, tzinfo=timezone.utc)
    task = _task(target)
    state = SessionState()
    plugin = _plugin([True])

    await plugin._check_daily_greetings(
        "private:1",
        state,
        UserProfile(subscribed=True),
        target + timedelta(hours=4),
        [task],
        "UTC",
        0,
    )

    assert state.daily_state(task).status == "missed"
    plugin._proactive_reply.assert_not_awaited()


@pytest.mark.asyncio
async def test_scheduler_ignore_judge_bypasses_judge_and_sends():
    target = datetime(2026, 8, 4, 8, 0, tzinfo=timezone.utc)
    task = _task(target, ignore_judge=True)
    state = SessionState()
    plugin = _plugin([True])

    await plugin._check_daily_greetings(
        "private:judge-switch",
        state,
        UserProfile(subscribed=True),
        target + timedelta(seconds=10),
        [task],
        "UTC",
        0,
    )

    assert state.daily_state(task).status == "sent"
    plugin._judge_should_reply.assert_not_awaited()
    plugin._proactive_reply.assert_awaited_once()


@pytest.mark.asyncio
async def test_scheduler_runs_judge_when_daily_greeting_requires_it():
    target = datetime(2026, 8, 4, 8, 0, tzinfo=timezone.utc)
    task = _task(target, ignore_judge=False)
    state = SessionState()
    plugin = _plugin([True])

    await plugin._check_daily_greetings(
        "private:judge-switch",
        state,
        UserProfile(subscribed=True),
        target + timedelta(seconds=10),
        [task],
        "UTC",
        0,
    )

    assert state.daily_state(task).status == "sent"
    plugin._judge_should_reply.assert_awaited_once_with("private:judge-switch", "UTC")
    plugin._proactive_reply.assert_awaited_once()


@pytest.mark.asyncio
async def test_judge_rejection_is_terminal_and_does_not_retry():
    target = datetime(2026, 8, 4, 8, 0, tzinfo=timezone.utc)
    task = _task(target, ignore_judge=False)
    state = SessionState()
    plugin = _plugin([True])
    plugin._judge_should_reply = AsyncMock(return_value=False)
    now = target + timedelta(seconds=10)

    await plugin._check_daily_greetings(
        "private:judge-switch",
        state,
        UserProfile(subscribed=True),
        now,
        [task],
        "UTC",
        0,
    )
    await plugin._check_daily_greetings(
        "private:judge-switch",
        state,
        UserProfile(subscribed=True),
        now + timedelta(minutes=1),
        [task],
        "UTC",
        0,
    )

    assert state.daily_state(task).status == "skipped_judge"
    assert state.daily_state(task).attempts == 0
    plugin._judge_should_reply.assert_awaited_once()
    plugin._proactive_reply.assert_not_awaited()
    assert state.daily_greeting_success_times == {}


@pytest.mark.asyncio
async def test_activity_interval_skips_before_sixty_minutes():
    target = datetime(2026, 8, 4, 8, 0, tzinfo=timezone.utc)
    now = target + timedelta(seconds=30)
    last_success = now - timedelta(minutes=59)
    task = _task(
        target,
        source_type="activity",
        interval_minutes=60,
    )
    state = SessionState(
        daily_greeting_success_times={task.greeting_id: last_success.timestamp()}
    )
    plugin = _plugin([True])

    await plugin._check_daily_greetings(
        "private:interval",
        state,
        UserProfile(subscribed=True),
        now,
        [task],
        "UTC",
        0,
    )

    assert state.daily_state(task).status == "skipped_interval"
    plugin._proactive_reply.assert_not_awaited()
    assert (
        state.daily_greeting_success_times[task.greeting_id] == last_success.timestamp()
    )


@pytest.mark.asyncio
async def test_activity_interval_allows_exactly_sixty_minutes():
    target = datetime(2026, 8, 4, 8, 0, tzinfo=timezone.utc)
    now = target + timedelta(seconds=30)
    task = _task(
        target,
        source_type="activity",
        interval_minutes=60,
    )
    plugin = _plugin([True])
    state = SessionState(
        daily_greeting_success_times={
            task.greeting_id: (now - timedelta(minutes=60)).timestamp()
        }
    )

    await plugin._check_daily_greetings(
        "private:interval",
        state,
        UserProfile(subscribed=True),
        now,
        [task],
        "UTC",
        0,
    )

    assert state.daily_state(task).status == "sent"
    plugin._proactive_reply.assert_awaited_once()
    assert state.daily_greeting_success_times[task.greeting_id] > 0


@pytest.mark.asyncio
async def test_activity_interval_is_scoped_to_greeting_identity():
    target = datetime(2026, 8, 4, 8, 0, tzinfo=timezone.utc)
    now = target + timedelta(seconds=30)
    task = _task(
        target,
        source_type="activity",
        interval_minutes=60,
        greeting_id="greeting-b",
    )
    state = SessionState(
        daily_greeting_success_times={
            "greeting-a": (now - timedelta(minutes=1)).timestamp()
        }
    )
    plugin = _plugin([True])

    await plugin._check_daily_greetings(
        "private:interval",
        state,
        UserProfile(subscribed=True),
        now,
        [task],
        "UTC",
        0,
    )

    assert state.daily_state(task).status == "sent"
    plugin._proactive_reply.assert_awaited_once()
    assert "greeting-a" in state.daily_greeting_success_times
    assert "greeting-b" in state.daily_greeting_success_times


@pytest.mark.asyncio
async def test_activity_success_interval_changes_only_after_delivery():
    target = datetime(2026, 8, 4, 8, 0, tzinfo=timezone.utc)
    task = _task(
        target,
        source_type="activity",
        interval_minutes=60,
    )
    plugin = _plugin([False])
    state = SessionState()

    await plugin._check_daily_greetings(
        "private:interval",
        state,
        UserProfile(subscribed=True),
        target + timedelta(seconds=30),
        [task],
        "UTC",
        0,
    )

    assert state.daily_state(task).status == "retrying"
    assert state.daily_greeting_success_times == {}

    restored = SessionState.from_dict(state.to_dict())
    assert restored.daily_greeting_success_times == {}
    assert len(restored.daily_task_states) == 1


def test_success_interval_state_is_bounded_across_restart():
    records = {f"greeting-{index}": float(index + 1) for index in range(100)}
    restored = SessionState.from_dict({"daily_greeting_success_times": records})

    assert len(restored.daily_greeting_success_times) == 64
    assert restored.daily_greeting_success_times["greeting-99"] == 100.0
    assert "greeting-0" not in restored.daily_greeting_success_times


@pytest.mark.asyncio
async def test_missing_emotion_callback_safely_returns_empty_context():
    plugin = Spark.__new__(Spark)
    plugin.context = SimpleNamespace()

    assert await plugin._get_emotion_judge_context("private:judge") == ""


def test_daily_control_migration_preserves_old_behavior_and_removes_occurrence():
    plugin = Spark.__new__(Spark)
    plugin.cfg = SaveableConfig(
        {
            "daily_prompts": {
                "daily_greetings": [
                    {
                        "trigger_source": "日程活动",
                        ("activity_" + "occurrences"): "1,2",
                    },
                    {"trigger_source": "固定时间"},
                ]
            }
        }
    )

    plugin._migrate_daily_greeting_controls()

    greetings = plugin.cfg["daily_prompts"]["daily_greetings"]
    assert [item["ignore_judge"] for item in greetings] == [True, True]
    assert greetings[0]["activity_trigger_interval_minutes"] == 0
    assert ("activity_" + "occurrences") not in greetings[0]
    assert plugin.cfg.save_calls == 1


def test_new_daily_task_defaults_to_running_judge():
    plugin = Spark.__new__(Spark)
    target = datetime(2026, 8, 4, 8, 0, tzinfo=timezone.utc)

    task = plugin._daily_task(
        slot_num=0,
        source_date=target.date(),
        base=target,
        item={"prompt": "hello"},
    )

    assert task.ignore_judge is False


@pytest.mark.asyncio
async def test_judge_automatically_appends_emotion_to_old_template():
    provider = _provider("是")
    plugin = Spark.__new__(Spark)
    plugin.cfg = {
        "proactive_settings": {
            "proactive_judge_prompt": "旧字段={legacy_field}",
            "proactive_judge_rules": "只输出是或否",
            "judge_history_rounds": 0,
        },
        "heat_settings": {},
    }
    plugin.context = SimpleNamespace()
    plugin._states = {}
    plugin._refresh_realtime_context = AsyncMock()
    plugin._get_last_messages = AsyncMock(return_value=("last user", "last ai"))
    plugin._get_conversation_contexts = AsyncMock(return_value=[])
    plugin._get_judge_providers = lambda _umo: [provider]
    plugin._resolve_persona = lambda *_keys: "judge persona"
    plugin._get_emotion_judge_context = AsyncMock(
        return_value="<emotion_state_snapshot>当前心境</emotion_state_snapshot>"
    )

    should_reply = await plugin._judge_should_reply("private:judge", "UTC")

    assert should_reply is True
    sent_contexts = provider.text_chat.await_args.kwargs["contexts"]
    judge_prompt = sent_contexts[-2]["content"]
    assert "{legacy_field}" in judge_prompt
    assert "内心世界：" in judge_prompt
    assert "当前心境" in judge_prompt
    plugin._get_emotion_judge_context.assert_awaited_once_with("private:judge")
