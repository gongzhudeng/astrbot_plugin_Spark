import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from data.plugins.astrbot_plugin_Spark.main import SessionState, Spark, UserProfile

UMO = "default:FriendMessage:idle-test"


def _plugin(mode: str = "大模型判断") -> tuple[Spark, SessionState, UserProfile]:
    plugin = Spark.__new__(Spark)
    plugin.cfg = {
        "idle_greetings": {
            "enable_idle_greetings": True,
            "mode": mode,
            "ignore_judge": False,
            "idle_after_minutes": 120,
            "idle_random_fluctuation_minutes": "0",
            "hot_delay_minutes": 30,
            "cold_delay_minutes": 200,
            "judge_after_minutes": 60,
            "judge_min_delay_minutes": 5,
            "judge_max_delay_minutes": 180,
            "judge_retry_minutes": 5,
            "cooldown_minutes": 0,
            "idle_prompt_templates": ["继续聊聊"],
        },
        "heat_settings": {
            "enable_heat": True,
            "heat_window_minutes": 45,
            "heat_long_window_minutes": 720,
            "heat_short_weight": 0.7,
            "heat_messages_for_full_score": 10,
        },
    }
    plugin.context = SimpleNamespace()
    plugin._save_session_data = MagicMock()
    plugin._judge_idle_delay_minutes = AsyncMock(return_value=0)
    plugin._proactive_reply = AsyncMock(return_value=True)

    profile = UserProfile(subscribed=True)
    state = SessionState(last_user_reply_ts=1_000.0, msg_timestamps=[])
    plugin._states = {UMO: state}
    plugin._user_profiles = {UMO: profile}
    plugin._reset_idle_schedule(state, 1_000.0, profile)
    return plugin, state, profile


def _at(timestamp: float) -> datetime:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("0", 0),
        ("5", 5),
        ("180", 180),
        (" 15 ", 15),
        ("4", None),
        ("181", None),
        ("-1", None),
        ("+5", None),
        ("5分钟", None),
        ("delay=5", None),
        ('{"minutes": 5}', None),
        ("５", None),
        ("5\n分钟", None),
        ("", None),
    ],
)
def test_delay_protocol_accepts_only_complete_ascii_integer(
    response: str, expected: int | None
):
    assert Spark._parse_delay_minutes(response, 5, 180) == expected


def test_inflight_judgement_retries_after_restart():
    restored = SessionState.from_dict(
        {
            "idle_judge_cycle": 7,
            "idle_judge_checked_cycle": 7,
            "idle_judge_inflight_cycle": 7,
            "idle_judge_anchor_ts": 1_000.0,
            "idle_judge_task_ts": 0.0,
            "next_idle_ts": 0.0,
        }
    )

    assert restored.idle_judge_checked_cycle == -1
    assert restored.idle_judge_inflight_cycle == -1
    assert restored.next_idle_ts == 0.0


def test_persisted_candidate_survives_restart_without_rejudging():
    restored = SessionState.from_dict(
        {
            "idle_judge_cycle": 7,
            "idle_judge_checked_cycle": 7,
            "idle_judge_inflight_cycle": -1,
            "idle_judge_anchor_ts": 1_000.0,
            "idle_judge_task_ts": 5_000.0,
            "next_idle_ts": 5_000.0,
            "idle_schedule_mode": "大模型判断",
        }
    )

    assert restored.idle_judge_checked_cycle == 7
    assert restored.idle_judge_task_ts == 5_000.0
    assert restored.next_idle_ts == 5_000.0


def test_new_activity_invalidates_candidate_and_starts_new_cycle():
    plugin, state, profile = _plugin()
    old_cycle = state.idle_judge_cycle
    state.idle_judge_checked_cycle = old_cycle
    state.idle_judge_task_ts = 8_000.0
    state.next_idle_ts = 8_000.0

    plugin._reset_idle_schedule(state, 2_000.0, profile)

    assert state.idle_judge_cycle == old_cycle + 1
    assert state.idle_judge_checked_cycle == -1
    assert state.idle_judge_task_ts == 0.0
    assert state.idle_judge_anchor_ts == 2_000.0
    assert state.next_idle_ts == 5_600.0


@pytest.mark.asyncio
async def test_zero_result_ends_cycle_without_candidate_or_repeat():
    plugin, state, _ = _plugin()
    cycle = state.idle_judge_cycle

    await plugin._check_idle_greeting(UMO, state, _at(4_600.0), "UTC", 0)
    await plugin._check_idle_greeting(UMO, state, _at(8_000.0), "UTC", 0)

    assert plugin._judge_idle_delay_minutes.await_count == 1
    assert state.idle_judge_checked_cycle == cycle
    assert state.idle_judge_task_ts == 0.0
    assert state.next_idle_ts == 0.0
    plugin._proactive_reply.assert_not_awaited()


@pytest.mark.asyncio
async def test_positive_result_persists_one_candidate_without_repeat():
    plugin, state, _ = _plugin()
    plugin._judge_idle_delay_minutes.return_value = 25

    await plugin._check_idle_greeting(UMO, state, _at(4_600.0), "UTC", 0)
    candidate = state.idle_judge_task_ts
    await plugin._check_idle_greeting(UMO, state, _at(4_601.0), "UTC", 0)

    assert plugin._judge_idle_delay_minutes.await_count == 1
    assert candidate > 0
    assert state.next_idle_ts == candidate
    plugin._proactive_reply.assert_not_awaited()


@pytest.mark.asyncio
async def test_chat_during_judgement_discards_model_result():
    plugin, state, profile = _plugin()

    async def judge_and_receive_message(_umo: str, _tz: str | None) -> int:
        state.last_user_reply_ts = 4_601.0
        plugin._reset_idle_schedule(state, 4_601.0, profile)
        return 20

    plugin._judge_idle_delay_minutes.side_effect = judge_and_receive_message

    await plugin._check_idle_greeting(UMO, state, _at(4_600.0), "UTC", 0)

    assert state.idle_judge_task_ts == 0.0
    assert state.idle_judge_checked_cycle == -1
    assert state.idle_judge_anchor_ts == 4_601.0
    assert state.next_idle_ts == 8_201.0


@pytest.mark.asyncio
async def test_persisted_candidate_fires_once_and_skips_boolean_judge():
    plugin, state, _ = _plugin()
    state.idle_judge_checked_cycle = state.idle_judge_cycle
    state.idle_judge_task_ts = 4_500.0
    state.next_idle_ts = 4_500.0

    await plugin._check_idle_greeting(UMO, state, _at(4_600.0), "UTC", 0)
    await plugin._check_idle_greeting(UMO, state, _at(4_601.0), "UTC", 0)

    assert plugin._proactive_reply.await_count == 1
    assert plugin._proactive_reply.await_args.kwargs["skip_judge"] is True
    assert plugin._proactive_reply.await_args.kwargs["source"] == "silence_greeting"
    assert state.idle_judge_task_ts == 0.0


def test_mode_switch_keeps_config_values_and_rearms_selected_mode():
    plugin, state, profile = _plugin("固定时间")
    assert state.next_idle_ts == 8_200.0

    plugin.cfg["idle_greetings"]["mode"] = "对话热度"
    plugin._initialize_idle_cycle(state, profile, 2_000.0)

    assert plugin.cfg["idle_greetings"]["idle_after_minutes"] == 120
    assert plugin.cfg["idle_greetings"]["hot_delay_minutes"] == 30
    assert state.idle_schedule_mode == "对话热度"
    assert 3_800.0 <= state.next_idle_ts <= 14_000.0


@pytest.mark.asyncio
@pytest.mark.parametrize("ignore_judge", [False, True])
async def test_enhancement_uses_independent_ignore_judge(
    ignore_judge: bool, monkeypatch
):
    plugin, state, profile = _plugin()
    plugin.cfg["enhancement"] = {
        "ignore_judge": ignore_judge,
        "enhancement_prompt_templates": ["补充一句"],
    }
    plugin._enhancement_gen = {UMO: 3}
    plugin._enhancement_tasks = {}
    state.last_user_reply_ts = 2_000.0
    state.last_ai_reply_ts = 2_001.0
    plugin._user_profiles[UMO] = profile

    async def no_wait(_delay: float) -> None:
        return None

    monkeypatch.setattr("data.plugins.astrbot_plugin_Spark.main.asyncio.sleep", no_wait)

    await plugin._delayed_enhancement(UMO, 10.0, 3)

    assert plugin._proactive_reply.await_count == 1
    assert plugin._proactive_reply.await_args.kwargs["skip_judge"] is ignore_judge
    assert (
        plugin._proactive_reply.await_args.kwargs["source"]
        == "conversation_enhancement"
    )


@pytest.mark.asyncio
async def test_enhancement_is_cancelled_when_chat_changes_during_wait(monkeypatch):
    plugin, state, profile = _plugin()
    plugin.cfg["enhancement"] = {
        "ignore_judge": True,
        "enhancement_prompt_templates": ["补充一句"],
    }
    plugin._enhancement_gen = {UMO: 3}
    plugin._enhancement_tasks = {}
    state.last_user_reply_ts = 2_000.0
    state.last_ai_reply_ts = 2_001.0
    plugin._user_profiles[UMO] = profile

    async def receive_message(_delay: float) -> None:
        state.last_user_reply_ts = 2_001.001

    monkeypatch.setattr(
        "data.plugins.astrbot_plugin_Spark.main.asyncio.sleep", receive_message
    )

    await plugin._delayed_enhancement(UMO, 10.0, 3)

    plugin._proactive_reply.assert_not_awaited()


def test_fixed_enhancement_schedule_uses_fixed_seconds_only(monkeypatch):
    plugin, state, _ = _plugin()
    plugin.cfg["enhancement"] = {
        "mode": "固定时间",
        "fixed_delay_seconds": 120,
        "fixed_random_fluctuation_seconds": "+30",
        "enhancement_hot_delay_seconds": 1,
        "enhancement_cold_delay_seconds": 2,
    }
    plugin._enhancement_gen = {UMO: 4}
    plugin._enhancement_tasks = {}
    plugin._delayed_enhancement = MagicMock(return_value=object())
    task = SimpleNamespace()
    monkeypatch.setattr(
        "data.plugins.astrbot_plugin_Spark.main.asyncio.create_task",
        MagicMock(return_value=task),
    )

    plugin._schedule_enhancement(UMO)

    assert plugin._delayed_enhancement.call_args.args[:3] == (UMO, 150.0, 4)
    assert plugin._enhancement_tasks[UMO] is task
    assert state.next_enhancement_ts > 0


def test_heat_enhancement_schedule_uses_heat_endpoints_only(monkeypatch):
    plugin, _, _ = _plugin()
    plugin.cfg["enhancement"] = {
        "mode": "对话热度",
        "fixed_delay_seconds": 9_999,
        "fixed_random_fluctuation_seconds": "+999",
        "enhancement_hot_delay_seconds": 45,
        "enhancement_cold_delay_seconds": 600,
    }
    plugin._enhancement_gen = {UMO: 2}
    plugin._enhancement_tasks = {}
    plugin._calc_heat = MagicMock(return_value=1.0)
    plugin._delayed_enhancement = MagicMock(return_value=object())
    monkeypatch.setattr(
        "data.plugins.astrbot_plugin_Spark.main.asyncio.create_task",
        MagicMock(return_value=SimpleNamespace()),
    )

    plugin._schedule_enhancement(UMO)

    assert plugin._delayed_enhancement.call_args.args[:3] == (UMO, 45.0, 2)


class SaveableConfig(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_calls = 0

    def save_config(self) -> None:
        self.save_calls += 1


def test_config_migration_prefers_new_values_and_fills_legacy_fallbacks():
    plugin = Spark.__new__(Spark)
    plugin.cfg = SaveableConfig(
        {
            "heat_settings": {
                "enable_heat": False,
                "heat_window_minutes": 45,
                "heat_messages_for_full_score": 10,
                "hot_delay_minutes": 10,
                "cold_delay_minutes": 300,
            },
            "idle_greetings": {
                "mode": "大模型判断",
                "hot_delay_minutes": 25,
                "random_fluctuation_minutes": 12,
            },
            "proactive_settings": {
                "proactive_judge_provider": "legacy-judge",
                "proactive_judge_fallback_providers": ["legacy-fallback"],
            },
            "enhancement": {
                "mode": "固定时间",
                "fixed_delay_seconds": 200,
                "enhancement_min_delay": 50,
                "jitter_seconds": 8,
                "ignore_judge": True,
            },
        }
    )

    plugin._migrate_config()

    idle = plugin.cfg["idle_greetings"]
    enhancement = plugin.cfg["enhancement"]
    assert idle["mode"] == "大模型判断"
    assert idle["hot_delay_minutes"] == 25
    assert idle["cold_delay_minutes"] == 300
    assert idle["idle_random_fluctuation_minutes"] == 12
    assert idle["idle_judge_provider"] == "legacy-judge"
    assert idle["idle_judge_fallback_providers"] == ["legacy-fallback"]
    assert idle["_idle_judge_provider_migrated"] is True
    assert enhancement["mode"] == "固定时间"
    assert enhancement["fixed_delay_seconds"] == 200
    assert enhancement["fixed_random_fluctuation_seconds"] == 8
    assert enhancement["ignore_judge"] is True
    assert plugin.cfg.save_calls == 1


def test_config_migration_keeps_explicit_independent_values():
    plugin = Spark.__new__(Spark)
    plugin.cfg = SaveableConfig(
        {
            "idle_greetings": {
                "idle_judge_provider": "new-judge",
                "idle_judge_fallback_providers": ["new-fallback"],
            },
            "proactive_settings": {
                "proactive_judge_provider": "legacy-judge",
                "proactive_judge_fallback_providers": ["legacy-fallback"],
            },
        }
    )

    plugin._migrate_config()

    idle = plugin.cfg["idle_greetings"]
    assert idle["idle_judge_provider"] == "new-judge"
    assert idle["idle_judge_fallback_providers"] == ["new-fallback"]
    assert idle["_idle_judge_provider_migrated"] is True


def test_config_migration_fills_schema_materialized_empty_values_once():
    plugin = Spark.__new__(Spark)
    plugin.cfg = SaveableConfig(
        {
            "idle_greetings": {
                "idle_judge_provider": "",
                "idle_judge_fallback_providers": [],
            },
            "proactive_settings": {
                "proactive_judge_provider": "legacy-judge",
                "proactive_judge_fallback_providers": ["legacy-fallback"],
            },
        }
    )

    plugin._migrate_config()

    idle = plugin.cfg["idle_greetings"]
    assert idle["idle_judge_provider"] == "legacy-judge"
    assert idle["idle_judge_fallback_providers"] == ["legacy-fallback"]
    assert idle["_idle_judge_provider_migrated"] is True


def test_config_migration_does_not_refill_user_cleared_values():
    plugin = Spark.__new__(Spark)
    plugin.cfg = SaveableConfig(
        {
            "idle_greetings": {
                "idle_judge_provider": "",
                "idle_judge_fallback_providers": [],
                "_idle_judge_provider_migrated": True,
            },
            "proactive_settings": {
                "proactive_judge_provider": "changed-judge",
                "proactive_judge_fallback_providers": ["changed-fallback"],
            },
        }
    )

    plugin._migrate_config()

    idle = plugin.cfg["idle_greetings"]
    assert idle["idle_judge_provider"] == ""
    assert idle["idle_judge_fallback_providers"] == []
    first_save_calls = plugin.cfg.save_calls

    plugin.cfg["proactive_settings"] = {
        "proactive_judge_provider": "changed-again",
        "proactive_judge_fallback_providers": ["changed-fallback-again"],
    }
    plugin._migrate_config()

    assert idle["idle_judge_provider"] == ""
    assert idle["idle_judge_fallback_providers"] == []
    assert plugin.cfg.save_calls == first_save_calls


def test_new_user_marker_prevents_future_legacy_provider_copy():
    plugin = Spark.__new__(Spark)
    plugin.cfg = SaveableConfig({"idle_greetings": {}, "proactive_settings": {}})

    plugin._migrate_config()
    assert plugin.cfg["idle_greetings"]["_idle_judge_provider_migrated"] is True

    plugin.cfg["proactive_settings"] = {
        "proactive_judge_provider": "later-judge",
        "proactive_judge_fallback_providers": ["later-fallback"],
    }
    plugin._migrate_config()

    idle = plugin.cfg["idle_greetings"]
    assert idle.get("idle_judge_provider", "") == ""
    assert idle.get("idle_judge_fallback_providers", []) == []


def test_idle_and_boolean_judges_use_independent_provider_chains():
    plugin = Spark.__new__(Spark)
    current = SimpleNamespace(provider_config={"id": "current"})
    providers = {
        provider_id: SimpleNamespace(provider_config={"id": provider_id})
        for provider_id in (
            "normal-judge",
            "normal-fallback",
            "idle-judge",
            "idle-fallback",
        )
    }
    plugin.cfg = {
        "proactive_settings": {
            "proactive_judge_provider": "normal-judge",
            "proactive_judge_fallback_providers": ["normal-fallback"],
        },
        "idle_greetings": {
            "idle_judge_provider": "idle-judge",
            "idle_judge_fallback_providers": ["idle-fallback"],
        },
    }
    plugin.context = SimpleNamespace(
        get_using_provider=lambda umo: current,
        get_provider_by_id=lambda provider_id: providers.get(provider_id),
    )

    normal_chain = plugin._get_judge_providers(UMO)
    idle_chain = plugin._get_judge_providers(UMO, delay_protocol=True)

    assert [plugin._provider_id(provider) for provider in normal_chain] == [
        "normal-judge",
        "normal-fallback",
    ]
    assert [plugin._provider_id(provider) for provider in idle_chain] == [
        "idle-judge",
        "idle-fallback",
    ]


def test_schema_orders_enhancement_before_idle_and_has_mode_conditions():
    schema = json.loads(
        Path(__file__).with_name("_conf_schema.json").read_text(encoding="utf-8")
    )
    keys = list(schema)
    assert keys.index("enhancement") < keys.index("idle_greetings")
    assert "reminders_settings" not in schema

    idle = schema["idle_greetings"]["items"]
    enhancement = schema["enhancement"]["items"]
    daily = schema["daily_prompts"]["items"]["daily_greetings"]["templates"]
    greeting = daily["greeting_slot"]["items"]

    assert idle["idle_after_minutes"]["condition"] == {"mode": "固定时间"}
    assert idle["hot_delay_minutes"]["condition"] == {"mode": "对话热度"}
    assert idle["idle_judge_prompt"]["condition"] == {"mode": "大模型判断"}
    assert enhancement["fixed_delay_seconds"]["condition"] == {"mode": "固定时间"}
    assert enhancement["enhancement_hot_delay_seconds"]["condition"] == {
        "mode": "对话热度"
    }
    assert greeting["time"]["condition"] == {"trigger_source": "固定时间"}
    assert greeting["activity_keywords"]["condition"] == {"trigger_source": "日程活动"}
