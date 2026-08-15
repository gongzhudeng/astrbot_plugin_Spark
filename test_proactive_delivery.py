from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from astrbot.core.message.message_event_result import MessageChain
from data.plugins.astrbot_plugin_Spark.core.proactive_delivery import (
    DEFAULT_DIRECT_DELIVERY_HISTORY,
    AgentDeliveryResult,
    resolve_agent_delivery,
)
from data.plugins.astrbot_plugin_Spark.main import Spark


def test_text_completion_requires_normal_delivery():
    result = resolve_agent_delivery(
        "  晚安  ",
        has_send_operation=True,
        direct_history_text="不应使用",
        direct_delivery_kind="voice",
    )

    assert result is not None
    assert result.response_text == "晚安"
    assert result.history_text == "晚安"
    assert result.already_delivered is False
    assert result.delivery_kind == "text"


def test_direct_voice_delivery_uses_tool_history_text():
    result = resolve_agent_delivery(
        "",
        has_send_operation=True,
        direct_history_text="困死了，晚安。",
        direct_delivery_kind="voice",
    )

    assert result is not None
    assert result.response_text == ""
    assert result.history_text == "困死了，晚安。"
    assert result.already_delivered is True
    assert result.delivery_kind == "voice"


def test_direct_delivery_without_text_uses_safe_history_marker():
    result = resolve_agent_delivery(
        None,
        has_send_operation=True,
    )

    assert result is not None
    assert result.history_text == DEFAULT_DIRECT_DELIVERY_HISTORY
    assert result.already_delivered is True


def test_empty_unsent_agent_result_is_technical_failure():
    assert (
        resolve_agent_delivery(
            "",
            has_send_operation=False,
        )
        is None
    )


def test_delivered_text_is_retained_when_a_tool_directly_sends_media():
    result = resolve_agent_delivery(
        "",
        has_send_operation=True,
        delivered_texts=("我开饭啦，给你拍张看看。",),
    )

    assert result is not None
    assert result.response_text == ""
    assert result.history_text == "我开饭啦，给你拍张看看。"
    assert result.already_delivered is True
    assert result.delivery_kind == "text+tool"


def test_final_text_after_early_text_is_delivered_once_and_kept_in_history():
    result = resolve_agent_delivery(
        "拍好了，你慢慢看。",
        has_send_operation=True,
        delivered_texts=("我开饭啦，给你拍张看看。",),
    )

    assert result is not None
    assert result.response_text == "拍好了，你慢慢看。"
    assert result.history_text == "我开饭啦，给你拍张看看。\n拍好了，你慢慢看。"
    assert result.already_delivered is False


def test_duplicate_final_text_is_not_sent_twice_after_early_delivery():
    result = resolve_agent_delivery(
        "我开饭啦，给你拍张看看。",
        has_send_operation=True,
        delivered_texts=("我开饭啦，给你拍张看看。",),
    )

    assert result is not None
    assert result.response_text == ""
    assert result.history_text == "我开饭啦，给你拍张看看。"
    assert result.already_delivered is True


@pytest.mark.asyncio
async def test_visible_agent_text_is_sent_once_and_ignores_reasoning():
    plugin = Spark.__new__(Spark)
    plugin._send_text = AsyncMock(return_value=True)
    delivered: list[str] = []
    text_response = SimpleNamespace(
        type="llm_result",
        data={"chain": MessageChain().message("我开饭啦，给你拍张看看。")},
    )
    reasoning_response = SimpleNamespace(
        type="llm_result",
        data={"chain": MessageChain(type="reasoning").message("internal")},
    )

    await plugin._send_visible_agent_text(
        "default:FriendMessage:1", text_response, delivered
    )
    await plugin._send_visible_agent_text(
        "default:FriendMessage:1", reasoning_response, delivered
    )
    await plugin._send_visible_agent_text(
        "default:FriendMessage:1", text_response, delivered
    )

    assert delivered == ["我开饭啦，给你拍张看看。"]
    plugin._send_text.assert_awaited_once_with(
        "default:FriendMessage:1", "我开饭啦，给你拍张看看。"
    )


@pytest.mark.asyncio
async def test_agent_response_consumption_sends_text_before_direct_media():
    plugin = Spark.__new__(Spark)
    events: list[str] = []

    async def send_text(_umo: str, text: str) -> bool:
        events.append(f"text:{text}")
        return True

    class Runner:
        async def step_until_done(self, _max_step: int):
            yield SimpleNamespace(
                type="llm_result",
                data={"chain": MessageChain().message("我开饭啦，给你拍张看看。")},
            )
            events.append("media")
            yield SimpleNamespace(type="tool_call_result", data={})
            yield SimpleNamespace(
                type="llm_result",
                data={"chain": MessageChain().message("拍好了，你慢慢看。")},
            )

    plugin._send_text = send_text

    delivered = await plugin._consume_agent_responses(
        "default:FriendMessage:1", Runner()
    )

    assert delivered == ["我开饭啦，给你拍张看看。", "拍好了，你慢慢看。"]
    assert events == [
        "text:我开饭啦，给你拍张看看。",
        "media",
        "text:拍好了，你慢慢看。",
    ]


@pytest.mark.asyncio
async def test_proactive_reply_accepts_direct_voice_without_text_resend(monkeypatch):
    plugin = Spark.__new__(Spark)
    plugin.cfg = {"proactive_settings": {"proactive_judge_enable": False}}
    plugin.context = SimpleNamespace()
    plugin._states = {}
    plugin._get_last_messages = AsyncMock(return_value=("", ""))
    plugin._get_gen_provider = MagicMock(return_value=None)
    plugin._get_gen_persona = AsyncMock(return_value="")
    plugin._run_agent_pipeline = AsyncMock(
        return_value=AgentDeliveryResult(
            response_text="",
            history_text="困死了，晚安。",
            already_delivered=True,
            delivery_kind="voice",
        )
    )
    plugin._send_text = AsyncMock(return_value=True)
    plugin._record_proactive_delivery = MagicMock()
    plugin._debounced_save_session_data = AsyncMock()
    monkeypatch.setattr(
        "data.plugins.astrbot_plugin_Spark.main.HAS_AGENT_PIPELINE",
        True,
    )

    sent = await plugin._proactive_reply(
        "default:FriendMessage:1",
        "UTC",
        "晚安",
        skip_judge=True,
        source="daily_greeting",
    )

    assert sent is True
    plugin._send_text.assert_not_awaited()
    plugin._record_proactive_delivery.assert_called_once()
    assert (
        plugin._record_proactive_delivery.call_args.kwargs["response_text"]
        == "困死了，晚安。"
    )
