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
from data.plugins.astrbot_plugin_thinking_cleaner.core.sanitizer import (
    clean_thinking_text,
)


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
async def test_visible_agent_text_is_cleaned_before_delivery():
    plugin = Spark.__new__(Spark)
    plugin.context = SimpleNamespace(_thinking_cleaner_clean_text=clean_thinking_text)
    plugin._send_text = AsyncMock(return_value=True)
    delivered: list[str] = []
    response = SimpleNamespace(
        type="llm_result",
        data={
            "chain": MessageChain().message(
                "<tag>\n豆包工具轮次推理</thinking>\n给你的回复"
            )
        },
    )

    await plugin._send_visible_agent_text(
        "default:FriendMessage:1", response, delivered
    )

    assert delivered == ["给你的回复"]
    plugin._send_text.assert_awaited_once_with("default:FriendMessage:1", "给你的回复")


@pytest.mark.asyncio
async def test_proactive_history_is_cleaned_before_persistence():
    plugin = Spark.__new__(Spark)
    conversation = SimpleNamespace(history=[])
    conversation_manager = SimpleNamespace(
        get_conversation=AsyncMock(return_value=conversation),
        update_conversation=AsyncMock(),
    )
    plugin.context = SimpleNamespace(
        _thinking_cleaner_clean_text=clean_thinking_text,
        conversation_manager=conversation_manager,
    )
    plugin._proactive_placeholder = MagicMock(return_value="[主动轮]")
    plugin._remove_internal_history_tail = AsyncMock(return_value=0)

    await plugin._save_standard_proactive_history(
        "default:FriendMessage:1",
        "conversation",
        "<tag>\n豆包最终轮次推理</thinking>\n给你的回复",
        0,
    )

    persisted = conversation_manager.update_conversation.await_args.kwargs["history"]
    assert persisted[-1] == {"role": "assistant", "content": "给你的回复"}


def test_final_delivery_uses_cleaned_response_and_history_text():
    plugin = Spark.__new__(Spark)
    plugin.context = SimpleNamespace(_thinking_cleaner_clean_text=clean_thinking_text)
    source = "<tag>\n豆包最终轮次推理</thinking>\n给你的回复"

    delivery = resolve_agent_delivery(
        plugin._clean_output_text(source),
        has_send_operation=False,
    )

    assert delivery is not None
    assert delivery.response_text == "给你的回复"
    assert delivery.history_text == "给你的回复"


def test_missing_cleaner_keeps_spark_output_unchanged():
    plugin = Spark.__new__(Spark)
    plugin.context = SimpleNamespace()
    source = "<tag>\n豆包内部推理</thinking>\n给你的回复"

    assert plugin._clean_output_text(source) == source


@pytest.mark.asyncio
async def test_legacy_generation_returns_cleaned_text(monkeypatch):
    plugin = Spark.__new__(Spark)
    provider = SimpleNamespace(
        provider_config={"id": "doubao"},
        text_chat=AsyncMock(
            return_value=SimpleNamespace(
                completion_text=("<tag>\n豆包旧路径推理</thinking>\n给你的回复")
            )
        ),
    )
    plugin.context = SimpleNamespace(_thinking_cleaner_clean_text=clean_thinking_text)
    plugin._get_gen_providers = MagicMock(return_value=[provider])
    plugin._get_gen_persona = AsyncMock(return_value="")
    plugin._get_conversation_contexts = AsyncMock(return_value=[])
    plugin._format_context_tail_for_log = MagicMock(return_value="")
    monkeypatch.setattr(
        "data.plugins.astrbot_plugin_Spark.main.HAS_REQUEST_HOOKS",
        False,
    )

    result = await plugin._run_legacy_llm(
        "default:FriendMessage:1",
        "继续聊天",
    )

    assert result == "给你的回复"


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
