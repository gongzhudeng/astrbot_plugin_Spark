from dataclasses import dataclass
from types import SimpleNamespace

from astrbot.core.cron.events import CronMessageEvent
from astrbot.core.platform.message_session import MessageSession
from astrbot.core.provider.entities import ProviderRequest
from data.plugins.astrbot_plugin_busy_schedule.main import (
    _position_emotion_anchor,
    _replace_prompt_block,
)
from data.plugins.astrbot_plugin_emotion_state.core.injector import (
    ANCHOR,
    BLOCK_END,
    BLOCK_START,
    inject_prompt,
    request_source,
)
from data.plugins.astrbot_plugin_emotion_state.core.models import StateLedger
from data.plugins.astrbot_plugin_Spark.core.message_input import (
    is_slash_prefixed_message,
)
from data.plugins.astrbot_plugin_Spark.main import Spark


@dataclass
class TextComponent:
    text: str


class NonTextComponent:
    pass


class FakeContext:
    async def send_message(self, *_args):
        return None


CACHE_START = "<!-- BUSY_SCHEDULE_CACHE -->"
CACHE_END = "<!-- /BUSY_SCHEDULE_CACHE -->"
CUSTOM_START = "<!-- BUSY_SCHEDULE_CUSTOM -->"
CUSTOM_END = "<!-- /BUSY_SCHEDULE_CUSTOM -->"


def build_prompt_with_emotion() -> str:
    prompt = _replace_prompt_block(
        "persona",
        CACHE_START,
        CACHE_END,
        "<character_static>daily facts</character_static>",
    )
    prompt = _position_emotion_anchor(prompt, CACHE_END)
    prompt = _replace_prompt_block(
        prompt,
        CUSTOM_START,
        CUSTOM_END,
        "<character_custom>dynamic facts</character_custom>",
    )
    return inject_prompt(prompt, StateLedger(user_key="default:FriendMessage:42"))


def test_detects_slash_in_first_non_empty_text_component():
    assert is_slash_prefixed_message([TextComponent("/命令")])
    assert is_slash_prefixed_message([TextComponent("  /命令  ")])
    assert is_slash_prefixed_message(
        [NonTextComponent(), TextComponent(""), TextComponent(" /命令")]
    )


def test_ignores_non_text_components_before_command_text():
    assert is_slash_prefixed_message(
        [NonTextComponent(), NonTextComponent(), TextComponent("/群聊命令")]
    )


def test_rejects_slash_outside_the_first_effective_character():
    assert not is_slash_prefixed_message([TextComponent("普通消息 / 不是命令")])
    assert not is_slash_prefixed_message(
        [TextComponent("普通消息"), TextComponent("/后续文本")]
    )


def test_proactive_state_contract_is_read_only_and_precise():
    plugin = Spark.__new__(Spark)
    plugin._states = {
        "private:pending": SimpleNamespace(
            last_proactive_reply_ts=200.0,
            last_user_reply_ts=100.0,
        ),
        "private:answered": SimpleNamespace(
            last_proactive_reply_ts=200.0,
            last_user_reply_ts=250.0,
        ),
    }

    pending = plugin._get_proactive_state("private:pending")
    answered = plugin._get_proactive_state("private:answered")
    missing = plugin._get_proactive_state("private:missing")

    assert pending["awaiting_user_reply"] is True
    assert answered["awaiting_user_reply"] is False
    assert missing["available"] is False
    assert plugin._states["private:pending"].last_proactive_reply_ts == 200.0


def test_cron_event_keeps_original_session_and_emotion_prompt_order():
    session = MessageSession.from_str("default:FriendMessage:42")
    event = CronMessageEvent(
        context=FakeContext(),
        session=session,
        message="proactive prompt",
    )
    req = ProviderRequest(system_prompt=build_prompt_with_emotion())

    positions = [
        req.system_prompt.index(CACHE_START),
        req.system_prompt.index(CACHE_END),
        req.system_prompt.index(ANCHOR),
        req.system_prompt.index(BLOCK_START),
        req.system_prompt.index(BLOCK_END),
        req.system_prompt.index(CUSTOM_START),
    ]

    assert event.unified_msg_origin == "default:FriendMessage:42"
    assert event.is_private_chat() is True
    assert request_source(event) == "spark_proactive"
    assert positions == sorted(positions)


def test_prompt_diagnostics_exposes_only_markers_and_counts():
    req = ProviderRequest(
        system_prompt=build_prompt_with_emotion(),
        extra_user_content_parts=[SimpleNamespace(text="private memory")],
    )

    diagnostics = Spark._proactive_prompt_diagnostics(req)

    assert diagnostics == {
        "emotion": True,
        "emotion_anchor": True,
        "busy_schedule": True,
        "system_prompt_chars": len(req.system_prompt),
        "extra_user_parts": 1,
    }
    assert "private memory" not in str(diagnostics)


def test_handles_empty_or_non_text_message_chains():
    assert not is_slash_prefixed_message([])
    assert not is_slash_prefixed_message([NonTextComponent(), TextComponent("  ")])
