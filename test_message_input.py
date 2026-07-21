from dataclasses import dataclass

from core.message_input import is_slash_prefixed_message


@dataclass
class TextComponent:
    text: str


class NonTextComponent:
    pass


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


def test_handles_empty_or_non_text_message_chains():
    assert not is_slash_prefixed_message([])
    assert not is_slash_prefixed_message([NonTextComponent(), TextComponent("  ")])
