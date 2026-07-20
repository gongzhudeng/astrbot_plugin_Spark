from dataclasses import dataclass

from core.provider_fallback import (
    ASTRBOT_FALLBACK_MODE,
    PLUGIN_FALLBACK_MODE,
    normalize_provider_ids,
    resolve_provider_chain,
    select_generation_fallback_ids,
)


@dataclass
class FakeProvider:
    provider_id: str

    @property
    def provider_config(self):
        return {"id": self.provider_id}


def test_normalize_provider_ids_keeps_order_and_removes_duplicates():
    assert normalize_provider_ids(["first", "", " first ", None, "second"]) == [
        "first",
        "second",
    ]


def test_generation_plugin_mode_uses_plugin_fallbacks():
    assert select_generation_fallback_ids(
        PLUGIN_FALLBACK_MODE,
        ["plugin-1", "plugin-2"],
        ["astrbot-1"],
    ) == ["plugin-1", "plugin-2"]


def test_generation_astrbot_mode_uses_conversation_fallbacks():
    assert select_generation_fallback_ids(
        ASTRBOT_FALLBACK_MODE,
        ["plugin-1"],
        ["astrbot-1", "astrbot-2"],
    ) == ["astrbot-1", "astrbot-2"]


def test_resolve_chain_uses_current_model_when_primary_is_empty():
    current = FakeProvider("current")
    fallback = FakeProvider("fallback")
    providers = {"fallback": fallback}

    chain, missing = resolve_provider_chain(
        primary_id="",
        fallback_ids=["fallback"],
        get_provider=providers.get,
        current_provider=current,
    )

    assert [provider.provider_id for provider in chain] == ["current", "fallback"]
    assert missing == []


def test_resolve_chain_falls_back_to_current_and_skips_invalid_or_duplicate_ids():
    current = FakeProvider("current")
    fallback = FakeProvider("fallback")
    providers = {"current": current, "fallback": fallback}

    chain, missing = resolve_provider_chain(
        primary_id="deleted-primary",
        fallback_ids=["missing", "current", "fallback", "fallback"],
        get_provider=providers.get,
        current_provider=current,
    )

    assert [provider.provider_id for provider in chain] == ["current", "fallback"]
    assert missing == ["deleted-primary", "missing"]
