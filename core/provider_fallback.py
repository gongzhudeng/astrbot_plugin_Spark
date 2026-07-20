from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TypeVar

PLUGIN_FALLBACK_MODE = "插件独立回退"
ASTRBOT_FALLBACK_MODE = "跟随 AstrBot 对话回退"
T = TypeVar("T")


def normalize_provider_ids(raw_ids: object) -> list[str]:
    """Return non-empty provider IDs in stable order without duplicates."""
    if not isinstance(raw_ids, (list, tuple)):
        return []

    result: list[str] = []
    seen: set[str] = set()
    for raw_id in raw_ids:
        provider_id = str(raw_id or "").strip()
        if not provider_id or provider_id in seen:
            continue
        seen.add(provider_id)
        result.append(provider_id)
    return result


def select_generation_fallback_ids(
    mode: object,
    plugin_ids: object,
    astrbot_ids: object,
) -> list[str]:
    """Select the configured fallback source for proactive generation."""
    source = str(mode or PLUGIN_FALLBACK_MODE).strip()
    raw_ids = astrbot_ids if source == ASTRBOT_FALLBACK_MODE else plugin_ids
    return normalize_provider_ids(raw_ids)


def resolve_provider_chain(
    *,
    primary_id: object,
    fallback_ids: object,
    get_provider: Callable[[str], T | None],
    current_provider: T | None,
) -> tuple[list[T], list[str]]:
    """Resolve a primary/current provider followed by valid configured fallbacks."""
    providers: list[T] = []
    missing_ids: list[str] = []
    normalized_primary_id = str(primary_id or "").strip()
    if normalized_primary_id:
        primary = get_provider(normalized_primary_id)
        if primary is not None:
            providers.append(primary)
        else:
            missing_ids.append(normalized_primary_id)

    if not providers and current_provider is not None:
        providers.append(current_provider)

    for provider_id in normalize_provider_ids(fallback_ids):
        provider = get_provider(provider_id)
        if provider is None:
            missing_ids.append(provider_id)
            continue
        providers.append(provider)

    return dedupe_provider_chain(providers), missing_ids


def dedupe_provider_chain(providers: Iterable[T]) -> list[T]:
    """Deduplicate resolved providers by configured ID, then object identity."""
    result: list[T] = []
    seen_keys: set[tuple[str, object]] = set()
    for provider in providers:
        if provider is None:
            continue
        config = getattr(provider, "provider_config", None)
        provider_id = (
            str(config.get("id") or "").strip() if isinstance(config, dict) else ""
        )
        key = ("id", provider_id) if provider_id else ("object", id(provider))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        result.append(provider)
    return result
