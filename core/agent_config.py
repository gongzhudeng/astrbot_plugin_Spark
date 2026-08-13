from __future__ import annotations

from collections.abc import Mapping
from typing import Any

try:
    from astrbot.core.astr_main_agent import (
        build_main_agent_config as _framework_build_main_agent_config,
    )
except ImportError:
    from astrbot.core.astr_main_agent import (
        MainAgentBuildConfig as _MainAgentBuildConfig,
    )

    _framework_build_main_agent_config = None
else:
    _MainAgentBuildConfig = None


def _settings(config: Mapping[str, Any]) -> dict[str, Any]:
    value = config.get("provider_settings", {})
    return dict(value) if isinstance(value, Mapping) else {}


def _bounded_dequeue_length(
    settings: Mapping[str, Any], max_context_length: int
) -> int:
    raw_value = settings.get("dequeue_context_length", 10)
    dequeue_length = raw_value if isinstance(raw_value, int) else 10
    if max_context_length > 1:
        return min(max(1, dequeue_length), max_context_length - 1)
    return 1


def build_main_agent_config(
    astrbot_config: Mapping[str, Any],
    *,
    timezone: str | None = None,
    streaming_response: bool | None = None,
) -> Any:
    if _framework_build_main_agent_config is not None:
        return _framework_build_main_agent_config(
            astrbot_config,
            timezone=timezone,
            streaming_response=streaming_response,
        )

    assert _MainAgentBuildConfig is not None
    settings = _settings(astrbot_config)
    file_extract = settings.get("file_extract", {})
    if not isinstance(file_extract, Mapping):
        file_extract = {}
    proactive = settings.get("proactive_capability", {})
    if not isinstance(proactive, Mapping):
        proactive = {}

    raw_max_context_length = settings.get("max_context_length", 50)
    max_context_length = (
        raw_max_context_length if isinstance(raw_max_context_length, int) else 50
    )

    if streaming_response is None:
        streaming_response = bool(settings.get("streaming_response", True))

    runtime = settings.get("computer_use_runtime", "local")
    if not isinstance(runtime, str) or runtime not in {"none", "local", "sandbox"}:
        runtime = "local"

    return _MainAgentBuildConfig(
        tool_call_timeout=settings.get("tool_call_timeout", 60),
        tool_schema_mode=settings.get("tool_schema_mode", "full"),
        provider_wake_prefix=settings.get("wake_prefix", ""),
        streaming_response=streaming_response,
        sanitize_context_by_modalities=bool(
            settings.get("sanitize_context_by_modalities", False)
        ),
        kb_agentic_mode=bool(astrbot_config.get("kb_agentic_mode", False)),
        file_extract_enabled=bool(file_extract.get("enable", False)),
        file_extract_prov=file_extract.get("provider", "moonshotai"),
        file_extract_msh_api_key=file_extract.get("moonshotai_api_key", ""),
        context_limit_reached_strategy=settings.get(
            "context_limit_reached_strategy", "truncate_by_turns"
        ),
        llm_compress_instruction=settings.get("llm_compress_instruction", ""),
        llm_compress_keep_recent_ratio=settings.get(
            "llm_compress_keep_recent_ratio", 0.15
        ),
        llm_compress_provider_id=settings.get("llm_compress_provider_id", ""),
        max_context_length=max_context_length,
        dequeue_context_length=_bounded_dequeue_length(settings, max_context_length),
        fallback_max_context_tokens=settings.get("fallback_max_context_tokens", 128000),
        llm_safety_mode=bool(settings.get("llm_safety_mode", True)),
        safety_mode_strategy=settings.get("safety_mode_strategy", "system_prompt"),
        computer_use_runtime=runtime,
        sandbox_cfg=dict(settings.get("sandbox", {}))
        if isinstance(settings.get("sandbox"), Mapping)
        else {},
        add_cron_tools=bool(proactive.get("add_cron_tools", True)),
        provider_settings=settings,
        subagent_orchestrator=dict(astrbot_config.get("subagent_orchestrator", {}))
        if isinstance(astrbot_config.get("subagent_orchestrator"), Mapping)
        else {},
        timezone=timezone or astrbot_config.get("timezone"),
        max_quoted_fallback_images=settings.get("max_quoted_fallback_images", 20),
    )
