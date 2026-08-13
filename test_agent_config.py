from core.agent_config import build_main_agent_config


def test_build_main_agent_config_maps_current_astrbot_settings():
    config = build_main_agent_config(
        {
            "timezone": "Asia/Shanghai",
            "kb_agentic_mode": True,
            "subagent_orchestrator": {"main_enable": True},
            "provider_settings": {
                "tool_call_timeout": 500,
                "tool_schema_mode": "full",
                "wake_prefix": "/ai",
                "streaming_response": True,
                "sanitize_context_by_modalities": True,
                "context_limit_reached_strategy": "llm_compress",
                "llm_compress_provider_id": "compressor",
                "max_context_length": 14,
                "dequeue_context_length": 2,
                "llm_safety_mode": False,
                "computer_use_runtime": "sandbox",
                "sandbox": {"booter": "shipyard_neo"},
                "file_extract": {
                    "enable": True,
                    "provider": "moonshotai",
                    "moonshotai_api_key": "test-key",
                },
                "proactive_capability": {"add_cron_tools": False},
                "max_quoted_fallback_images": 8,
            },
        },
        streaming_response=False,
    )

    assert config.tool_call_timeout == 500
    assert config.provider_wake_prefix == "/ai"
    assert config.streaming_response is False
    assert config.sanitize_context_by_modalities is True
    assert config.kb_agentic_mode is True
    assert config.file_extract_enabled is True
    assert config.file_extract_msh_api_key == "test-key"
    assert config.context_limit_reached_strategy == "llm_compress"
    assert config.llm_compress_provider_id == "compressor"
    assert config.max_context_length == 14
    assert config.dequeue_context_length == 2
    assert config.llm_safety_mode is False
    assert config.computer_use_runtime == "sandbox"
    assert config.sandbox_cfg == {"booter": "shipyard_neo"}
    assert config.add_cron_tools is False
    assert config.subagent_orchestrator == {"main_enable": True}
    assert config.timezone == "Asia/Shanghai"
    assert config.max_quoted_fallback_images == 8


def test_build_main_agent_config_preserves_unlimited_context_and_safe_dequeue():
    config = build_main_agent_config(
        {
            "provider_settings": {
                "max_context_length": -1,
                "dequeue_context_length": 0,
            }
        }
    )

    assert config.max_context_length == -1
    assert config.dequeue_context_length == 1
