"""Classify proactive Agent output without coupling delivery to text responses."""

from __future__ import annotations

from dataclasses import dataclass

DIRECT_DELIVERY_TEXT_EXTRA = "spark_direct_delivery_history_text"
DIRECT_DELIVERY_KIND_EXTRA = "spark_direct_delivery_kind"
DEFAULT_DIRECT_DELIVERY_HISTORY = "[通过工具发送了一条消息]"


@dataclass(frozen=True)
class AgentDeliveryResult:
    response_text: str
    history_text: str
    already_delivered: bool = False
    delivery_kind: str = "text"


def resolve_agent_delivery(
    completion_text: object,
    *,
    has_send_operation: bool,
    direct_history_text: object = "",
    direct_delivery_kind: object = "",
) -> AgentDeliveryResult | None:
    """Map Agent output to an explicit delivery result for proactive callers."""
    response_text = str(completion_text or "").strip()
    if response_text:
        return AgentDeliveryResult(
            response_text=response_text,
            history_text=response_text,
        )
    if not has_send_operation:
        return None

    history_text = str(direct_history_text or "").strip()
    delivery_kind = str(direct_delivery_kind or "tool").strip() or "tool"
    return AgentDeliveryResult(
        response_text="",
        history_text=history_text or DEFAULT_DIRECT_DELIVERY_HISTORY,
        already_delivered=True,
        delivery_kind=delivery_kind,
    )
