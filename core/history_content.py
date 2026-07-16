from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

_DATETIME_REMINDER_RE = re.compile(
    r"^<system_reminder>(?:[^\n]*\n)*Current datetime: "
    r"\d{4}-\d{2}-\d{2} \d{2}:\d{2} \([^)]+\)"
    r"(?:\n[^\n]*)*</system_reminder>$"
)


def text_from_part(part: Any) -> str:
    if isinstance(part, str):
        return part
    if isinstance(part, dict):
        return str(part.get("text") or part.get("content") or part.get("value") or "")
    return str(getattr(part, "text", "") or getattr(part, "content", "") or "")


def is_datetime_reminder(text: str) -> bool:
    return bool(_DATETIME_REMINDER_RE.fullmatch(str(text or "").strip()))


def extract_history_text(content: Any, *, exclude_datetime: bool = False) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for part in content:
            text = text_from_part(part).strip()
            if text and not (exclude_datetime and is_datetime_reminder(text)):
                parts.append(text)
        return " ".join(parts).strip()
    return text_from_part(content).strip()


def find_datetime_reminder(parts: Iterable[Any]) -> str:
    for part in parts:
        text = text_from_part(part).strip()
        if is_datetime_reminder(text):
            return text
    return ""


def build_user_content_with_datetime(
    text: str,
    reminder: str = "",
) -> str | list[dict]:
    if not reminder:
        return text
    return [
        {"type": "text", "text": text},
        {"type": "text", "text": reminder},
    ]


def build_proactive_user_content(
    placeholder: str, reminder: str = ""
) -> str | list[dict]:
    return build_user_content_with_datetime(placeholder, reminder)
