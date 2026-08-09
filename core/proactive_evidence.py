"""Bounded, privacy-preserving evidence for natural proactive deliveries."""

from __future__ import annotations

import re
from hashlib import sha256
from typing import Any
from uuid import uuid4

EVIDENCE_SCHEMA_VERSION = 1
DEFAULT_MAX_RECORDS = 16
DEFAULT_SUMMARY_CHARS = 160

_BLOCK_PATTERNS = (
    re.compile(
        r"<!--\s*astrbot-chat-merger:image-context(?::[^>]*)?-->.*?"
        r"<!--\s*/?astrbot-chat-merger:image-context(?::[^>]*)?-->",
        re.I | re.S,
    ),
    re.compile(
        r"<!--\s*astrbot-chat-merger:image-context(?::[^>]*)?-->.*$",
        re.I | re.S,
    ),
    re.compile(r"<image_context\b[^>]*>.*?</image_context>", re.I | re.S),
    re.compile(r"<image_context\b[^>]*>.*$", re.I | re.S),
    re.compile(
        r"<(?:system|emotion_state|character_static|character_custom|"
        r"proactive_request_context)\b[^>]*>.*?</(?:system|emotion_state|"
        r"character_static|character_custom|proactive_request_context)>",
        re.I | re.S,
    ),
    re.compile(r"<!--.*?-->", re.S),
    re.compile(r"\[图片上下文[^\]]*\]", re.S),
)
_MEDIA_MARKER = re.compile(
    r"\[(?:(?:视频|图片|语音)(?:消息)?|文件(?:消息|:[^\]]*)?)\]", re.I
)
_PRIVATE_PATTERNS = (
    re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"),
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    re.compile(r"https?://[^\s]+", re.I),
    re.compile(
        r"(?i)\b(?:bearer\s+|api[_-]?key\s*[:=]\s*|token\s*[:=]\s*)"
        r"[A-Za-z0-9._~+/-]{8,}"
    ),
)


def sanitize_evidence_summary(
    value: object, max_chars: int = DEFAULT_SUMMARY_CHARS
) -> str:
    """Return a short user-readable fact without generated or obvious private data."""
    raw = str(value or "")
    had_media = bool(_MEDIA_MARKER.search(raw) or "image_context" in raw.lower())
    clean = raw
    for pattern in _BLOCK_PATTERNS:
        clean = pattern.sub(" ", clean)
    clean = _MEDIA_MARKER.sub(" ", clean)
    for pattern in _PRIVATE_PATTERNS:
        clean = pattern.sub("[已隐去]", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    if not clean and had_media:
        clean = "媒体消息"
    return clean[: max(40, int(max_chars))]


def normalize_evidence_records(
    records: object,
    *,
    max_records: int = DEFAULT_MAX_RECORDS,
    summary_chars: int = DEFAULT_SUMMARY_CHARS,
) -> list[dict[str, Any]]:
    """Validate persisted evidence and return a bounded canonical snapshot."""
    if not isinstance(records, list):
        return []
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in records:
        if not isinstance(item, dict):
            continue
        evidence_id = str(item.get("evidence_id") or "").strip()[:96]
        if not evidence_id or evidence_id in seen:
            continue
        try:
            sent_at = max(0.0, float(item.get("sent_at", 0.0) or 0.0))
            reply_at = max(0.0, float(item.get("first_reply_at", 0.0) or 0.0))
        except (TypeError, ValueError):
            continue
        if sent_at <= 0.0:
            continue
        source = re.sub(
            r"[^a-z0-9_.:-]+", "_", str(item.get("source") or "unknown").lower()
        )
        source = source.strip("_")[:48] or "unknown"
        reply_summary = sanitize_evidence_summary(
            item.get("first_reply_summary", ""), summary_chars
        )
        replied = reply_at >= sent_at
        normalized.append(
            {
                "evidence_id": evidence_id,
                "source": source,
                "sent_at": sent_at,
                "proactive_summary": sanitize_evidence_summary(
                    item.get("proactive_summary", ""), summary_chars
                ),
                "reply_status": "replied" if replied else "pending",
                "first_reply_at": reply_at if replied else 0.0,
                "first_reply_summary": reply_summary if replied else "",
            }
        )
        seen.add(evidence_id)
    normalized.sort(key=lambda item: (item["sent_at"], item["evidence_id"]))
    return normalized[-max(1, int(max_records)) :]


def record_proactive_delivery(
    records: object,
    *,
    source: str,
    sent_at: float,
    proactive_text: object,
    evidence_id: str = "",
    max_records: int = DEFAULT_MAX_RECORDS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Append one record only after the caller has proven successful delivery."""
    current = normalize_evidence_records(records, max_records=max_records)
    normalized_source = re.sub(r"[^a-z0-9_.:-]+", "_", str(source or "unknown").lower())
    normalized_source = normalized_source.strip("_")[:48] or "unknown"
    timestamp = max(0.0, float(sent_at))
    summary = sanitize_evidence_summary(proactive_text)
    stable_id = str(evidence_id or "").strip()[:96]
    if not stable_id:
        nonce = uuid4().hex
        digest = sha256(
            f"{normalized_source}:{timestamp:.6f}:{summary}:{nonce}".encode()
        ).hexdigest()[:24]
        stable_id = f"spark-{digest}"
    record = {
        "evidence_id": stable_id,
        "source": normalized_source,
        "sent_at": timestamp,
        "proactive_summary": summary,
        "reply_status": "pending",
        "first_reply_at": 0.0,
        "first_reply_summary": "",
    }
    current = [item for item in current if item["evidence_id"] != stable_id]
    current.append(record)
    return normalize_evidence_records(current, max_records=max_records), dict(record)


def acknowledge_pending_evidence(
    records: object,
    *,
    reply_at: float,
    reply_text: object,
    max_records: int = DEFAULT_MAX_RECORDS,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Attach the first real user reply to every earlier pending delivery."""
    current = normalize_evidence_records(records, max_records=max_records)
    timestamp = max(0.0, float(reply_at))
    summary = sanitize_evidence_summary(reply_text)
    acknowledged: list[str] = []
    updated: list[dict[str, Any]] = []
    for item in current:
        copy = dict(item)
        if copy["reply_status"] == "pending" and timestamp >= copy["sent_at"]:
            copy.update(
                reply_status="replied",
                first_reply_at=timestamp,
                first_reply_summary=summary,
            )
            acknowledged.append(copy["evidence_id"])
        updated.append(copy)
    return normalize_evidence_records(updated, max_records=max_records), acknowledged
