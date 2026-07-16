from core.history_content import (
    build_proactive_user_content,
    build_user_content_with_datetime,
    extract_history_text,
    find_datetime_reminder,
    is_datetime_reminder,
)

REMINDER = "<system_reminder>Current datetime: 2026-07-16 21:30 (CST)</system_reminder>"


def test_semantic_projection_excludes_structured_datetime_part():
    content = [
        {"type": "text", "text": "今天去吃火锅"},
        {"type": "text", "text": REMINDER},
    ]

    assert extract_history_text(content) == f"今天去吃火锅 {REMINDER}"
    assert extract_history_text(content, exclude_datetime=True) == "今天去吃火锅"


def test_plain_user_text_is_not_removed():
    text = f"请解释这段文字：{REMINDER}"

    assert extract_history_text(text, exclude_datetime=True) == text
    assert not is_datetime_reminder(text)


def test_model_context_keeps_normal_message_shape():
    content = build_user_content_with_datetime("今天去吃火锅", REMINDER)

    assert content == [
        {"type": "text", "text": "今天去吃火锅"},
        {"type": "text", "text": REMINDER},
    ]
    assert extract_history_text(content) == f"今天去吃火锅 {REMINDER}"
    assert extract_history_text(content, exclude_datetime=True) == "今天去吃火锅"


def test_proactive_history_uses_normal_message_shape():
    content = build_proactive_user_content("[主动轮]", REMINDER)

    assert content == [
        {"type": "text", "text": "[主动轮]"},
        {"type": "text", "text": REMINDER},
    ]
    assert find_datetime_reminder(content) == REMINDER
