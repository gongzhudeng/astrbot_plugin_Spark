from data.plugins.astrbot_plugin_Spark.core.proactive_evidence import (
    acknowledge_pending_evidence,
    normalize_evidence_records,
    record_proactive_delivery,
    sanitize_evidence_summary,
)
from data.plugins.astrbot_plugin_Spark.main import SessionState, Spark


def test_sanitizes_and_bounds_proactive_evidence():
    raw = (
        "你好 13800138000 user@example.com https://example.com/private?q=1 "
        "<image_context>generated private image description</image_context> "
        "<!-- hidden system prompt -->"
    )

    summary = sanitize_evidence_summary(raw, max_chars=80)

    assert "13800138000" not in summary
    assert "user@example.com" not in summary
    assert "example.com" not in summary
    assert "generated private image" not in summary
    assert "hidden system" not in summary
    assert len(summary) <= 80


def test_records_deliveries_and_first_reply_without_raw_history():
    records, first = record_proactive_delivery(
        [],
        source="daily_greeting",
        sent_at=100.0,
        proactive_text="早上好，今天也要照顾好自己。",
        evidence_id="delivery-1",
    )
    records, second = record_proactive_delivery(
        records,
        source="conversation_enhancement",
        sent_at=110.0,
        proactive_text="刚才的话题，我还想再陪你聊一会儿。",
        evidence_id="delivery-2",
    )

    acknowledged, ids = acknowledge_pending_evidence(
        records,
        reply_at=120.0,
        reply_text="我回来了，电话是 13800138000。",
    )

    assert first["source"] == "daily_greeting"
    assert second["source"] == "conversation_enhancement"
    assert ids == ["delivery-1", "delivery-2"]
    assert all(item["reply_status"] == "replied" for item in acknowledged)
    assert all(item["first_reply_at"] == 120.0 for item in acknowledged)
    assert all(
        "13800138000" not in item["first_reply_summary"] for item in acknowledged
    )


def test_evidence_is_bounded_and_survives_session_round_trip():
    records = []
    for index in range(20):
        records, _ = record_proactive_delivery(
            records,
            source="silence_greeting",
            sent_at=100.0 + index,
            proactive_text=f"第 {index} 条问候",
            evidence_id=f"delivery-{index}",
            max_records=4,
        )

    state = SessionState(
        last_proactive_reply_ts=119.0,
        last_user_reply_ts=90.0,
        proactive_evidence=records,
    )
    restored = SessionState.from_dict(state.to_dict())

    assert [item["evidence_id"] for item in restored.proactive_evidence] == [
        "delivery-16",
        "delivery-17",
        "delivery-18",
        "delivery-19",
    ]


def test_spark_snapshot_returns_detached_structured_evidence():
    plugin = Spark.__new__(Spark)
    state = SessionState(last_proactive_reply_ts=200.0, last_user_reply_ts=100.0)
    state.proactive_evidence, _ = record_proactive_delivery(
        [],
        source="daily_greeting",
        sent_at=200.0,
        proactive_text="早上好",
        evidence_id="delivery-1",
    )
    plugin._states = {"private:1": state}

    snapshot = plugin._get_proactive_state("private:1")
    snapshot["evidence"][0]["source"] = "mutated"

    assert snapshot["schema_version"] == 1
    assert snapshot["awaiting_user_reply"] is True
    assert state.proactive_evidence[0]["source"] == "daily_greeting"


def test_normalizer_rejects_duplicate_or_malformed_records():
    normalized = normalize_evidence_records(
        [
            {"evidence_id": "same", "source": "daily_greeting", "sent_at": 10},
            {"evidence_id": "same", "source": "other", "sent_at": 11},
            {"evidence_id": "missing-time", "source": "other"},
            "invalid",
        ]
    )

    assert len(normalized) == 1
    assert normalized[0]["evidence_id"] == "same"
