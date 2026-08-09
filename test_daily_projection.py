from datetime import datetime

from core.daily_projection import (
    project_activity_candidates,
    select_highest_priority,
)


def _timeline():
    """Timeline using new DSL tags as activity keywords."""
    return [
        {
            "owner_date": "2026-07-16",
            "activity": "小怡醒来，赖在床上玩手机",
            "period_type": "activity",
            "start": datetime(2026, 7, 16, 9, 17),
            "end": datetime(2026, 7, 16, 9, 45),
            "valid": True,
            "error": "",
        },
        {
            # 外卖午餐：带 DSL 标签
            "owner_date": "2026-07-16",
            "activity": "小怡点了一份外卖麻辣烫【用餐】",
            "period_type": "activity",
            "start": datetime(2026, 7, 16, 11, 23),
            "end": datetime(2026, 7, 16, 11, 51),
            "valid": True,
            "error": "",
        },
        {
            # 出门堂食晚餐：外出+用餐叠加
            "owner_date": "2026-07-16",
            "activity": "小怡去餐厅吃晚饭【外出】【用餐】",
            "period_type": "activity",
            "start": datetime(2026, 7, 16, 18, 34),
            "end": datetime(2026, 7, 16, 19, 2),
            "valid": True,
            "error": "",
        },
        {
            # 主动分享：看电影
            "owner_date": "2026-07-16",
            "activity": "小怡在家看了一部很有感触的电影【主动分享】",
            "period_type": "activity",
            "start": datetime(2026, 7, 16, 20, 0),
            "end": datetime(2026, 7, 16, 22, 15),
            "valid": True,
            "error": "",
        },
        {
            "owner_date": "2026-07-16",
            "activity": "小怡睡觉",
            "period_type": "sleep",
            "start": datetime(2026, 7, 17, 1, 32),
            "end": None,
            "valid": True,
            "error": "",
        },
    ]


# ─── Legacy tests (adapted to new timeline) ────────────────────────────────────


def test_real_timeline_matches_wake():
    wake = project_activity_candidates(_timeline(), ["醒来"], "start")
    assert [(item.occurrence, item.boundary) for item in wake.candidates] == [
        (1, datetime(2026, 7, 16, 9, 17))
    ]


def test_dsl_tag_外出_matches_all_outdoor_activities():
    """【外出】 tag reliably matches any activity with that tag, regardless of sentence structure."""
    projection = project_activity_candidates(_timeline(), ["【外出】"], "start")
    assert len(projection.candidates) == 1
    assert "餐厅" in projection.candidates[0].activity


def test_dsl_tag_用餐_matches_takeaway_and_dine_in():
    """【用餐】 matches both 外卖 and 堂食 activities."""
    projection = project_activity_candidates(_timeline(), ["【用餐】"], "start")
    activities = [c.activity for c in projection.candidates]
    assert any("麻辣烫" in a for a in activities), "should match takeaway"
    assert any("餐厅" in a for a in activities), "should match dine-in"
    assert len(projection.candidates) == 2


def test_dsl_tag_主动分享_matches_sharing_activity():
    projection = project_activity_candidates(_timeline(), ["【主动分享】"], "start")
    assert len(projection.candidates) == 1
    assert "电影" in projection.candidates[0].activity


def test_one_activity_projected_by_both_外出_and_用餐_slots():
    """The dine-out entry can be matched by both 外出 and 用餐 keywords independently."""
    outdoor = project_activity_candidates(_timeline(), ["【外出】"], "start")
    meal = project_activity_candidates(_timeline(), ["【用餐】"], "start")
    dine_out = "小怡去餐厅吃晚饭【外出】【用餐】"
    outdoor_candidate = next(c for c in outdoor.candidates if c.activity == dine_out)
    meal_candidate = next(c for c in meal.candidates if c.activity == dine_out)

    assert outdoor_candidate.timeline_index == meal_candidate.timeline_index == 2
    assert outdoor_candidate.occurrence == 1
    assert meal_candidate.occurrence == 2


def test_identical_activity_text_keeps_distinct_timeline_identity():
    timeline = [
        {
            "activity": "小怡吃饭【用餐】",
            "start": datetime(2026, 7, 16, 12, 0),
            "valid": True,
        },
        {
            "activity": "小怡吃饭【用餐】",
            "start": datetime(2026, 7, 16, 18, 0),
            "valid": True,
        },
    ]

    projection = project_activity_candidates(timeline, ["【用餐】"], "start")

    assert [candidate.timeline_index for candidate in projection.candidates] == [0, 1]


def test_priority_selection_keeps_all_tied_highest_items():
    selected = select_highest_priority(
        [
            ((datetime(2026, 7, 16), 2), 3, "outdoor"),
            ((datetime(2026, 7, 16), 2), 3, "meal"),
            ((datetime(2026, 7, 16), 2), 1, "fallback"),
            ((datetime(2026, 7, 16), 3), 0, "another activity"),
        ]
    )

    assert selected == ["outdoor", "meal", "another activity"]


def test_wake_and_sleep_keywords_still_stable():
    """醒来 and 睡觉 match without relying on DSL tags."""
    wake = project_activity_candidates(_timeline(), ["醒来"], "start")
    sleep = project_activity_candidates(_timeline(), ["睡觉"], "start")
    assert wake.candidates[0].boundary == datetime(2026, 7, 16, 9, 17)
    assert sleep.candidates[0].boundary == datetime(2026, 7, 17, 1, 32)


def test_negative_sentence_no_false_positive():
    """Natural-language phrases like '没点外卖' or '不出门' must not be matched by DSL tags."""
    negative_timeline = [
        {
            "owner_date": "2026-07-16",
            "activity": "小怡今天懒得出门，也没点外卖，就吃了昨天剩的饭",
            "period_type": "activity",
            "start": datetime(2026, 7, 16, 12, 0),
            "end": datetime(2026, 7, 16, 12, 30),
            "valid": True,
            "error": "",
        }
    ]
    outdoor = project_activity_candidates(negative_timeline, ["【外出】"], "start")
    meal = project_activity_candidates(negative_timeline, ["【用餐】"], "start")
    assert outdoor.candidates == [], "no 【外出】 tag → should not match"
    assert meal.candidates == [], "no 【用餐】 tag → should not match"


def test_all_keyword_matches_are_projected_for_interval_filtering():
    meal_proj = project_activity_candidates(_timeline(), ["【用餐】"], "start")
    assert [item.occurrence for item in meal_proj.candidates] == [1, 2]
    assert "麻辣烫" in meal_proj.candidates[0].activity
    assert "餐厅" in meal_proj.candidates[1].activity


def test_invalid_boundary_is_reported_instead_of_scheduled():
    timeline = _timeline()
    timeline[1] = {
        **timeline[1],
        "valid": False,
        "start": None,
        "error": "ordinary activity is missing end_time",
    }

    projection = project_activity_candidates(timeline, ["【用餐】"], "start")

    assert [item.occurrence for item in projection.candidates] == [2]
    assert len(projection.issues) == 1
    assert projection.issues[0].status == "invalid_boundary"
    assert projection.issues[0].occurrence == 1


def test_missing_keyword_match_is_explicit():
    projection = project_activity_candidates(_timeline(), ["【不存在标签】"], "start")
    assert projection.candidates == []
    assert projection.issues[0].status == "not_matched"
