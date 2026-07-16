from datetime import datetime

from core.daily_projection import project_activity_candidates


def _timeline():
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
            "owner_date": "2026-07-16",
            "activity": "小怡点了一份外卖麻辣烫",
            "period_type": "activity",
            "start": datetime(2026, 7, 16, 11, 23),
            "end": datetime(2026, 7, 16, 11, 51),
            "valid": True,
            "error": "",
        },
        {
            "owner_date": "2026-07-16",
            "activity": "小怡叫了外卖披萨",
            "period_type": "activity",
            "start": datetime(2026, 7, 16, 18, 34),
            "end": datetime(2026, 7, 16, 19, 2),
            "valid": True,
            "error": "",
        },
    ]


def test_real_timeline_matches_wake_and_two_delivery_activities():
    wake = project_activity_candidates(_timeline(), ["醒来"], None, "start")
    deliveries = project_activity_candidates(_timeline(), ["外卖"], None, "start")

    assert [(item.occurrence, item.boundary) for item in wake.candidates] == [
        (1, datetime(2026, 7, 16, 9, 17))
    ]
    assert [item.occurrence for item in deliveries.candidates] == [1, 2]
    assert [item.boundary for item in deliveries.candidates] == [
        datetime(2026, 7, 16, 11, 23),
        datetime(2026, 7, 16, 18, 34),
    ]


def test_occurrence_selection_uses_keyword_match_order():
    projection = project_activity_candidates(_timeline(), ["外卖"], {2}, "start")

    assert len(projection.candidates) == 1
    assert projection.candidates[0].occurrence == 2
    assert "披萨" in projection.candidates[0].activity


def test_invalid_boundary_is_reported_instead_of_scheduled():
    timeline = _timeline()
    timeline[1] = {
        **timeline[1],
        "valid": False,
        "start": None,
        "error": "ordinary activity is missing end_time",
    }

    projection = project_activity_candidates(timeline, ["外卖"], None, "start")

    assert [item.occurrence for item in projection.candidates] == [2]
    assert len(projection.issues) == 1
    assert projection.issues[0].status == "invalid_boundary"
    assert projection.issues[0].occurrence == 1


def test_missing_keyword_match_is_explicit():
    projection = project_activity_candidates(_timeline(), ["出门"], None, "start")

    assert projection.candidates == []
    assert projection.issues[0].status == "not_matched"
