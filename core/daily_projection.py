from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from datetime import datetime
from typing import TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class ActivityCandidate:
    activity: str
    occurrence: int
    boundary: datetime
    timeline_index: int


@dataclass(frozen=True)
class ActivityProjectionIssue:
    status: str
    detail: str
    activity: str = ""
    occurrence: int = 0


@dataclass(frozen=True)
class ActivityProjection:
    candidates: list[ActivityCandidate]
    issues: list[ActivityProjectionIssue]


def select_highest_priority(
    items: list[tuple[Hashable, int, T]],
) -> list[T]:
    """Keep every item tied at the highest priority within each group."""
    groups: dict[Hashable, list[tuple[int, T]]] = {}
    for group_key, priority, item in items:
        groups.setdefault(group_key, []).append((priority, item))

    selected = []
    for candidates in groups.values():
        highest = max(priority for priority, _ in candidates)
        selected.extend(item for priority, item in candidates if priority == highest)
    return selected


def project_activity_candidates(
    timeline: list[dict],
    keywords: list[str],
    selected_occurrences: set[int] | None,
    boundary: str,
) -> ActivityProjection:
    matches = [
        (timeline_index, item)
        for timeline_index, item in enumerate(timeline)
        if any(keyword in str(item.get("activity", "")) for keyword in keywords)
    ]
    if not matches:
        return ActivityProjection(
            candidates=[],
            issues=[
                ActivityProjectionIssue(
                    status="not_matched",
                    detail=f"未匹配关键词：{', '.join(keywords)}",
                )
            ],
        )

    candidates = []
    issues = []
    for occurrence, (timeline_index, item) in enumerate(matches, start=1):
        if selected_occurrences is not None and occurrence not in selected_occurrences:
            continue

        activity = str(item.get("activity", ""))
        if not item.get("valid", True):
            issues.append(
                ActivityProjectionIssue(
                    status="invalid_boundary",
                    detail=str(item.get("error", "unknown error")),
                    activity=activity,
                    occurrence=occurrence,
                )
            )
            continue

        boundary_value = item.get(boundary)
        if not isinstance(boundary_value, datetime):
            issues.append(
                ActivityProjectionIssue(
                    status="invalid_boundary",
                    detail=f"缺少{boundary}边界",
                    activity=activity,
                    occurrence=occurrence,
                )
            )
            continue

        candidates.append(
            ActivityCandidate(
                activity=activity,
                occurrence=occurrence,
                boundary=boundary_value,
                timeline_index=timeline_index,
            )
        )

    return ActivityProjection(candidates=candidates, issues=issues)
