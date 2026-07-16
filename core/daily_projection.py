from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ActivityCandidate:
    activity: str
    occurrence: int
    boundary: datetime


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


def project_activity_candidates(
    timeline: list[dict],
    keywords: list[str],
    selected_occurrences: set[int] | None,
    boundary: str,
) -> ActivityProjection:
    matches = [
        item
        for item in timeline
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
    for occurrence, item in enumerate(matches, start=1):
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
            )
        )

    return ActivityProjection(candidates=candidates, issues=issues)
