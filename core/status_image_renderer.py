"""Glassmorphism Pillow renderer for the proactive status command (S2b)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .style_kit import Canvas, c, font


@dataclass(frozen=True)
class ProactiveStatusImageData:
    """Structured values displayed by the proactive status image."""

    subscription: str
    subscribed: bool
    is_busy: bool
    judge_enabled: bool
    heat_label: str
    heat_value: float | None
    facts: tuple[tuple[str, str], ...]
    daily_items: tuple[str, ...]
    pending_items: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "facts",
            tuple((str(k), str(v)) for k, v in self.facts),
        )
        object.__setattr__(self, "daily_items", tuple(str(i) for i in self.daily_items))
        object.__setattr__(
            self, "pending_items", tuple(str(i) for i in self.pending_items)
        )


def _split_status(item: str) -> tuple[str, str | None]:
    """Split a trailing `` · status`` tag off a daily item when present."""
    if " · " in item:
        main, status = item.rsplit(" · ", 1)
        return main, status
    return item, None


def _status_kind(status: str | None) -> str:
    if status is None:
        return "none"
    if "已发送" in status:
        return "sent"
    if "跳过" in status:
        return "skipped"
    if "待触发" in status:
        return "waiting"
    return "other"


class ProactiveStatusImageRenderer:
    """Render day and night proactive-status images without a browser."""

    width = 1080

    def __init__(self, plugin_dir: Path):
        """Initialize the renderer.

        Args:
            plugin_dir: Plugin root containing the replaceable ``logo.png``.
        """
        self.plugin_dir = Path(plugin_dir)

    def resolve_theme(self, mode: object, now: datetime) -> str:
        """Resolve a configured mode to ``day`` or ``night``.

        Args:
            mode: Configured Chinese or English theme mode.
            now: Current time in the plugin timezone.

        Returns:
            Concrete theme name.
        """
        normalized = str(mode or "").strip().casefold()
        if normalized in {"白天模式", "day", "light"}:
            return "day"
        if normalized in {"夜间模式", "night", "dark"}:
            return "night"
        return "day" if 7 <= now.hour < 19 else "night"

    def render(
        self,
        data: ProactiveStatusImageData,
        now: datetime,
        mode: object = "自动切换",
    ) -> bytes:
        """Render proactive status as encoded PNG bytes.

        Args:
            data: Normalized command state.
            now: Current time in the plugin timezone.
            mode: Automatic, day, or night display mode.

        Returns:
            Encoded RGB PNG bytes.
        """
        theme = self.resolve_theme(mode, now)
        return self._render_image(data, now, theme)

    # -- palette -----------------------------------------------------------
    @staticmethod
    def _palette(theme: str) -> dict[str, Any]:
        if theme == "night":
            return {
                "stops": [
                    (0, c("#131228")),
                    (0.5, c("#1B2140")),
                    (1, c("#0F1626")),
                ],
                "glows": [
                    ("#6A55C8", 880, 300, 320, 85),
                    ("#2E8C96", 140, 1400, 300, 50),
                    ("#8A5A88", 880, 2800, 320, 45),
                ],
                "ink": "#ECEAF7",
                "sub": "#9995B5",
                "violet": "#9C8CF0",
                "cyan": "#52B8B4",
                "teal": "#5FC8B4",
                "green": "#6FBF95",
                "orange": "#E0A060",
                "rose": "#E08BA0",
                "chip3": "#E08BA0",
                "tint": (44, 46, 88),
                "talpha": 135,
                "line": (255, 255, 255, 48),
                "panel": (255, 255, 255, 26),
                "shadow_a": 95,
            }
        return {
            "stops": [(0, c("#F1E6F8")), (0.5, c("#E2EAFA")), (1, c("#F8E9E6"))],
            "glows": [
                ("#B79BEB", 880, 300, 320, 85),
                ("#8FD0D8", 140, 1400, 300, 55),
                ("#E8A8B8", 880, 2800, 320, 50),
            ],
            "ink": "#33304A",
            "sub": "#7A7490",
            "violet": "#7E68DC",
            "cyan": "#3FA8A8",
            "teal": "#3E9C8A",
            "green": "#3E9C6E",
            "orange": "#C9822F",
            "rose": "#D97BA0",
            "chip3": "#D97BA0",
            "tint": (255, 255, 255),
            "talpha": 135,
            "line": (255, 255, 255, 200),
            "panel": (255, 255, 255, 95),
            "shadow_a": 42,
        }

    # -- helpers -----------------------------------------------------------
    def _card(self, cv: Canvas, box, pal, radius=26):
        cv.shadow(box, radius, 20, 12, alpha=pal["shadow_a"])
        cv.glass(
            box,
            radius=radius,
            tint=pal["tint"],
            alpha=pal["talpha"],
            outline=pal["line"],
            owidth=1.5,
        )

    def _status_color(self, status: str | None, pal):
        if status is None:
            return None
        kind = _status_kind(status)
        if kind == "skipped":
            return pal["sub"]
        return {
            "sent": pal["green"],
            "waiting": pal["orange"],
            "other": pal["rose"],
        }.get(kind, pal["sub"])

    def _daily_metrics(
        self, cv: Canvas, item, item_f, status_f, x, maxw, right, line_h, pad
    ):
        """Height of one daily entry; status tag sits inline when it fits."""
        main, status = _split_status(item)
        lines = cv.wrap(main, item_f, maxw)
        h = len(lines) * line_h
        if status is not None:
            sw = cv.tlen(status, status_f)
            if x + cv.tlen(lines[-1], item_f) + 14 + sw > right:
                h += line_h
        return lines, status, h + pad

    def _draw_daily_item(
        self,
        cv: Canvas,
        x,
        y,
        item,
        item_f,
        status_f,
        maxw,
        right,
        line_h,
        pad,
        ink,
        pal,
    ):
        lines, status, h = self._daily_metrics(
            cv, item, item_f, status_f, x, maxw, right, line_h, pad
        )
        ly = y
        for line in lines:
            cv.text(x, ly, line, item_f, ink)
            ly += line_h
        if status is not None:
            col = self._status_color(status, pal)
            sw = cv.tlen(status, status_f)
            if x + cv.tlen(lines[-1], item_f) + 14 + sw <= right:
                cv.text(
                    x + cv.tlen(lines[-1], item_f) + 14,
                    ly - line_h + 3,
                    status,
                    status_f,
                    col,
                )
            else:
                cv.text(x, ly, status, status_f, col)
        return h

    # -- sections ----------------------------------------------------------
    def _draw_header(self, cv: Canvas, pal, now: datetime):
        ink = pal["ink"]
        self._card(cv, (58, 52, 1022, 236), pal)
        cv.avatar(self.plugin_dir / "logo.png", 926, 144, 104, pal["line"], 3)
        cv.spaced(92, 84, "PROACTIVE CONVERSATION", font(17, 500), pal["violet"], 5)
        cv.text(92, 114, "主动对话状态", font(40, 800), ink)
        cv.text(92, 178, f"{now:%Y.%m.%d · %H:%M}", font(20, 450), pal["sub"])

    def _draw_subscription(self, cv: Canvas, top, data, pal):
        ink, sub = pal["ink"], pal["sub"]
        self._card(cv, (58, top, 1022, top + 216), pal)
        main_color = pal["teal"] if data.subscribed else pal["rose"]
        cv.dot(104, top + 56, 8, c(main_color))
        cv.text(132, top + 34, data.subscription, font(30, 700), ink)
        note = "主动对话正在运行" if data.subscribed else "当前会话不会触发主动对话"
        cv.text(100, top + 98, note, font(21, 450), sub)
        chips = (
            ("忙碌联动", "忙碌中" if data.is_busy else "空闲", not data.is_busy),
            ("智能判断", "开启" if data.judge_enabled else "关闭", data.judge_enabled),
            (
                "聊天热度",
                data.heat_label,
                data.heat_label not in {"冷", "已关闭"},
            ),
        )
        colors = [pal["teal"], pal["violet"], pal["chip3"]]
        x = 100
        for i, (label, value, active) in enumerate(chips):
            f = font(18, 600)
            text = f"{label}  {value}"
            w = cv.tlen(text, f) + 32
            col = colors[i] if active else pal["rose"]
            cv.rrect((x, top + 140, x + w, top + 178), 19, fill=c(col, 36))
            cv.text(x + 16, top + 148, text, f, c(col))
            x += w + 14

    def _draw_overview(self, cv: Canvas, top, data, pal):
        ink, sub = pal["ink"], pal["sub"]
        rows = []
        probe = Canvas(height=8)
        value_f = font(23, 600)
        for label, value in data.facts:
            lines = probe.wrap(value, value_f, 660)
            rows.append((label, lines, 14 + len(lines) * 35 + 18))
        h = 66 + sum(r[2] for r in rows) + 14
        self._card(cv, (58, top, 1022, top + h), pal)
        cv.text(92, top + 24, "运行概览", font(26, 700), ink)
        cv.spaced(988, top + 32, "OVERVIEW", font(16, 500), sub, 4, anchor="ra")
        y = top + 66
        label_f = font(20, 450)
        for i, (label, lines, rh) in enumerate(rows):
            cv.text(92, y + 6, label, label_f, sub)
            ly = y
            for line in lines:
                cv.text(320, ly, line, value_f, ink)
                ly += 35
            y += rh
            if i < len(rows) - 1:
                cv.hline(92, 988, y - 10, c(ink, 30), 1)
        return h

    def _draw_daily(self, cv: Canvas, top, data, pal):
        ink, _sub = pal["ink"], pal["sub"]
        items = data.daily_items or ("暂无相关问候记录",)
        item_f, status_f = font(24, 450), font(19, 600)
        probe = Canvas(height=8)
        daily_hs = [
            self._daily_metrics(probe, it, item_f, status_f, 126, 800, 988, 36, 34)[2]
            for it in items
        ]
        h = 76 + sum(daily_hs) + 20
        self._card(cv, (58, top, 1022, top + h), pal)
        cv.text(92, top + 24, "相关每日问候", font(26, 700), ink)
        cv.spaced(
            988,
            top + 32,
            "DAILY GREETINGS",
            font(16, 500),
            pal["violet"],
            4,
            anchor="ra",
        )
        y = top + 76
        for i, item in enumerate(items):
            cv.dot(102, y + 16, 5, c(pal["violet"], 220))
            ih = self._draw_daily_item(
                cv, 126, y, item, item_f, status_f, 800, 988, 36, 34, ink, pal
            )
            y += ih
            if i < len(items) - 1:
                cv.hline(126, 988, y - 22, c(ink, 22), 1)
        return h

    def _draw_pending(self, cv: Canvas, top, data, pal):
        ink, _sub = pal["ink"], pal["sub"]
        items = data.pending_items or ("暂无待触发任务",)
        n = len(items)
        h = 76 + (n - 1) * 56 + 48
        self._card(cv, (58, top, 1022, top + h), pal)
        cv.text(92, top + 24, "待触发任务", font(26, 700), ink)
        cv.spaced(
            988,
            top + 32,
            "PENDING TASKS",
            font(16, 500),
            pal["cyan"],
            4,
            anchor="ra",
        )
        y = top + 76
        name_f, time_f = font(24, 600), font(19, 600)
        for i, item in enumerate(items):
            name, _, time_text = item.partition(" → ")
            cv.dot(102, y + 16, 5, c(pal["cyan"], 220))
            cv.text(126, y, name, name_f, ink)
            if time_text:
                cv.text(
                    988, y + 4, "→ " + time_text, time_f, c(pal["cyan"]), anchor="ra"
                )
            y += 56
            if i < n - 1:
                cv.hline(126, 988, y - 15, c(ink, 22), 1)
        return h

    # -- entry ---------------------------------------------------------------
    def _render_image(
        self, data: ProactiveStatusImageData, now: datetime, theme: str
    ) -> bytes:
        pal = self._palette(theme)
        _ink, sub = pal["ink"], pal["sub"]

        # measure pass fixes the canvas height before any drawing
        probe = Canvas(height=8)
        value_f = font(23, 600)
        ov_rows = []
        for label, value in data.facts:
            lines = probe.wrap(value, value_f, 660)
            ov_rows.append(14 + len(lines) * 35 + 18)
        daily_items = data.daily_items or ("暂无相关问候记录",)
        item_f, status_f = font(24, 450), font(19, 600)
        daily_hs = [
            self._daily_metrics(probe, it, item_f, status_f, 126, 800, 988, 36, 34)[2]
            for it in daily_items
        ]
        sub_top = 264
        ov_h = 66 + sum(ov_rows) + 14
        ov_top = sub_top + 216 + 28
        daily_h = 76 + sum(daily_hs) + 20
        daily_top = ov_top + ov_h + 28
        pend_items = data.pending_items or ("暂无待触发任务",)
        pend_h = 76 + (len(pend_items) - 1) * 56 + 48
        pend_top = daily_top + daily_h + 28
        yf = pend_top + pend_h + 40

        cv = Canvas(
            height=int(yf + 52) + 40, bg="#F1E6F8" if theme == "day" else "#131228"
        )
        cv.bg_gradient(pal["stops"])
        for col, gx, gy, gr, ga in pal["glows"]:
            cv.glow(gx, gy, gr, c(col), ga)

        self._draw_header(cv, pal, now)
        self._draw_subscription(cv, sub_top, data, pal)
        self._draw_overview(cv, ov_top, data, pal)
        self._draw_daily(cv, daily_top, data, pal)
        self._draw_pending(cv, pend_top, data, pal)

        cv.spaced(
            540, yf, "LINGXI · PROACTIVE STATUS", font(17, 500), sub, 5, anchor="ma"
        )
        return cv.finish(int(yf + 52))
