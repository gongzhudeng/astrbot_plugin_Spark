"""Local Pillow renderer for the proactive status command."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps


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
        image = self._render_image(data, now, theme)
        output = BytesIO()
        image.save(output, format="PNG", optimize=True)
        return output.getvalue()

    def _font(self, size: int, bold: bool = False):
        candidates = []
        if bold:
            candidates.extend(
                [
                    Path(r"C:\Windows\Fonts\Dengb.ttf"),
                    Path(r"C:\Windows\Fonts\msyhbd.ttc"),
                    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
                    Path("/System/Library/Fonts/PingFang.ttc"),
                ]
            )
        candidates.extend(
            [
                Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf"),
                Path(r"C:\Windows\Fonts\msyh.ttc"),
                Path(r"C:\Windows\Fonts\simhei.ttf"),
                Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
                Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
                Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
                Path("/System/Library/Fonts/PingFang.ttc"),
            ]
        )
        for path in candidates:
            if path.is_file():
                return ImageFont.truetype(str(path), size)
        return ImageFont.load_default(size=size)

    def _fonts(self) -> dict[str, Any]:
        return {
            "hero": self._font(56, True),
            "title": self._font(38, True),
            "section": self._font(31, True),
            "body": self._font(25),
            "body_bold": self._font(25, True),
            "small": self._font(21),
            "small_bold": self._font(21, True),
            "tiny": self._font(18),
        }

    def _load_avatar(self, size: int, border: tuple[int, int, int, int]):
        # Read on every render so replacing logo.png takes effect immediately.
        try:
            source = Image.open(self.plugin_dir / "logo.png").convert("RGB")
        except Exception:
            return None
        source = ImageOps.fit(
            source,
            (size, size),
            method=Image.Resampling.LANCZOS,
            centering=(0.55, 0.45),
        )
        mask = Image.new("L", (size, size), 0)
        ImageDraw.Draw(mask).ellipse((0, 0, size - 1, size - 1), fill=255)
        result = Image.new("RGBA", (size + 12, size + 12), (0, 0, 0, 0))
        ImageDraw.Draw(result).ellipse((0, 0, size + 11, size + 11), fill=border)
        result.paste(source, (6, 6), mask)
        return result

    @staticmethod
    def _rounded(draw, box, radius, fill, outline=None, width=1):
        draw.rounded_rectangle(
            box, radius=radius, fill=fill, outline=outline, width=width
        )

    @staticmethod
    def _shadow(image, box, radius=24, blur=14, offset=(0, 6)):
        layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        left, top, right, bottom = box
        ox, oy = offset
        draw.rounded_rectangle(
            (left + ox, top + oy, right + ox, bottom + oy),
            radius=radius,
            fill=(30, 40, 60, 26),
        )
        image.alpha_composite(layer.filter(ImageFilter.GaussianBlur(blur)))

    @staticmethod
    def _text_width(draw, text, font):
        bounds = draw.textbbox((0, 0), text, font=font)
        return bounds[2] - bounds[0]

    def _wrap(self, draw, text: str, font, max_width: int) -> list[str]:
        lines = []
        for paragraph in str(text).splitlines() or [""]:
            if not paragraph:
                lines.append("")
                continue
            line = ""
            for char in paragraph:
                candidate = line + char
                if line and self._text_width(draw, candidate, font) > max_width:
                    lines.append(line)
                    line = char
                else:
                    line = candidate
            if line:
                lines.append(line)
        return lines or [""]

    @staticmethod
    def _draw_lines(draw, x, y, lines, font, fill, step):
        for line in lines:
            draw.text((x, y), line, font=font, fill=fill)
            y += step
        return y

    @staticmethod
    def _palette(theme: str) -> dict[str, str]:
        if theme == "night":
            return {
                "bg": "#17191D",
                "grid": "#202329",
                "surface": "#1D2025",
                "surface_alt": "#22262C",
                "border": "#343941",
                "text": "#F1EEE8",
                "muted": "#9298A3",
                "blue": "#80D7C5",
                "coral": "#F06F61",
                "yellow": "#F4C866",
                "blue_soft": "#263F3A",
                "coral_soft": "#442B2B",
            }
        return {
            "bg": "#F5F7FB",
            "grid": "#E9EDF5",
            "surface": "#FFFFFF",
            "surface_alt": "#F8FAFD",
            "border": "#E0E6F0",
            "text": "#27324A",
            "muted": "#69748B",
            "blue": "#3C98D4",
            "coral": "#E76E68",
            "yellow": "#D29A26",
            "blue_soft": "#E7F4FF",
            "coral_soft": "#FFEAE8",
        }

    def _measure_fact_rows(self, draw, facts, fonts):
        rows = []
        for index in range(0, len(facts), 2):
            pair = facts[index : index + 2]
            measured = []
            for label, value in pair:
                lines = self._wrap(draw, value, fonts["body_bold"], 340)
                height = max(112, 55 + len(lines) * 34)
                measured.append((label, lines, height))
            rows.append((measured, max(item[2] for item in measured)))
        return rows

    def _measure_list(self, draw, items, fonts):
        measured = []
        for item in items or ("无",):
            lines = self._wrap(draw, item, fonts["body"], 820)
            height = max(52, 16 + len(lines) * 34)
            measured.append((lines, height))
        return measured

    def _render_image(self, data, now, theme):
        fonts = self._fonts()
        colors = self._palette(theme)
        probe = ImageDraw.Draw(Image.new("RGB", (self.width, 100)))
        fact_rows = self._measure_fact_rows(probe, data.facts, fonts)
        daily_rows = self._measure_list(probe, data.daily_items, fonts)
        pending_rows = self._measure_list(probe, data.pending_items, fonts)

        header_top = 42
        header_height = 210
        summary_top = 284
        summary_height = 226
        facts_top = summary_top + summary_height + 32
        facts_height = 76 + sum(row[1] + 14 for row in fact_rows) + 12
        daily_top = facts_top + facts_height + 30
        daily_height = 82 + sum(row[1] for row in daily_rows) + 24
        pending_top = daily_top + daily_height + 30
        pending_height = 82 + sum(row[1] for row in pending_rows) + 24
        height = pending_top + pending_height + 82

        image = Image.new("RGBA", (self.width, height), colors["bg"])
        draw = ImageDraw.Draw(image)
        for y in range(0, height, 48 if theme == "day" else 64):
            draw.line((0, y, self.width, y), fill=colors["grid"], width=1)
        if theme == "day":
            draw.rectangle((0, 0, 18, height), fill="#FF8E88")
            draw.rectangle((18, 0, 28, height), fill="#87BDE8")
            self._rounded(
                draw,
                (58, header_top, 1022, header_top + header_height),
                32,
                colors["surface"],
            )
        else:
            draw.rectangle((0, 0, self.width, 18), fill=colors["coral"])

        avatar = self._load_avatar(
            138,
            (135, 189, 232, 255) if theme == "day" else (240, 111, 97, 255),
        )
        if avatar:
            image.alpha_composite(avatar, (82 if theme == "day" else 844, 72))
        title_x = 254 if theme == "day" else 58
        draw.text(
            (title_x, 72), "主动对话状态", font=fonts["hero"], fill=colors["text"]
        )
        draw.text(
            (title_x + 3, 145),
            f"{now:%Y.%m.%d  ·  %H:%M}",
            font=fonts["small"],
            fill=colors["muted"],
        )
        draw.text(
            (title_x + 3, 188),
            "LINGXI · PROACTIVE CONVERSATION",
            font=fonts["tiny"],
            fill=colors["blue"],
        )

        if theme == "day":
            self._shadow(image, (58, summary_top, 1022, summary_top + summary_height))
            draw = ImageDraw.Draw(image)
        self._rounded(
            draw,
            (58, summary_top, 1022, summary_top + summary_height),
            28 if theme == "day" else 20,
            colors["surface_alt"],
            outline=colors["blue"] if theme == "night" else None,
            width=3 if theme == "night" else 1,
        )
        main_color = colors["blue"] if data.subscribed else colors["coral"]
        draw.ellipse((92, summary_top + 44, 122, summary_top + 74), fill=main_color)
        draw.text(
            (148, summary_top + 25),
            data.subscription,
            font=fonts["title"],
            fill=colors["text"],
        )
        subtitle = "主动对话正在运行" if data.subscribed else "当前会话不会触发主动对话"
        draw.text(
            (94, summary_top + 100),
            subtitle,
            font=fonts["body"],
            fill=colors["muted"],
        )
        chips = (
            ("忙碌联动", "忙碌中" if data.is_busy else "空闲", not data.is_busy),
            ("智能判断", "开启" if data.judge_enabled else "关闭", data.judge_enabled),
            ("聊天热度", data.heat_label, data.heat_label not in {"冷", "已关闭"}),
        )
        chip_x = 94
        for label, value, active in chips:
            box = (chip_x, summary_top + 156, chip_x + 260, summary_top + 202)
            fill = colors["blue_soft"] if active else colors["coral_soft"]
            self._rounded(draw, box, 12, fill)
            draw.text(
                (chip_x + 16, summary_top + 166),
                f"{label}  {value}",
                font=fonts["tiny"],
                fill=colors["blue"] if active else colors["coral"],
            )
            chip_x += 284

        self._rounded(
            draw,
            (58, facts_top, 1022, facts_top + facts_height),
            26 if theme == "day" else 18,
            colors["surface"],
            outline=colors["border"],
        )
        draw.text(
            (88, facts_top + 25), "运行概览", font=fonts["section"], fill=colors["text"]
        )
        y = facts_top + 76
        for row, row_height in fact_rows:
            for column, (label, lines, _height) in enumerate(row):
                x = 84 + column * 476
                self._rounded(
                    draw,
                    (x, y, x + 448, y + row_height),
                    15,
                    colors["surface_alt"],
                    outline=colors["border"],
                )
                draw.text(
                    (x + 20, y + 16), label, font=fonts["tiny"], fill=colors["muted"]
                )
                self._draw_lines(
                    draw,
                    x + 20,
                    y + 49,
                    lines,
                    fonts["body_bold"],
                    colors["text"],
                    34,
                )
            y += row_height + 14

        self._draw_list_section(
            draw,
            daily_top,
            daily_height,
            "相关每日问候",
            daily_rows,
            colors["yellow"],
            colors,
            fonts,
            theme,
        )
        self._draw_list_section(
            draw,
            pending_top,
            pending_height,
            "待触发任务",
            pending_rows,
            colors["blue"],
            colors,
            fonts,
            theme,
        )
        draw.text(
            (58, height - 48),
            "LINGXI  ·  PROACTIVE STATUS",
            font=fonts["tiny"],
            fill=colors["muted"],
        )
        return image.convert("RGB")

    def _draw_list_section(
        self, draw, top, height, title, rows, accent, colors, fonts, theme
    ):
        self._rounded(
            draw,
            (58, top, 1022, top + height),
            26 if theme == "day" else 18,
            colors["surface"],
            outline=colors["border"],
        )
        draw.text((88, top + 24), title, font=fonts["section"], fill=colors["text"])
        y = top + 76
        for lines, row_height in rows:
            draw.ellipse((94, y + 12, 106, y + 24), fill=accent)
            self._draw_lines(
                draw,
                126,
                y + 3,
                lines,
                fonts["body"],
                colors["text"],
                34,
            )
            y += row_height
