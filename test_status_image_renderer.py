from datetime import datetime
from io import BytesIO

from core.status_image_renderer import (
    ProactiveStatusImageData,
    ProactiveStatusImageRenderer,
)
from PIL import Image


def make_status(**overrides):
    values = {
        "subscription": "已订阅",
        "subscribed": True,
        "is_busy": False,
        "judge_enabled": True,
        "heat_label": "温",
        "heat_value": 0.42,
        "facts": (
            ("专属免打扰", "23:00-07:00"),
            ("全局免打扰", "00:00-06:30"),
            ("当前热度", "温(0.42) · 短期 30 分钟 4 条 · 长期 720 分钟 · 权重 70%"),
            ("距上次聊天", "18 分钟"),
        ),
        "daily_items": (
            "早安: 固定时间 → 08-18 08:30 · 待触发",
            "看完电影: 结束 08-17 20:46 → 08-17 21:16 · 已发送",
        ),
        "pending_items": (
            "沉寂问候 → 约 32 分钟后",
            "每日问候重试 → 23:18:30",
        ),
    }
    values.update(overrides)
    return ProactiveStatusImageData(**values)


def open_png(payload: bytes) -> Image.Image:
    image = Image.open(BytesIO(payload))
    image.load()
    return image


def test_theme_mode_resolves_explicit_and_automatic_hours(tmp_path):
    renderer = ProactiveStatusImageRenderer(tmp_path)

    assert renderer.resolve_theme("白天模式", datetime(2026, 8, 17, 23)) == "day"
    assert renderer.resolve_theme("夜间模式", datetime(2026, 8, 17, 12)) == "night"
    assert renderer.resolve_theme("自动切换", datetime(2026, 8, 17, 7)) == "day"
    assert renderer.resolve_theme("自动切换", datetime(2026, 8, 17, 18, 59)) == "day"
    assert renderer.resolve_theme("自动切换", datetime(2026, 8, 17, 19)) == "night"


def test_status_renders_day_and_night_without_logo(tmp_path):
    renderer = ProactiveStatusImageRenderer(tmp_path)
    status = make_status()

    day = open_png(renderer.render(status, datetime(2026, 8, 17, 12), "白天模式"))
    night = open_png(renderer.render(status, datetime(2026, 8, 17, 22), "夜间模式"))

    assert day.mode == night.mode == "RGB"
    assert day.width == night.width == 1080
    assert day.height == night.height
    assert day.height > 1200
    assert day.getpixel((500, 20)) != night.getpixel((500, 20))


def test_more_tasks_increase_image_height(tmp_path):
    renderer = ProactiveStatusImageRenderer(tmp_path)
    short = open_png(
        renderer.render(make_status(), datetime(2026, 8, 17, 12), "白天模式")
    )
    long_items = tuple(
        f"第 {index} 个很长的待触发任务，需要完整显示具体状态和时间"
        for index in range(8)
    )
    long = open_png(
        renderer.render(
            make_status(pending_items=long_items),
            datetime(2026, 8, 17, 12),
            "白天模式",
        )
    )

    assert long.height > short.height


def test_replacing_logo_changes_next_render(tmp_path):
    renderer = ProactiveStatusImageRenderer(tmp_path)
    status = make_status()
    logo_path = tmp_path / "logo.png"
    Image.new("RGB", (300, 300), "red").save(logo_path)
    first = renderer.render(status, datetime(2026, 8, 17, 12), "白天模式")

    Image.new("RGB", (300, 300), "blue").save(logo_path)
    second = renderer.render(status, datetime(2026, 8, 17, 12), "白天模式")

    assert first != second
