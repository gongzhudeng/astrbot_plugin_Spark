from __future__ import annotations

import asyncio
import json
import os
import random
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.star import Context, Star, register
from astrbot.core.message.components import Image

from .core.daily_projection import (
    project_activity_candidates,
    select_highest_priority,
)
from .core.daily_task_state import (
    DAILY_STATE_SCHEMA_VERSION,
    DailyTaskState,
    begin_attempt,
    legacy_daily_state,
    new_task_state,
    normalize_daily_states,
    normalize_success_times,
    plan_task,
    record_success_time,
    store_daily_state,
    success_interval_deadline,
    technical_failure,
    terminal_state,
)
from .core.heat_policy import (
    dual_scale_heat,
    geometric_delay,
    heat_scaled_delay_seconds,
)
from .core.history_content import (
    build_proactive_user_content,
    build_user_content_with_datetime,
    extract_history_text,
    find_datetime_reminder,
)
from .core.message_input import is_slash_prefixed_message
from .core.proactive_delivery import (
    DIRECT_DELIVERY_KIND_EXTRA,
    DIRECT_DELIVERY_TEXT_EXTRA,
    AgentDeliveryResult,
    resolve_agent_delivery,
)
from .core.proactive_evidence import (
    EVIDENCE_SCHEMA_VERSION,
    acknowledge_pending_evidence,
    normalize_evidence_records,
    record_proactive_delivery,
)
from .core.provider_fallback import (
    ASTRBOT_FALLBACK_MODE,
    dedupe_provider_chain,
    normalize_provider_ids,
    resolve_provider_chain,
    select_generation_fallback_ids,
)
from .core.status_image_renderer import (
    ProactiveStatusImageData,
    ProactiveStatusImageRenderer,
)
from .core.time_policy import (
    apply_datetime_policy,
    apply_delay_policy,
    apply_seconds_policy,
    cooldown_deadline,
    migrate_compact_policy_values,
    parse_policy,
)

SESSION_DATA_SCHEMA_VERSION = 4

# 尝试导入 StarTools（如果可用）
try:
    from astrbot.api.star import StarTools

    HAS_STARTOOLS = True
except ImportError:
    HAS_STARTOOLS = False

# 尝试导入新的Message模型（新版本astrbot）
try:
    from astrbot.core.agent.message import (
        AssistantMessageSegment,
        TextPart,
        UserMessageSegment,
    )

    HAS_NEW_MESSAGE_API = True
except ImportError:
    HAS_NEW_MESSAGE_API = False

# 尝试导入 llm_tool（Agent 工具注册装饰器）
try:
    from astrbot.api import llm_tool

    HAS_LLM_TOOL = True
except ImportError:
    # 兼容旧版本：提供空装饰器
    def llm_tool(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    HAS_LLM_TOOL = False

# Request-hook APIs are useful even when the full Agent Pipeline is unavailable.
try:
    from astrbot.core.cron.events import CronMessageEvent
    from astrbot.core.pipeline.context import call_event_hook
    from astrbot.core.platform.message_session import MessageSession
    from astrbot.core.provider.entities import ProviderRequest
    from astrbot.core.star.star_handler import EventType

    HAS_REQUEST_HOOKS = True
    REQUEST_HOOK_IMPORT_ERROR = ""
except ImportError as exc:
    HAS_REQUEST_HOOKS = False
    REQUEST_HOOK_IMPORT_ERROR = repr(exc)

# Import the full Agent Pipeline separately so the legacy provider path can still use hooks.
try:
    from astrbot.core.astr_main_agent import build_main_agent

    from .core.agent_config import build_main_agent_config

    HAS_AGENT_PIPELINE = HAS_REQUEST_HOOKS
    AGENT_PIPELINE_IMPORT_ERROR = "" if HAS_REQUEST_HOOKS else REQUEST_HOOK_IMPORT_ERROR
except ImportError as exc:
    HAS_AGENT_PIPELINE = False
    AGENT_PIPELINE_IMPORT_ERROR = repr(exc)


# 工具函数
def _is_slash_prefixed_event(event: AstrMessageEvent) -> bool:
    marker = "spark_slash_input"
    cached = event.get_extra(marker, None)
    if isinstance(cached, bool):
        return cached
    is_slash = is_slash_prefixed_message(event.get_messages())
    event.set_extra(marker, is_slash)
    return is_slash


def _ensure_dir(p: str) -> str:
    """确保目录存在，不存在则创建"""
    os.makedirs(p, exist_ok=True)
    return p


def _now_tz(tz_name: str | None) -> datetime:
    """获取指定时区的当前时间，失败则返回本地时间"""
    try:
        if tz_name:
            import zoneinfo

            try:
                return datetime.now(zoneinfo.ZoneInfo(tz_name))
            except (zoneinfo.ZoneInfoNotFoundError, ValueError) as e:
                logger.warning(f"[Spark] 无效时区 '{tz_name}': {e}，使用系统默认时区")
                return datetime.now()
    except ImportError:
        # Python < 3.9 需要 backports.zoneinfo
        try:
            from backports import zoneinfo

            return datetime.now(zoneinfo.ZoneInfo(tz_name))
        except Exception:
            pass
    return datetime.now()


def _compute_heat(
    msg_timestamps: list,
    now_ts: float,
    window_minutes: float,
    full_score_messages: float = 10.0,
) -> float:
    """Compatibility wrapper for the original single-window heat calculation."""
    return dual_scale_heat(
        msg_timestamps,
        now_ts,
        short_window_minutes=window_minutes,
        long_window_minutes=window_minutes,
        messages_for_full_score=full_score_messages,
        short_weight=1.0,
    )


def _parse_hhmm(s: str) -> tuple[int, int] | None:
    """解析 HH:MM 格式时间字符串，返回 (小时, 分钟) 或 None"""
    if not s:
        return None
    m = re.match(r"^([01]?\d|2[0-3]):([0-5]\d)$", s.strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _in_quiet(now: datetime, quiet: str) -> bool:
    """检查当前时间是否在免打扰时间段内（支持跨天）"""
    if not quiet or "-" not in quiet:
        return False
    a, b = quiet.split("-", 1)
    p1 = _parse_hhmm(a)
    p2 = _parse_hhmm(b)
    if not p1 or not p2:
        return False
    t1 = time(p1[0], p1[1])
    t2 = time(p2[0], p2[1])
    nt = now.time()
    if t1 <= t2:
        return t1 <= nt <= t2
    else:
        return nt >= t1 or nt <= t2


def _fmt_now(fmt: str, tz: str | None) -> str:
    """格式化当前时间为指定格式"""
    return _now_tz(tz).strftime(fmt)


def _format_time_delta(seconds: float) -> str:
    """将时间差（秒）格式化为友好的文本

    示例：
    - 180秒 -> "3分钟"
    - 3600秒 -> "1小时"
    - 7200秒 -> "2小时"
    - 86400秒 -> "1天"
    - 90000秒 -> "1天1小时"
    """
    if seconds < 60:
        return "不到1分钟"

    minutes = int(seconds / 60)
    hours = int(minutes / 60)
    days = int(hours / 24)

    if days > 0:
        remaining_hours = hours % 24
        if remaining_hours > 0:
            return f"{days}天{remaining_hours}小时"
        return f"{days}天"
    elif hours > 0:
        remaining_minutes = minutes % 60
        if remaining_minutes > 0:
            return f"{hours}小时{remaining_minutes}分钟"
        return f"{hours}小时"
    else:
        return f"{minutes}分钟"


class _SafeTemplateValues(dict[str, object]):
    """Keep unknown legacy placeholders visible instead of dropping the prompt."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _format_template(template: str, values: dict[str, object]) -> str:
    try:
        return template.format_map(_SafeTemplateValues(values))
    except (AttributeError, IndexError, KeyError, ValueError):
        rendered = template
        for key, value in values.items():
            rendered = rendered.replace("{" + key + "}", str(value))
        return rendered


# 数据类定义
@dataclass
class UserProfile:
    """用户订阅信息和个性化设置"""

    subscribed: bool = False
    idle_after_minutes: int | None = None
    daily_reminders_enabled: bool = True
    quiet_hours: str | None = None  # 用户专属免打扰时间 "HH:MM-HH:MM"
    manual_unsubscribe: bool = False  # 标记是否是手动退订（强开关）
    auto_unsubscribed: bool = False  # 标记是否是自动退订（用于自动重新激活判断）

    def to_dict(self):
        return {
            "subscribed": self.subscribed,
            "idle_after_minutes": self.idle_after_minutes,
            "daily_reminders_enabled": self.daily_reminders_enabled,
            "quiet_hours": self.quiet_hours,
            "manual_unsubscribe": self.manual_unsubscribe,
            "auto_unsubscribed": self.auto_unsubscribed,
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            subscribed=data.get("subscribed", False),
            idle_after_minutes=data.get("idle_after_minutes"),
            daily_reminders_enabled=data.get("daily_reminders_enabled", True),
            quiet_hours=data.get("quiet_hours"),
            manual_unsubscribe=data.get("manual_unsubscribe", False),
            auto_unsubscribed=data.get("auto_unsubscribed", False),
        )


@dataclass
class SessionState:
    """运行时会话状态（内存中维护）"""

    last_ts: float = 0.0
    last_fired_tag: str = ""  # 保留用于向后兼容
    last_fired_tags: dict = None  # 改为字典：{tag: timestamp}，支持过期清理
    daily_task_results: dict = (
        None  # Legacy terminal outcomes for one compatibility cycle
    )
    daily_task_state_schema_version: int = DAILY_STATE_SCHEMA_VERSION
    daily_task_states: dict = None
    daily_greeting_success_times: dict = None
    last_user_reply_ts: float = 0.0
    consecutive_no_reply_count: int = 0
    next_idle_ts: float = 0.0
    idle_retry_after_ts: float = 0.0
    idle_judge_cycle: int = 0
    idle_judge_checked_cycle: int = -1
    idle_judge_inflight_cycle: int = -1
    idle_judge_task_ts: float = 0.0
    idle_judge_anchor_ts: float = 0.0
    idle_schedule_mode: str = ""
    last_proactive_reply_ts: float = 0.0  # 最近一次主动回复时间戳
    last_ai_reply_ts: float = 0.0  # 最近一次 AI 普通回复时间戳（用于对话增强取消判断）
    msg_timestamps: list = (
        None  # rolling window of user message timestamps for heat computation
    )
    proactive_recent_messages: list = (
        None  # Deprecated; kept only for old session data compatibility
    )
    proactive_evidence_schema_version: int = EVIDENCE_SCHEMA_VERSION
    proactive_evidence: list = None
    next_enhancement_ts: float = (
        0.0  # scheduled enhancement fire time (runtime only, not persisted)
    )

    def __post_init__(self):
        """初始化后处理"""
        if self.last_fired_tags is None:
            self.last_fired_tags = {}
        if self.last_fired_tag and self.last_fired_tag not in self.last_fired_tags:
            self.last_fired_tags[self.last_fired_tag] = _now_tz(None).timestamp()
        if self.daily_task_results is None:
            self.daily_task_results = {}
        self.daily_task_state_schema_version = DAILY_STATE_SCHEMA_VERSION
        self.daily_task_states = normalize_daily_states(
            self.daily_task_states,
            legacy_results=self.daily_task_results,
            last_fired_tags=self.last_fired_tags,
        )
        self.daily_greeting_success_times = normalize_success_times(
            self.daily_greeting_success_times
        )
        if self.msg_timestamps is None:
            self.msg_timestamps = []
        if self.proactive_recent_messages is None:
            self.proactive_recent_messages = []
        self.proactive_evidence_schema_version = EVIDENCE_SCHEMA_VERSION
        self.proactive_evidence = normalize_evidence_records(self.proactive_evidence)
        if (
            self.idle_judge_inflight_cycle == self.idle_judge_cycle
            and self.idle_judge_task_ts <= 0
        ):
            self.idle_judge_checked_cycle = -1
            self.idle_judge_inflight_cycle = -1
            self.next_idle_ts = 0.0

    def to_dict(self):
        return {
            "last_ts": self.last_ts,
            "last_fired_tag": self.last_fired_tag,  # 保留用于向后兼容
            "last_fired_tags": self.last_fired_tags if self.last_fired_tags else {},
            "daily_task_results": self.daily_task_results
            if self.daily_task_results
            else {},
            "daily_task_state_schema_version": self.daily_task_state_schema_version,
            "daily_task_states": normalize_daily_states(self.daily_task_states),
            "daily_greeting_success_times": normalize_success_times(
                self.daily_greeting_success_times
            ),
            "last_user_reply_ts": self.last_user_reply_ts,
            "consecutive_no_reply_count": self.consecutive_no_reply_count,
            "next_idle_ts": self.next_idle_ts,
            "idle_retry_after_ts": self.idle_retry_after_ts,
            "idle_judge_cycle": self.idle_judge_cycle,
            "idle_judge_checked_cycle": self.idle_judge_checked_cycle,
            "idle_judge_inflight_cycle": self.idle_judge_inflight_cycle,
            "idle_judge_task_ts": self.idle_judge_task_ts,
            "idle_judge_anchor_ts": self.idle_judge_anchor_ts,
            "idle_schedule_mode": self.idle_schedule_mode,
            "last_proactive_reply_ts": self.last_proactive_reply_ts,
            "last_ai_reply_ts": self.last_ai_reply_ts,
            "msg_timestamps": self.msg_timestamps if self.msg_timestamps else [],
            "proactive_evidence_schema_version": self.proactive_evidence_schema_version,
            "proactive_evidence": normalize_evidence_records(self.proactive_evidence),
        }

    @classmethod
    def from_dict(cls, data: dict):
        tags_dict = data.get("last_fired_tags", {})
        if not isinstance(tags_dict, dict):
            tags_dict = {}
        task_results = data.get("daily_task_results", {})
        if not isinstance(task_results, dict):
            task_results = {}
        msg_ts = data.get("msg_timestamps", [])
        if not isinstance(msg_ts, list):
            msg_ts = []
        proactive_recent = []

        return cls(
            last_ts=data.get("last_ts", 0.0),
            last_fired_tag=data.get("last_fired_tag", ""),
            last_fired_tags=tags_dict,
            daily_task_results=task_results,
            daily_task_state_schema_version=data.get(
                "daily_task_state_schema_version", DAILY_STATE_SCHEMA_VERSION
            ),
            daily_task_states=data.get("daily_task_states", {}),
            daily_greeting_success_times=data.get("daily_greeting_success_times", {}),
            last_user_reply_ts=data.get("last_user_reply_ts", 0.0),
            consecutive_no_reply_count=data.get("consecutive_no_reply_count", 0),
            next_idle_ts=data.get("next_idle_ts", 0.0),
            idle_retry_after_ts=data.get("idle_retry_after_ts", 0.0),
            idle_judge_cycle=data.get("idle_judge_cycle", 0),
            idle_judge_checked_cycle=data.get("idle_judge_checked_cycle", -1),
            idle_judge_inflight_cycle=data.get("idle_judge_inflight_cycle", -1),
            idle_judge_task_ts=data.get("idle_judge_task_ts", 0.0),
            idle_judge_anchor_ts=data.get("idle_judge_anchor_ts", 0.0),
            idle_schedule_mode=data.get("idle_schedule_mode", ""),
            last_proactive_reply_ts=data.get("last_proactive_reply_ts", 0.0),
            last_ai_reply_ts=data.get("last_ai_reply_ts", 0.0),
            msg_timestamps=msg_ts,
            proactive_recent_messages=proactive_recent,
            proactive_evidence_schema_version=data.get(
                "proactive_evidence_schema_version", EVIDENCE_SCHEMA_VERSION
            ),
            proactive_evidence=normalize_evidence_records(
                data.get("proactive_evidence", [])
            ),
        )

    def has_fired(self, tag: str) -> bool:
        """检查某个标记是否已触发（支持过期清理）"""
        if not self.last_fired_tags:
            return False
        return tag in self.last_fired_tags

    def mark_fired(self, tag: str):
        """标记某个事件已触发"""
        if self.last_fired_tags is None:
            self.last_fired_tags = {}
        self.last_fired_tags[tag] = _now_tz(None).timestamp()
        # 同时更新 last_fired_tag 用于向后兼容
        self.last_fired_tag = tag

        # 清理过期标记（保留最近7天的记录）
        now_ts = _now_tz(None).timestamp()
        expired_tags = [
            t for t, ts in self.last_fired_tags.items() if now_ts - ts > 7 * 86400
        ]
        for t in expired_tags:
            del self.last_fired_tags[t]

    def daily_state(self, task: DailyGreetingTask) -> DailyTaskState:
        raw = self.daily_task_states.get(task.tag, {})
        state = (
            DailyTaskState.from_dict(raw, tag=task.tag)
            if isinstance(raw, dict) and raw
            else None
        )
        if state is not None:
            return state
        if self.has_fired(task.tag):
            return legacy_daily_state(
                task.tag,
                self.daily_task_results.get(task.tag),
                fired_at=self.last_fired_tags.get(task.tag, 0.0),
            )
        return new_task_state(
            task.tag,
            target_at=task.target.timestamp(),
            source_date=task.source_date.isoformat(),
        )

    def set_daily_state(self, state: DailyTaskState) -> None:
        self.daily_task_states = store_daily_state(self.daily_task_states, state)
        if not state.terminal:
            return
        legacy_result = {
            "skipped_cooldown": "cooldown_skipped",
        }.get(state.status, state.status)
        self.mark_fired(state.tag)
        self.daily_task_results[state.tag] = legacy_result
        active_tags = set(self.last_fired_tags)
        self.daily_task_results = {
            task_tag: task_result
            for task_tag, task_result in self.daily_task_results.items()
            if task_tag in active_tags
        }

    def mark_daily_result(self, tag: str, result: str):
        """Persist a legacy terminal outcome for compatibility callers."""
        self.set_daily_state(legacy_daily_state(tag, result))


@dataclass(frozen=True)
class DailyGreetingTask:
    slot_num: int
    greeting_id: str
    target: datetime
    tag: str
    prompt: object
    ignore_dnd: bool
    ignore_judge: bool
    cooldown_minutes: int
    activity_trigger_interval_minutes: int
    source_date: date
    source_type: str = "fixed"
    activity: str = ""
    occurrence: int = 0
    timeline_index: int = -1
    boundary: str = ""
    base: datetime | None = None


@dataclass(frozen=True)
class DailyGreetingIssue:
    slot_num: int
    source_date: date
    status: str
    detail: str = ""
    activity: str = ""
    occurrence: int = 0


@dataclass(frozen=True)
class DailyGreetingProjection:
    tasks: list[DailyGreetingTask]
    issues: list[DailyGreetingIssue]


# 灵犀 · 主动对话插件
# 灵感参考：astrbot_plugin_Conversa v3.0.0 (Luna-channel)
@register(
    "astrbot_plugin_Spark",
    "灵犀 · 主动对话",
    "让 AI 像真人一样主动找你聊天——通过大模型智能判断何时该开口、何时该沉默，支持忙碌时段免打扰、独立判断/生成双模型、无限定时问候",
    "2.7.0",
    "https://github.com/gongzhudeng/astrbot_plugin_Spark",
)
class Spark(Star):
    # 初始化
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.cfg: AstrBotConfig = config
        self._loop_task: asyncio.Task | None = None
        self._stopped: bool = False  # 插件停止标志

        # 运行时数据
        self._states: dict[str, SessionState] = {}
        self._user_profiles: dict[str, UserProfile] = {}
        self._timeline_warning_at: float = 0.0

        # 文件保存去抖相关
        self._save_user_data_task: asyncio.Task | None = None
        self._save_session_data_task: asyncio.Task | None = None
        self._save_delay_seconds = 2.0  # 去抖延迟：2秒

        # 对话增强相关
        self._enhancement_tasks: dict[str, asyncio.Task] = {}
        self._enhancement_gen: dict[str, int] = {}  # generation counter per umo
        self._heat_event_marker = "_spark_heat_counted"
        self._status_image_renderer = ProactiveStatusImageRenderer(Path(__file__).parent)

        # 数据文件路径（使用规范的方式获取插件数据目录）
        if HAS_STARTOOLS:
            # 使用 StarTools 获取规范的数据目录
            data_dir_path = StarTools.get_data_dir() / "astrbot_plugin_conversa"
            self._data_dir = str(data_dir_path)
            os.makedirs(self._data_dir, exist_ok=True)
        else:
            # 后备方案：使用更可靠的方式获取数据目录
            # 尝试从 context 获取，如果不可用则使用当前文件的相对路径
            try:
                # 尝试使用 context 获取数据路径
                if hasattr(context, "get_data_path") or hasattr(self, "get_data_path"):
                    data_path_func = getattr(context, "get_data_path", None) or getattr(
                        self, "get_data_path", None
                    )
                    if data_path_func:
                        base_path = data_path_func()
                        self._data_dir = _ensure_dir(
                            os.path.join(base_path, "astrbot_plugin_conversa")
                        )
                    else:
                        raise AttributeError
                else:
                    raise AttributeError
            except (AttributeError, TypeError):
                # 最终后备：基于当前工作目录，但添加警告
                import warnings

                warnings.warn(
                    "[Spark] 无法使用 StarTools，使用 os.getcwd() 作为后备方案",
                    stacklevel=2,
                )
                root = os.getcwd()
                self._data_dir = _ensure_dir(
                    os.path.join(root, "data", "plugin_data", "astrbot_plugin_conversa")
                )

        self._user_data_path = os.path.join(self._data_dir, "user_data.json")
        self._session_data_path = os.path.join(self._data_dir, "session_data.json")

        # 加载数据
        self._load_user_data()
        self._load_session_data()
        self._sync_subscribed_users_from_config()
        self._migrate_config()
        self._migrate_daily_greetings()
        self._migrate_daily_greeting_controls()
        self._migrate_compact_time_policy_values()

    def _migrate_compact_time_policy_values(self):
        try:
            if migrate_compact_policy_values(self.cfg):
                self.cfg.save_config()
                logger.info("[Spark] 已迁移旧整数时间浮动配置为单框字符串格式")
        except Exception as e:
            logger.warning(f"[Spark] 时间浮动配置迁移失败: {e}")

    def _migrate_daily_greetings(self):
        """一次性迁移：旧的 daily1/2/3 扁平配置 -> 新的 daily_greetings 列表格式"""
        try:
            daily = self.cfg.get("daily_prompts") or {}
            if not isinstance(daily, dict):
                return

            # 已经是新格式，跳过
            if "daily_greetings" in daily:
                return

            greetings = []
            # 检查旧的 slot 格式（slot1/slot2/slot3）
            for slot_num in [1, 2, 3]:
                slot_cfg = daily.get(f"slot{slot_num}", {})
                if isinstance(slot_cfg, dict):
                    greetings.append(
                        {
                            "enable": slot_cfg.get("enable", False),
                            "time": slot_cfg.get("time", ""),
                            "prompt": slot_cfg.get("prompt", ""),
                            "ignore_dnd": False,
                        }
                    )

            # 如果没有 slot 格式，检查扁平格式（daily1_enable/time1/prompt1）
            if not any(g.get("time") for g in greetings):
                greetings = []
                for n in [1, 2, 3]:
                    if daily.get(f"daily{n}_enable", False) or daily.get(
                        f"time{n}", ""
                    ):
                        greetings.append(
                            {
                                "enable": daily.get(f"daily{n}_enable", False),
                                "time": daily.get(f"time{n}", ""),
                                "prompt": daily.get(f"prompt{n}", ""),
                                "ignore_dnd": False,
                            }
                        )

            if greetings:
                daily["daily_greetings"] = greetings
                self.cfg["daily_prompts"] = daily
                self.cfg.save_config()
                logger.info(
                    f"[Spark] 已迁移旧每日问候配置到新列表格式（{len(greetings)} 个时段）"
                )
        except Exception as e:
            logger.warning(f"[Spark] 每日问候配置迁移失败: {e}")

    def _migrate_daily_greeting_controls(self):
        """Migrate legacy per-greeting controls without changing delivery behavior."""
        try:
            daily = self.cfg.get("daily_prompts") or {}
            greetings = (
                daily.get("daily_greetings") if isinstance(daily, dict) else None
            )
            if not isinstance(greetings, list):
                return
            changed = False
            legacy_occurrences_key = "activity_" + "occurrences"
            for greeting in greetings:
                if not isinstance(greeting, dict):
                    continue
                if "ignore_judge" not in greeting:
                    greeting["ignore_judge"] = True
                    changed = True
                if (
                    greeting.get("trigger_source", "固定时间")
                    in {
                        "activity",
                        "日程活动",
                    }
                    and "activity_trigger_interval_minutes" not in greeting
                ):
                    greeting["activity_trigger_interval_minutes"] = 0
                    changed = True
                if legacy_occurrences_key in greeting:
                    greeting.pop(legacy_occurrences_key, None)
                    changed = True
            if changed:
                self.cfg["daily_prompts"] = daily
                self.cfg.save_config()
                logger.info("[Spark] 已迁移每日问候判断开关与活动触发间隔配置")
        except Exception as e:
            logger.warning(f"[Spark] 每日问候控制配置迁移失败: {e}")

    def _migrate_config(self):
        """One-time config migration: old locations -> new locations"""
        try:
            changed = False
            proactive = self.cfg.get("proactive_settings") or {}
            basic = self.cfg.get("basic_settings") or {}
            advanced = self.cfg.get("advanced") or {}
            heat = self.cfg.get("heat_settings") or {}

            if isinstance(heat, dict):
                if "heat_window_minutes" not in heat:
                    heat["heat_window_minutes"] = int(
                        float(heat.get("heat_window_hours", 4) or 4) * 60
                    )
                    changed = True
                    logger.info(
                        "[Spark] migrated heat_window_hours -> heat_window_minutes"
                    )
                if "heat_messages_for_full_score" not in heat:
                    heat["heat_messages_for_full_score"] = 10
                    changed = True
                idle = self.cfg.get("idle_greetings") or {}
                if not isinstance(idle, dict):
                    idle = {}
                if "mode" not in idle:
                    idle["mode"] = (
                        "对话热度"
                        if bool(heat.get("enable_heat", True))
                        else "固定时间"
                    )
                    changed = True
                for key in ("hot_delay_minutes", "cold_delay_minutes"):
                    if key not in idle and key in heat:
                        idle[key] = heat[key]
                        changed = True
                if "idle_after_minutes" not in idle and "idle_after_minutes" in heat:
                    idle["idle_after_minutes"] = heat["idle_after_minutes"]
                    changed = True
                for key, default in (
                    ("judge_after_minutes", 60),
                    ("judge_min_delay_minutes", 5),
                    ("judge_max_delay_minutes", 1440),
                ):
                    if key not in idle:
                        idle[key] = default
                        changed = True
                self.cfg["idle_greetings"] = idle
                self.cfg["heat_settings"] = heat

            idle = self.cfg.get("idle_greetings") or {}
            if isinstance(idle, dict):
                legacy_idle_fluctuation = idle.get(
                    "random_fluctuation_minutes",
                    idle.get("jitter_minutes"),
                )
                if (
                    "idle_random_fluctuation_minutes" not in idle
                    and legacy_idle_fluctuation not in (None, "")
                ):
                    idle["idle_random_fluctuation_minutes"] = legacy_idle_fluctuation
                    changed = True
                for old_key, new_key in (
                    ("hot_delay_minutes", "hot_delay_minutes"),
                    ("cold_delay_minutes", "cold_delay_minutes"),
                ):
                    if new_key not in idle and old_key in heat:
                        idle[new_key] = heat[old_key]
                        changed = True
                self.cfg["idle_greetings"] = idle

            idle = self.cfg.get("idle_greetings") or {}
            if isinstance(idle, dict):
                provider_migration_marker = "_idle_judge_provider_migrated"
                migration_complete = bool(idle.get(provider_migration_marker, False))
                legacy_provider = proactive.get("proactive_judge_provider", "")
                legacy_fallbacks = proactive.get(
                    "proactive_judge_fallback_providers", []
                )
                has_legacy_provider = bool(str(legacy_provider or "").strip())
                has_legacy_fallbacks = isinstance(legacy_fallbacks, list) and bool(
                    legacy_fallbacks
                )
                if not migration_complete:
                    # Schema defaults may materialize the new fields as empty values
                    # before this hook runs. Only a non-empty legacy chain is copied.
                    if (
                        not str(idle.get("idle_judge_provider") or "").strip()
                        and has_legacy_provider
                    ):
                        idle["idle_judge_provider"] = legacy_provider
                        changed = True
                    if (
                        not idle.get("idle_judge_fallback_providers")
                        and has_legacy_fallbacks
                    ):
                        idle["idle_judge_fallback_providers"] = list(legacy_fallbacks)
                        changed = True

                    # This marker is written even when no legacy chain exists. That
                    # distinguishes a new user's intentional empty configuration from
                    # an old configuration that still needs one-time migration.
                    idle[provider_migration_marker] = True
                    changed = True
                    logger.info("[Spark] completed idle judge provider migration")
                self.cfg["idle_greetings"] = idle

            enhancement = self.cfg.get("enhancement") or {}
            if isinstance(enhancement, dict):
                if "mode" not in enhancement:
                    enhancement["mode"] = "对话热度"
                    changed = True
                if "fixed_delay_seconds" not in enhancement:
                    old_fixed = enhancement.get("enhancement_min_delay")
                    if old_fixed not in (None, ""):
                        enhancement["fixed_delay_seconds"] = old_fixed
                        changed = True
                if "fixed_random_fluctuation_seconds" not in enhancement:
                    old_jitter = enhancement.get(
                        "random_fluctuation_seconds",
                        enhancement.get("jitter_seconds"),
                    )
                    enhancement["fixed_random_fluctuation_seconds"] = (
                        old_jitter if old_jitter not in (None, "") else "0"
                    )
                    changed = True
                if "ignore_judge" not in enhancement:
                    enhancement["ignore_judge"] = False
                    changed = True
                self.cfg["enhancement"] = enhancement
            if advanced.get("fixed_provider") and not proactive.get("fixed_provider"):
                proactive["fixed_provider"] = advanced["fixed_provider"]
                changed = True
                logger.info(
                    "[Spark] migrated advanced.fixed_provider -> proactive_settings.fixed_provider"
                )
            if advanced.get("history_depth") and not proactive.get("history_depth"):
                proactive["history_depth"] = advanced["history_depth"]
                changed = True
                logger.info(
                    "[Spark] migrated advanced.history_depth -> proactive_settings.history_depth"
                )
            if advanced.get("persona_override") and not proactive.get("gen_persona_id"):
                proactive["persona_override_legacy"] = advanced["persona_override"]
                changed = True

            # special.provider -> proactive_settings.fixed_provider
            special = self.cfg.get("special")
            if isinstance(special, dict) and special.get("provider"):
                if not proactive.get("fixed_provider"):
                    proactive["fixed_provider"] = special["provider"]
                    changed = True
                    logger.info(
                        "[Spark] migrated special.provider -> proactive_settings.fixed_provider"
                    )

            # basic_settings.fixed_provider -> proactive_settings.fixed_provider
            if basic.get("fixed_provider") and not proactive.get("fixed_provider"):
                proactive["fixed_provider"] = basic["fixed_provider"]
                changed = True
                logger.info(
                    "[Spark] migrated basic_settings.fixed_provider -> proactive_settings.fixed_provider"
                )

            if changed:
                self.cfg["proactive_settings"] = proactive
                self.cfg.save_config()
        except Exception as e:
            logger.debug(f"[Spark] config migration: {e}")

    async def initialize(self):
        """插件激活时的初始化方法（框架生命周期）"""
        self.context._spark_get_proactive_state = self._get_proactive_state
        # 启动后台调度器
        self._loop_task = asyncio.create_task(self._scheduler_loop())
        logger.info("[Spark] Scheduler started.")

        # Agent 订阅工具：仅在 agent 模式下激活
        if HAS_LLM_TOOL:
            mode = self._get_cfg("basic_settings", "subscribe_mode") or "manual"
            if mode == "agent":
                self.context.activate_llm_tool("conversa_subscribe")
                logger.info("[Spark] Agent 订阅工具已激活")
            else:
                try:
                    self.context.deactivate_llm_tool("conversa_subscribe")
                except Exception:
                    pass  # 工具可能未注册，忽略

    def _is_admin(self, event: AstrMessageEvent) -> bool:
        """检查事件发送者是否为AstrBot管理员"""
        return event.role == "admin"

    def _get_cfg(self, group_key: str, sub_key: str, default=None):
        group = self.cfg.get(group_key)
        if not isinstance(group, dict):
            return default
        return group.get(sub_key, default)

    def _get_proactive_state(self, session_id: str) -> dict:
        """Expose a bounded, read-only delivery/reply snapshot to collaborating plugins."""
        state = self._states.get(str(session_id))
        if state is None:
            return {
                "session_id": str(session_id),
                "available": False,
                "schema_version": EVIDENCE_SCHEMA_VERSION,
                "evidence": [],
                "last_proactive_reply_ts": 0.0,
                "last_user_reply_ts": 0.0,
                "awaiting_user_reply": False,
            }
        proactive_ts = max(
            0.0, float(getattr(state, "last_proactive_reply_ts", 0.0) or 0.0)
        )
        user_ts = max(0.0, float(getattr(state, "last_user_reply_ts", 0.0) or 0.0))
        evidence = normalize_evidence_records(getattr(state, "proactive_evidence", []))
        return {
            "session_id": str(session_id),
            "available": True,
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "evidence": [dict(item) for item in evidence],
            "last_proactive_reply_ts": proactive_ts,
            "last_user_reply_ts": user_ts,
            "awaiting_user_reply": any(
                item["reply_status"] == "pending" for item in evidence
            )
            or proactive_ts > user_ts,
        }

    @staticmethod
    def _acknowledge_proactive_evidence(
        state: SessionState, *, reply_at: float, reply_text: object
    ) -> list[str]:
        state.proactive_evidence, acknowledged = acknowledge_pending_evidence(
            state.proactive_evidence,
            reply_at=reply_at,
            reply_text=reply_text,
        )
        return acknowledged

    def _record_proactive_delivery(
        self,
        umo: str,
        *,
        source: str,
        sent_at: float,
        response_text: object,
    ) -> dict:
        state = self._states.setdefault(umo, SessionState())
        state.proactive_evidence, record = record_proactive_delivery(
            state.proactive_evidence,
            source=source,
            sent_at=sent_at,
            proactive_text=response_text,
        )
        state.last_ts = sent_at
        state.last_proactive_reply_ts = sent_at
        profile = self._user_profiles.get(umo)
        if profile:
            # A successful proactive message starts a fresh idle cycle. This also
            # invalidates a different pending candidate for the same session.
            self._reset_idle_schedule(state, sent_at, profile)
        return record

    def _get_int_cfg(self, group_key: str, sub_key: str, default: int) -> int:
        value = self._get_cfg(group_key, sub_key, None)
        if value is None or value == "":
            return int(default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    def _get_heat_args(self) -> tuple[float, float, float, float]:
        heat_cfg = self.cfg.get("heat_settings") or {}
        if not isinstance(heat_cfg, dict):
            heat_cfg = {}
        short_window_minutes = heat_cfg.get("heat_window_minutes")
        if short_window_minutes is None:
            short_window_minutes = (
                float(heat_cfg.get("heat_window_hours", 4) or 4) * 60.0
            )
        long_window_minutes = heat_cfg.get("heat_long_window_minutes", 720)
        full_score_messages = float(
            heat_cfg.get("heat_messages_for_full_score", 10) or 10
        )
        short_weight = float(heat_cfg.get("heat_short_weight", 0.7) or 0.7)
        return (
            max(float(short_window_minutes or 1), 1.0),
            max(float(long_window_minutes or 1), 1.0),
            max(full_score_messages, 1.0),
            min(max(short_weight, 0.0), 1.0),
        )

    def _calc_heat(self, st: SessionState, now_ts: float) -> float:
        short_window_m, long_window_m, full_score_messages, short_weight = (
            self._get_heat_args()
        )
        return dual_scale_heat(
            st.msg_timestamps or [],
            now_ts,
            short_window_minutes=short_window_m,
            long_window_minutes=long_window_m,
            messages_for_full_score=full_score_messages,
            short_weight=short_weight,
        )

    def _idle_mode(self) -> str:
        mode = str(self._get_cfg("idle_greetings", "mode", "") or "").strip()
        if mode in {"对话热度", "固定时间", "大模型判断"}:
            return mode
        return (
            "对话热度"
            if bool(self._get_cfg("heat_settings", "enable_heat", True))
            else "固定时间"
        )

    def _idle_judge_after_minutes(self) -> int:
        return max(1, self._get_int_cfg("idle_greetings", "judge_after_minutes", 60))

    def _latest_chat_ts(self, st: SessionState) -> float:
        return max(
            float(st.last_user_reply_ts or 0),
            float(st.last_ai_reply_ts or 0),
            float(st.last_proactive_reply_ts or 0),
        )

    def _calc_idle_delay(
        self, st: SessionState, now_ts: float, profile: UserProfile
    ) -> float:
        """Calculate the baseline delay for the selected idle greeting mode."""
        mode = self._idle_mode()
        if mode == "大模型判断":
            return float(self._idle_judge_after_minutes())
        if mode == "固定时间":
            if profile.idle_after_minutes is not None:
                return max(1.0, float(profile.idle_after_minutes))
            return max(
                1.0,
                float(
                    self._get_cfg("idle_greetings", "idle_after_minutes", 1200) or 1200
                ),
            )

        hot_m = float(self._get_cfg("idle_greetings", "hot_delay_minutes", 30) or 30)
        cold_m = float(
            self._get_cfg("idle_greetings", "cold_delay_minutes", 200) or 200
        )
        heat = self._calc_heat(st, now_ts)
        delay_m = geometric_delay(hot_m, cold_m, heat)
        logger.debug(f"[Spark] 热度计算: heat={heat:.2f}, delay={delay_m:.0f}m")
        return delay_m

    def _calc_fluctuated_idle_delay(
        self, st: SessionState, now_ts: float, profile: UserProfile
    ) -> float | None:
        """Apply fixed-mode jitter while leaving model-mode timing unjittered."""
        delay_m = self._calc_idle_delay(st, now_ts, profile)
        if self._idle_mode() != "固定时间":
            return delay_m

        idle_cfg = self.cfg.get("idle_greetings") or {}
        if not isinstance(idle_cfg, dict):
            idle_cfg = {}
        policy_config = dict(idle_cfg)
        compact_value = idle_cfg.get("idle_random_fluctuation_minutes", 30)
        if (
            idle_cfg.get("offset_mode") in (None, "")
            and str(compact_value).strip().isdigit()
        ):
            policy_config["idle_random_fluctuation_minutes"] = min(
                max(0, int(compact_value or 0)), max(0, int(delay_m) - 1)
            )
        policy = parse_policy(
            policy_config,
            legacy_jitter_key="idle_random_fluctuation_minutes",
        )
        result = apply_delay_policy(
            delay_m,
            policy,
            seed=f"idle:{now_ts:.0f}:{len(st.msg_timestamps or [])}",
        )
        if result.is_valid:
            st.idle_retry_after_ts = 0.0
            return result.minutes
        if result.retryable:
            retry_minutes = max(
                1, self._get_int_cfg("idle_greetings", "offset_retry_minutes", 1)
            )
            st.idle_retry_after_ts = now_ts + retry_minutes * 60
            logger.info(f"[Spark] 延时问候随机偏移结果无效，{retry_minutes} 分钟后重算")
        else:
            st.idle_retry_after_ts = -1.0
            logger.warning("[Spark] 延时问候固定偏移结果小于等于 0，本轮不安排")
        return None

    def _reset_idle_schedule(
        self, st: SessionState, now_ts: float, profile: UserProfile
    ) -> None:
        """Start a new idle cycle after any real user or AI activity."""
        st.idle_judge_cycle += 1
        st.idle_judge_checked_cycle = -1
        st.idle_judge_inflight_cycle = -1
        st.idle_judge_task_ts = 0.0
        st.idle_judge_anchor_ts = now_ts
        st.idle_retry_after_ts = 0.0
        st.idle_schedule_mode = self._idle_mode()
        if not profile.subscribed or not bool(
            self._get_cfg("idle_greetings", "enable_idle_greetings", True)
        ):
            st.next_idle_ts = 0.0
            return
        delay_m = self._calc_fluctuated_idle_delay(st, now_ts, profile)
        st.next_idle_ts = now_ts + delay_m * 60 if delay_m else 0.0

    def _idle_judge_bounds(self) -> tuple[int, int]:
        minimum = max(
            1, self._get_int_cfg("idle_greetings", "judge_min_delay_minutes", 5)
        )
        maximum = max(
            minimum,
            self._get_int_cfg("idle_greetings", "judge_max_delay_minutes", 1440),
        )
        return minimum, maximum

    @staticmethod
    def _parse_delay_minutes(
        response: object, minimum: int, maximum: int
    ) -> int | None:
        text = str(response or "").strip()
        if not re.fullmatch(r"[0-9]+", text):
            return None
        value = int(text)
        if value == 0:
            return 0
        if value < minimum or value > maximum:
            return None
        return value

    # 数据持久化
    def _load_user_data(self):
        """Load user profiles and ignore deprecated top-level fields."""
        if not os.path.exists(self._user_data_path):
            return
        try:
            with open(self._user_data_path, encoding="utf-8") as f:
                data = json.load(f)

                profiles_data = data.get("profiles", {})
                for user_id, profile_dict in profiles_data.items():
                    self._user_profiles[user_id] = UserProfile.from_dict(profile_dict)
                logger.debug(
                    f"[Spark] Loaded {len(self._user_profiles)} user profiles."
                )

        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"[Spark] Failed to load user data: {e}")
        except OSError as e:
            logger.error(f"[Spark] Failed to read user data file: {e}")

    def _save_user_data(self):
        """Save user profiles without serializing deprecated reminder data."""
        try:
            profiles_dict = {
                uid: profile.to_dict() for uid, profile in self._user_profiles.items()
            }
            data = {"profiles": profiles_dict}
            with open(self._user_data_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        except OSError as e:
            logger.error(f"[Spark] Failed to write user data file: {e}")
        except (TypeError, ValueError) as e:
            logger.error(f"[Spark] Failed to serialize user data: {e}")

    def _load_session_data(self):
        """加载运行时状态（从 session_data.json）"""
        if not os.path.exists(self._session_data_path):
            return
        try:
            with open(self._session_data_path, encoding="utf-8") as f:
                data = json.load(f)

                states_data = data.get("states", {})
                for conv_id, state_dict in states_data.items():
                    self._states[conv_id] = SessionState.from_dict(state_dict)
                logger.debug(f"[Spark] Loaded {len(self._states)} session states.")

        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"[Spark] Failed to load session data: {e}")
        except OSError as e:
            logger.error(f"[Spark] Failed to read session data file: {e}")

    def _save_session_data(self):
        """保存运行时状态（到 session_data.json）"""
        try:
            states_dict = {cid: state.to_dict() for cid, state in self._states.items()}
            data = {
                "schema_version": SESSION_DATA_SCHEMA_VERSION,
                "states": states_dict,
            }
            with open(self._session_data_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        except OSError as e:
            logger.error(f"[Spark] Failed to write session data file: {e}")
        except (TypeError, ValueError) as e:
            logger.error(f"[Spark] Failed to serialize session data: {e}")

    async def _debounced_save_user_data(self):
        """
        去抖保存用户数据：在最后一次调用后的指定延迟后执行一次保存
        避免高频消息时的频繁磁盘I/O
        """
        # 取消之前的保存任务（如果存在）
        if self._save_user_data_task and not self._save_user_data_task.done():
            self._save_user_data_task.cancel()

        async def delayed_save():
            try:
                await asyncio.sleep(self._save_delay_seconds)
                self._save_user_data()
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"[Spark] Debounced save user data failed: {e}")

        # 创建新的延迟保存任务
        self._save_user_data_task = asyncio.create_task(delayed_save())

    async def _debounced_save_session_data(self):
        """
        去抖保存会话数据：在最后一次调用后的指定延迟后执行一次保存
        避免高频消息时的频繁磁盘I/O
        """
        # 取消之前的保存任务（如果存在）
        if self._save_session_data_task and not self._save_session_data_task.done():
            self._save_session_data_task.cancel()

        async def delayed_save():
            try:
                await asyncio.sleep(self._save_delay_seconds)
                self._save_session_data()
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"[Spark] Debounced save session data failed: {e}")

        # 创建新的延迟保存任务
        self._save_session_data_task = asyncio.create_task(delayed_save())

    def _sync_subscribed_users_from_config(self, silent: bool = False):
        """
        从配置文件同步订阅用户列表到内部状态

        Args:
            silent: 是否静默模式（不打印日志，仅在状态变化时打印）
        """
        try:
            config_subscribed_ids = (
                self._get_cfg("basic_settings", "subscribed_users") or []
            )
            if not isinstance(config_subscribed_ids, list):
                logger.warning(f"[Spark] subscribed_users 配置格式错误，应为列表")  # noqa: F541
                return

            # 记录变化
            changes = {"added": [], "removed": []}

            # 同步所有用户的订阅状态（包括设置为 True 和 False）
            for user_id, profile in self._user_profiles.items():
                if user_id in config_subscribed_ids:
                    if not profile.subscribed:
                        profile.subscribed = True
                        profile.manual_unsubscribe = False  # 清除手动退订标记
                        profile.auto_unsubscribed = False  # 清除自动退订标记
                        changes["added"].append(user_id)
                        if not silent:
                            logger.debug(f"[Spark] 从配置同步订阅状态(启用): {user_id}")
                else:
                    # 如果用户不在配置列表中，设置为未订阅（来自 WebUI 的手动退订）
                    if profile.subscribed:
                        profile.subscribed = False
                        profile.manual_unsubscribe = (
                            True  # 标记为手动退订（WebUI操作视为手动）
                        )
                        profile.auto_unsubscribed = False  # 清除自动退订标记
                        changes["removed"].append(user_id)
                        if not silent:
                            logger.debug(f"[Spark] 从配置同步订阅状态(禁用): {user_id}")

            # 只在有变化或非静默模式时打印信息
            if not silent or changes["added"] or changes["removed"]:
                if changes["added"]:
                    logger.info(f"[Spark] 配置热重载：新增订阅 {changes['added']}")
                if changes["removed"]:
                    logger.info(f"[Spark] 配置热重载：取消订阅 {changes['removed']}")

                if not silent and not changes["added"] and not changes["removed"]:
                    logger.debug(
                        f"[Spark] 已从配置同步 {len(config_subscribed_ids)} 个订阅用户ID"
                    )
                    subscribed_sessions = [
                        user_id
                        for user_id, profile in self._user_profiles.items()
                        if profile.subscribed
                    ]
                    logger.debug(
                        f"[Spark] 当前已订阅的会话数: {len(subscribed_sessions)}"
                    )

        except Exception as e:
            logger.error(f"[Spark] 同步订阅用户配置失败: {e}")

    def _sync_subscribed_users_to_config(self):
        """将插件内部订阅状态同步回配置文件"""
        try:
            subscribed_users = []
            for user_id, profile in self._user_profiles.items():
                if profile.subscribed:
                    subscribed_users.append(user_id)

            # 直接更新配置
            if "basic_settings" not in self.cfg:
                self.cfg["basic_settings"] = {}
            self.cfg["basic_settings"]["subscribed_users"] = subscribed_users
            self.cfg.save_config()
            logger.debug(f"[Spark] 已同步 {len(subscribed_users)} 个订阅用户到配置文件")
        except Exception as e:
            logger.error(f"[Spark] 同步订阅用户到配置失败: {e}")

    def _save_user_profiles(self):
        """兼容旧API，实际调用整合后的保存函数"""
        self._save_user_data()

    # 事件处理

    @filter.event_message_type(filter.EventMessageType.ALL, priority=30)
    async def _on_any_message(self, event: AstrMessageEvent):
        """
        监听所有消息事件

        功能：
        1. 更新会话的最后活跃时间戳
        2. 更新用户最后回复时间（用于自动退订检测）
        3. 重置连续无回复计数器
        4. 自动订阅模式下自动订阅新会话
        5. 计算下一次延时问候触发时间
        """
        umo = event.unified_msg_origin

        # 初始化数据结构
        if umo not in self._states:
            self._states[umo] = SessionState()
        if umo not in self._user_profiles:
            self._user_profiles[umo] = UserProfile()

        st = self._states[umo]
        profile = self._user_profiles[umo]

        # Detect commands from the untouched source message chain because AstrBot
        # removes the wake prefix from event.message_str before plugin hooks run.
        message_text = (
            event.message_str.strip()
            if hasattr(event, "message_str") and event.message_str
            else ""
        )
        is_slash_input = _is_slash_prefixed_event(event)
        is_real_message = bool(message_text) and not is_slash_input

        # Enhancement task cancellation is handled inside _delayed_enhancement
        # via last_user_reply_ts check. Do NOT cancel here — it would kill tasks
        # that are already past the sleep phase and executing LLM calls.

        # 保存旧的 last_user_reply_ts 用于判断是否是老用户
        old_last_user_reply_ts = st.last_user_reply_ts

        # 更新时间戳
        now_ts = _now_tz(
            self._get_cfg("basic_settings", "timezone") or None
        ).timestamp()
        if is_real_message:
            st.last_ts = now_ts
            st.last_user_reply_ts = now_ts
            acknowledged = self._acknowledge_proactive_evidence(
                st,
                reply_at=now_ts,
                reply_text=message_text,
            )
            if acknowledged:
                logger.debug(
                    f"[Spark] 主动证据记录首次用户回复: {umo}, count={len(acknowledged)}"
                )
            if not event.get_extra(self._heat_event_marker, False):
                if st.msg_timestamps is None:
                    st.msg_timestamps = []
                st.msg_timestamps.append(now_ts)
                if len(st.msg_timestamps) > 100:
                    st.msg_timestamps = st.msg_timestamps[-100:]
                event.set_extra(self._heat_event_marker, True)
            st.consecutive_no_reply_count = 0

        # 自动订阅模式：仅在首次创建用户且收到真实消息时自动订阅
        if (
            is_real_message
            and (self._get_cfg("basic_settings", "subscribe_mode") or "manual")
            == "auto"
        ):
            # 只在用户第一次发消息时（old_last_user_reply_ts == 0）自动订阅
            if old_last_user_reply_ts == 0 and not profile.manual_unsubscribe:
                profile.subscribed = True
                profile.auto_unsubscribed = False  # 清除自动退订标记
                logger.info(f"[Spark] 自动订阅模式：新用户 {umo} 已自动订阅")
                self._sync_subscribed_users_to_config()  # 同步到配置文件

        # 自动重新激活：仅对"被自动退订"的用户生效，手动退订的用户不会被自动重新激活
        if (
            is_real_message
            and not profile.subscribed
            and profile.auto_unsubscribed
            and not profile.manual_unsubscribe
        ):
            auto_resubscribe = bool(
                self._get_cfg("basic_settings", "auto_resubscribe", True)
            )
            if auto_resubscribe:
                # 用户主动发消息，重新激活订阅
                profile.subscribed = True
                profile.auto_unsubscribed = False  # 清除自动退订标记
                logger.info(
                    f"[Spark] 自动重新激活订阅: {umo} (用户在自动退订后主动聊天)"
                )
                self._sync_subscribed_users_to_config()  # 同步到配置文件

        # 真实聊天消息会使等待中的模型候选任务失效，并重新进入当前模式周期。
        if is_real_message:
            try:
                self._reset_idle_schedule(st, now_ts, profile)
                if st.next_idle_ts:
                    logger.debug(
                        f"[Spark] 沉寂计时刷新(消息): {umo}, "
                        f"mode={st.idle_schedule_mode}, next={st.next_idle_ts:.0f}"
                    )
            except Exception as e:
                logger.warning(f"[Spark] 计算 next_idle_ts 失败: {e}")

        # 保存状态（使用去抖机制，减少高频磁盘I/O）
        await self._debounced_save_session_data()
        await self._debounced_save_user_data()

    @filter.on_llm_request()
    async def _on_llm_request_update_ts(self, event: AstrMessageEvent, req=None):
        """补偿时间戳：当 chat_merger 等插件吃掉第一次事件后，_on_any_message 不会被调用。
        在 LLM 请求阶段补充更新 last_user_reply_ts 和 msg_timestamps，保证沉寂计时正常工作。
        """
        try:
            if HAS_AGENT_PIPELINE and isinstance(event, CronMessageEvent):
                return
            if _is_slash_prefixed_event(event):
                return
            umo = event.unified_msg_origin
            st = self._states.get(umo)
            if not st:
                return
            if event.get_extra(self._heat_event_marker, False):
                return
            now_ts = _now_tz(
                self._get_cfg("basic_settings", "timezone") or None
            ).timestamp()
            st.last_user_reply_ts = now_ts
            acknowledged = self._acknowledge_proactive_evidence(
                st,
                reply_at=now_ts,
                reply_text=str(getattr(event, "message_str", "") or ""),
            )
            if acknowledged:
                logger.debug(
                    f"[Spark] 主动证据在 LLM 请求补偿入口记录回复: "
                    f"{umo}, count={len(acknowledged)}"
                )
            if st.msg_timestamps is None:
                st.msg_timestamps = []
            st.msg_timestamps.append(now_ts)
            if len(st.msg_timestamps) > 100:
                st.msg_timestamps = st.msg_timestamps[-100:]
            event.set_extra(self._heat_event_marker, True)
            profile = self._user_profiles.get(umo)
            if profile:
                self._reset_idle_schedule(st, now_ts, profile)
                if st.next_idle_ts:
                    logger.debug(
                        f"[Spark] 沉寂计时刷新(llm_request补偿): {umo}, "
                        f"mode={st.idle_schedule_mode}, next={st.next_idle_ts:.0f}"
                    )
            await self._debounced_save_session_data()
        except Exception as e:
            logger.debug(f"[Spark] _on_llm_request_update_ts 异常: {e}")

    @filter.on_llm_response()
    async def _on_llm_response_enhancement(
        self, event: AstrMessageEvent, _response=None
    ):
        """对话增强：LLM 回复后检查是否应触发短期追回复"""
        try:
            # Skip proactive replies triggered by this plugin itself (CronMessageEvent)
            if HAS_AGENT_PIPELINE and isinstance(event, CronMessageEvent):
                return
            if _is_slash_prefixed_event(event):
                return
            umo = event.unified_msg_origin
            st = self._states.get(umo)
            now_ts = _now_tz(
                self._get_cfg("basic_settings", "timezone") or None
            ).timestamp()
            if st:
                st.last_ai_reply_ts = now_ts
                profile = self._user_profiles.get(umo)
                if profile:
                    self._reset_idle_schedule(st, now_ts, profile)
                    if st.next_idle_ts:
                        logger.debug(
                            f"[Spark] 沉寂计时刷新(AI回复): {umo}, "
                            f"mode={st.idle_schedule_mode}, next={st.next_idle_ts:.0f}"
                        )
                await self._debounced_save_session_data()
            enhancement_enabled = bool(
                self._get_cfg("enhancement", "enable_enhancement", False)
            )
            if not enhancement_enabled:
                old_task = self._enhancement_tasks.pop(umo, None)
                if old_task and not old_task.done():
                    old_task.cancel()
                self._enhancement_gen[umo] = self._enhancement_gen.get(umo, 0) + 1
                if st:
                    st.next_enhancement_ts = 0.0
                return

            # Cancel any pending (sleeping) enhancement task so each LLM turn re-evaluates.
            old_task = self._enhancement_tasks.get(umo)
            if old_task and not old_task.done():
                old_task.cancel()
                self._enhancement_tasks.pop(umo, None)
            # Bump generation so any already-awake old task knows it's been superseded.
            self._enhancement_gen[umo] = self._enhancement_gen.get(umo, 0) + 1

            if self._should_trigger_enhancement(umo):
                current_user = str(getattr(event, "message_str", "") or "").strip()
                current_ai = self._extract_history_text(
                    getattr(_response, "completion_text", "")
                )
                current_round = None
                if current_user and current_ai:
                    current_round = {
                        "user": current_user,
                        "assistant": current_ai,
                    }
                self._schedule_enhancement(umo, current_round=current_round)
        except Exception as e:
            logger.debug(f"[Spark] 对话增强检查异常: {e}")

    # Agent 订阅工具
    @llm_tool(name="conversa_subscribe")
    async def _tool_subscribe(self, event: AstrMessageEvent, action: str):
        """管理主动对话功能。当用户希望你能主动找他聊天、保持联系时开启；当用户明确不需要时关闭。

        Args:
            action(string): "on" 开启主动对话, "off" 关闭主动对话
        """
        # 运行时检查：仅在 agent 模式下工作
        mode = self._get_cfg("basic_settings", "subscribe_mode") or "manual"
        if mode != "agent":
            return "主动对话的订阅方式当前不是 agent 模式，无法通过工具操作。"

        umo = event.unified_msg_origin
        if umo not in self._user_profiles:
            self._user_profiles[umo] = UserProfile()
        profile = self._user_profiles[umo]

        if action == "on":
            profile.subscribed = True
            profile.manual_unsubscribe = False
            profile.auto_unsubscribed = False
            logger.info(f"[Spark] Agent 工具订阅: {umo}")
            self._save_user_data()
            self._sync_subscribed_users_to_config()
            return "已开启主动对话订阅，我会在合适的时候主动找你聊天。"
        elif action == "off":
            profile.subscribed = False
            profile.manual_unsubscribe = True
            profile.auto_unsubscribed = False
            logger.info(f"[Spark] Agent 工具退订: {umo}")
            self._save_user_data()
            self._sync_subscribed_users_to_config()
            return "已关闭主动对话订阅，我不会再主动发起聊天了。"
        else:
            return f"无效的操作 '{action}'，请使用 'on' 或 'off'。"

    async def _cmd_conversa(self, event: AstrMessageEvent):
        """Internal implementation shared by /灵犀 and legacy /conversa."""
        text = (event.message_str or "").strip()

        # 动态处理主命令和别名
        command_parts = text.lstrip("/").split()
        if not command_parts:
            return

        # 提取真实命令和参数
        args_str = " ".join(command_parts[1:]) if len(command_parts) > 1 else ""

        # 将参数字符串分割成子命令和值
        args = args_str.split()
        sub_command = args[0] if args else ""

        # Chinese-to-English sub-command aliases
        _sub_alias = {
            "订阅": "watch",
            "退订": "unwatch",
            "开启": "on",
            "关闭": "off",
            "设置": "set",
            "帮助": "help",
        }
        sub_command = _sub_alias.get(sub_command, sub_command)

        # Chinese target aliases for "set" sub-command
        if sub_command == "set" and len(args) >= 2:
            _target_alias = {
                "免打扰": "quiet",
                "沉寂": "after",
            }
            if args[1] in _target_alias:
                args[1] = _target_alias[args[1]]
            elif args[1].startswith("定时"):
                # 定时1 → daily1, 定时2 → daily2, etc.
                m = re.match(r"定时(\d+)", args[1])
                if m:
                    args[1] = f"daily{m.group(1)}"

        def reply(msg: str):
            return event.plain_result(msg)

        # 帮助信息
        if not sub_command or sub_command == "help":
            yield reply(self._help_text())
            return

        # 调试信息
        if sub_command == "debug":
            debug_info = [
                f"插件启用状态: {self.cfg.get('enable', True)}",
                f"订阅模式: {self._get_cfg('basic_settings', 'subscribe_mode', 'manual')}",
                f"当前用户: {event.unified_msg_origin}",
            ]
            umo = event.unified_msg_origin
            if umo not in self._states:
                self._states[umo] = SessionState()
            profile = self._user_profiles.get(umo)
            debug_info.append(
                f"用户订阅状态: {profile.subscribed if profile else False}"
            )

            # 显示订阅/退订状态标记
            if profile:
                if profile.manual_unsubscribe:
                    debug_info.append("退订类型: 手动退订（强制，不会自动重新激活）")
                elif profile.auto_unsubscribed:
                    debug_info.append("退订类型: 自动退订（可自动重新激活）")
                elif profile.subscribed:
                    debug_info.append("订阅类型: 正常订阅")

            debug_info.append(
                f"用户专属免打扰: {profile.quiet_hours if profile and profile.quiet_hours else '未设置(使用全局)'}"
            )
            debug_info.append(
                f"全局免打扰时间: {self._get_cfg('basic_settings', 'quiet_hours', '未设置')}"
            )
            debug_info.append(
                f"延时基准: {self._get_cfg('idle_greetings', 'idle_after_minutes', 0)}分钟"
            )
            debug_info.append(
                f"最大无回复天数: {self._get_cfg('basic_settings', 'max_no_reply_days', 0)}"
            )
            debug_info.append(
                f"自动重新激活: {bool(self._get_cfg('basic_settings', 'auto_resubscribe', True))}"
            )

            # Heat debug info
            st_debug = self._states.get(umo)
            heat_enabled = bool(self._get_cfg("heat_settings", "enable_heat", True))
            if heat_enabled and st_debug:
                _short_window_m, _long_window_m, _full_score_messages, _short_weight = (
                    self._get_heat_args()
                )
                _heat_val = self._calc_heat(
                    st_debug,
                    _now_tz(
                        self._get_cfg("basic_settings", "timezone") or None
                    ).timestamp(),
                )
                _hot_m = float(
                    self._get_cfg("heat_settings", "hot_delay_minutes", 30) or 30
                )
                _cold_m = float(
                    self._get_cfg("heat_settings", "cold_delay_minutes", 200) or 200
                )
                _next_delay = geometric_delay(_hot_m, _cold_m, _heat_val)
                if _heat_val >= 0.6:
                    _heat_label = "热"
                elif _heat_val >= 0.2:
                    _heat_label = "温"
                else:
                    _heat_label = "冷"
                debug_info.append(
                    f"对话热度: {_heat_label}({_heat_val:.2f}) → 下次触发延迟约 {_next_delay:.0f} 分钟"
                )
                debug_info.append(
                    "热度窗口: "
                    f"短期 {int(_short_window_m)} 分钟（权重 {_short_weight:.0%}），"
                    f"长期 {int(_long_window_m)} 分钟，满热约需 "
                    f"{_full_score_messages:.0f} 条消息，记录消息数: "
                    f"{len(st_debug.msg_timestamps or [])}"
                )
            elif not heat_enabled:
                debug_info.append("对话热度: 已关闭（使用固定 idle_after_minutes）")

            yield reply("🔍 调试信息:\n" + "\n".join(debug_info))
            return

        # 启用/停用插件
        if sub_command == "on":
            if not self._is_admin(event):
                yield event.plain_result("错误：此命令仅限管理员使用。")
                return
            self.cfg["enable"] = True
            self.cfg["basic_settings"] = self.cfg.get("basic_settings") or {}
            self.cfg["basic_settings"]["enable"] = True
            self.cfg.save_config()
            yield reply("✅ 已启用灵犀")
            return

        if sub_command == "off":
            if not self._is_admin(event):
                yield event.plain_result("错误：此命令仅限管理员使用。")
                return
            self.cfg["enable"] = False
            self.cfg["basic_settings"] = self.cfg.get("basic_settings") or {}
            self.cfg["basic_settings"]["enable"] = False
            self.cfg.save_config()
            yield reply("🛑 已停用灵犀")
            return

        # 订阅/退订
        if sub_command == "watch":
            umo = event.unified_msg_origin
            if umo not in self._user_profiles:
                self._user_profiles[umo] = UserProfile()
            profile = self._user_profiles[umo]
            profile.subscribed = True
            profile.manual_unsubscribe = False  # 清除手动退订标记
            profile.auto_unsubscribed = False  # 清除自动退订标记
            logger.info(f"[Spark] 用户执行 watch 命令: {umo}")
            self._save_user_data()
            self._sync_subscribed_users_to_config()
            yield reply("📌 已订阅当前会话")
            return

        if sub_command == "unwatch":
            umo = event.unified_msg_origin
            if umo not in self._user_profiles:
                self._user_profiles[umo] = UserProfile()
            profile = self._user_profiles[umo]
            profile.subscribed = False
            profile.manual_unsubscribe = True  # 设置手动退订标记（强开关）
            profile.auto_unsubscribed = False  # 清除自动退订标记
            logger.info(f"[Spark] 用户执行 unwatch 命令（手动退订）: {umo}")
            self._save_user_data()
            self._sync_subscribed_users_to_config()
            yield reply("📭 已退订当前会话")
            return

        # 设置命令
        if sub_command == "set":
            if len(args) < 3:
                yield reply("❌ 参数不足。用法: /conversa set <目标> <值>")
                return

            target = args[1].lower()
            value = args[2]

            if target == "after":
                umo = event.unified_msg_origin
                profile = self._user_profiles.get(umo)
                if not profile:
                    self._user_profiles[umo] = UserProfile()
                    profile = self._user_profiles[umo]

                try:
                    hours = float(value)
                    if hours >= 0.5:
                        minutes = int(hours * 60)
                        profile.idle_after_minutes = minutes

                        # 立即更新 next_idle_ts，使设置立即生效
                        if umo not in self._states:
                            self._states[umo] = SessionState()
                        st = self._states[umo]
                        tz = self._get_cfg("basic_settings", "timezone") or None
                        now_ts = _now_tz(tz).timestamp()
                        st.next_idle_ts = now_ts + minutes * 60

                        self._save_user_data()
                        await self._debounced_save_session_data()
                        yield reply(f"⏱️ 已为您设置专属延时问候：{hours} 小时后触发")
                    else:
                        yield reply("⏱️ 延时问候的小时数不能少于 0.5 (30分钟)。")
                except ValueError:
                    yield reply("⏱️ 请输入有效的小时数 (例如 1, 1.5, 2)。")
                return

            elif target.startswith("daily"):
                match = re.match(r"daily(\d+)", target)
                if match:
                    n = int(match.group(1))
                    time_val = value
                    if not _parse_hhmm(time_val):
                        yield reply("❌ 时间格式错误，请使用 HH:MM 格式。")
                        return

                    daily = self.cfg.get("daily_prompts") or {}
                    if not isinstance(daily, dict):
                        daily = {}
                    greetings = daily.get("daily_greetings", [])
                    if not isinstance(greetings, list):
                        greetings = []

                    # Index is 0-based (daily1 -> index 0)
                    idx = n - 1
                    while len(greetings) <= idx:
                        greetings.append(
                            {
                                "enable": False,
                                "time": "",
                                "prompt": "",
                                "ignore_dnd": False,
                            }
                        )
                    greetings[idx]["time"] = time_val
                    greetings[idx]["enable"] = True

                    daily["daily_greetings"] = greetings
                    self.cfg["daily_prompts"] = daily
                    self.cfg.save_config()
                    yield reply(f"🗓️ 已设置每日问候 {n}：{time_val}")
                else:
                    yield reply("❌ 无效的 daily 目标。用法: /灵犀 设置 定时N <HH:MM>")
                return

            elif target == "quiet":
                # 用户可以设置自己的免打扰时间，管理员设置全局
                if re.match(r"^\d{1,2}:\d{2}-\d{1,2}:\d{2}$", value):
                    umo = event.unified_msg_origin

                    # 检查是否是管理员且想设置全局
                    if (
                        self._is_admin(event)
                        and len(args) > 3
                        and args[3].lower() == "global"
                    ):
                        # 管理员设置全局免打扰
                        settings = self.cfg.get("basic_settings") or {}
                        settings["quiet_hours"] = value
                        self.cfg["basic_settings"] = settings
                        self.cfg.save_config()
                        yield reply(f"🔕 已设置全局免打扰：{value}")
                    else:
                        # 用户设置自己的免打扰时间
                        if umo not in self._user_profiles:
                            self._user_profiles[umo] = UserProfile()
                        self._user_profiles[umo].quiet_hours = value
                        self._save_user_data()
                        yield reply(f"🔕 已为您设置专属免打扰：{value}")
                else:
                    yield reply("格式错误，请使用 HH:MM-HH:MM 格式。例如: 23:00-07:00")
                return

            elif target == "history":
                yield reply(
                    "🧵 历史条数已废弃，请使用「判断轮数」和「生成轮数」配置项替代。"
                )
                return

            yield reply(
                f"❌ 未知的 set 目标 '{target}'。可用: after, daily[1-3], quiet。"
            )
            return

        # 默认显示帮助
        yield reply(self._help_text())

    # 灵犀主命令（中文入口，逻辑与 conversa 相同）
    @filter.command("灵犀")
    async def _cmd_lingxi(self, event: AstrMessageEvent):
        """灵犀主命令入口。支持子命令：订阅/退订/开启/关闭/设置 免打扰/沉寂/定时"""
        async for r in self._cmd_conversa(event):
            yield r

    # 立即主动：跳过判断，立即触发一次主动回复
    @filter.command("立即主动")
    async def _cmd_instant_proactive(self, event: AstrMessageEvent):
        """立即触发一次沉寂问候，跳过 LLM 判断步骤。"""
        umo = event.unified_msg_origin
        profile = self._user_profiles.get(umo)
        if not profile or not profile.subscribed:
            yield event.plain_result("当前会话未订阅主动对话，请先发送 /灵犀 订阅")
            return

        idle_prompts = self._get_cfg("idle_greetings", "idle_prompt_templates") or []
        if not idle_prompts:
            yield event.plain_result(
                "未配置沉寂问候模板，请先在设置中配置 idle_prompt_templates"
            )
            return

        yield event.plain_result("正在发送主动消息...")
        tz = self._get_cfg("basic_settings", "timezone") or None
        await self._proactive_reply(
            umo,
            tz,
            random.choice(idle_prompts),
            skip_judge=True,
            slash_triggered=True,
        )

    # 主动状态：显示当前订阅和运行状态
    @filter.command("主动状态")
    async def _cmd_proactive_status(self, event: AstrMessageEvent):
        """查看当前会话的订阅状态、免打扰时段、沉寂问候设置和每日问候配置。"""
        umo = event.unified_msg_origin
        profile = self._user_profiles.get(umo)
        st = self._states.get(umo)
        tz = self._get_cfg("basic_settings", "timezone") or None
        now = _now_tz(tz)

        lines = ["--- 灵犀 · 主动对话 状态 ---"]
        facts: list[tuple[str, str]] = []
        daily_items: list[str] = []
        subscription_label = "未订阅"

        if profile:
            sub_status = "已订阅" if profile.subscribed else "未订阅"
            subscription_label = sub_status
            lines.append(f"订阅状态: {sub_status}")
            if profile.manual_unsubscribe:
                lines.append("退订类型: 手动退订")
                facts.append(("退订类型", "手动退订"))
            elif profile.auto_unsubscribed:
                lines.append("退订类型: 自动退订（可自动恢复）")
                facts.append(("退订类型", "自动退订（可自动恢复）"))
            if profile.quiet_hours:
                lines.append(f"专属免打扰: {profile.quiet_hours}")
                facts.append(("专属免打扰", profile.quiet_hours))
        else:
            lines.append("订阅状态: 未订阅")

        global_quiet = self._get_cfg("basic_settings", "quiet_hours", "") or ""
        if global_quiet:
            lines.append(f"全局免打扰: {global_quiet}")
            facts.append(("全局免打扰", global_quiet))

        is_busy = getattr(self.context, "_busy_schedule_is_busy", False)
        lines.append(f"忙碌时段: {'是' if is_busy else '否'}")

        heat_enabled = bool(self._get_cfg("heat_settings", "enable_heat", True))
        heat_label = "已关闭"
        heat_val: float | None = None
        if heat_enabled:
            short_window_m, long_window_m, _full_score_messages, short_weight = (
                self._get_heat_args()
            )
            now_ts = now.timestamp()
            heat_val = self._calc_heat(st, now_ts) if st else 0.0
            if heat_val >= 0.6:
                heat_label = "热"
            elif heat_val >= 0.2:
                heat_label = "温"
            else:
                heat_label = "冷"
            window_sec = float(short_window_m) * 60.0
            recent_msg_count = 0
            if st and st.msg_timestamps:
                recent_msg_count = sum(
                    1 for ts in st.msg_timestamps if 0 <= now_ts - ts <= window_sec
                )
            lines.append(
                "当前热度: "
                f"{heat_label}({heat_val:.2f})，短期 {int(short_window_m)} 分钟内 "
                f"{recent_msg_count} 条消息；长期余温 {int(long_window_m)} 分钟，"
                f"短期权重 {short_weight:.0%}"
            )
            facts.append(
                (
                    "当前热度",
                    f"{heat_label}({heat_val:.2f}) · 短期 {int(short_window_m)} 分钟 "
                    f"{recent_msg_count} 条 · 长期 {int(long_window_m)} 分钟 · "
                    f"权重 {short_weight:.0%}",
                )
            )
        else:
            lines.append("当前热度: 已关闭（使用固定沉寂延迟）")
            facts.append(("当前热度", "已关闭 · 使用固定沉寂延迟"))

        if st and st.last_user_reply_ts > 0:
            delta = now.timestamp() - st.last_user_reply_ts
            last_chat = _format_time_delta(delta)
            lines.append(f"距上次聊天: {last_chat}")
            facts.append(("距上次聊天", last_chat))

        judge_enabled = self._get_cfg(
            "proactive_settings", "proactive_judge_enable", True
        )
        lines.append(f"智能判断: {'开启' if judge_enabled else '关闭'}")

        # --- 待触发任务 ---
        # Each entry: (remaining_seconds, display_str); sorted ascending before render.
        # Items already elapsed or indeterminate get remaining_seconds = 0.
        pending: list[tuple[float, str]] = []
        if st and st.next_idle_ts > 0:
            remaining = st.next_idle_ts - now.timestamp()
            if remaining > 0:
                pending.append(
                    (remaining, f"  沉寂问候 → 约 {int(remaining / 60)} 分钟后")
                )
            else:
                pending.append((0.0, "  沉寂问候 → 等待触发条件"))

        if st and st.next_enhancement_ts > 0:
            remaining_enh = st.next_enhancement_ts - now.timestamp()
            if remaining_enh > 0:
                pending.append(
                    (remaining_enh, f"  对话增强 → 约 {int(remaining_enh / 60)} 分钟后")
                )

        projection = self._parse_daily_slots(now)
        visible_tasks = [
            task
            for task in projection.tasks
            if task.source_date == now.date()
            or task.target.date() == now.date()
            or (
                st
                and (raw := st.daily_task_states.get(task.tag))
                and isinstance(raw, dict)
                and raw.get("status") == "retrying"
            )
        ]
        visible_tasks.sort(key=lambda task: task.target)
        visible_issues = [
            issue for issue in projection.issues if issue.source_date == now.date()
        ]
        if visible_tasks or visible_issues:
            lines.append("相关每日问候:")

        daily_status_labels = {
            "sent": "已发送",
            "skipped_cooldown": "聊天冷却跳过",
            "skipped_dnd": "免打扰跳过",
            "skipped_busy": "忙碌跳过",
            "skipped_judge": "智能判断跳过",
            "skipped_interval": "同问候间隔跳过",
            "failed": "技术失败终止",
            "missed": "已错过",
            "legacy_processed": "已处理",
            "sending": "发送状态待确认",
        }
        for task in visible_tasks:
            state = (
                st.daily_state(task)
                if st
                else new_task_state(
                    task.tag,
                    target_at=task.target.timestamp(),
                    source_date=task.source_date.isoformat(),
                )
            )
            if state.status == "retrying":
                retry_at = datetime.fromtimestamp(state.next_retry_at, tz=now.tzinfo)
                status = f"技术失败，{retry_at.strftime('%H:%M:%S')} 重试"
                pending.append(
                    (
                        max(0.0, state.next_retry_at - now.timestamp()),
                        f"  每日问候重试 → {retry_at.strftime('%H:%M:%S')}",
                    )
                )
            else:
                status = daily_status_labels.get(state.status, "待触发")

            # Use a meaningful label instead of a sequence number
            if task.source_type == "activity" and task.activity:
                source = task.activity
            else:
                source = task.target.strftime("%H:%M")

            if task.source_type == "activity":
                boundary_label = "开始" if task.boundary == "start" else "结束"
                base_text = task.base.strftime("%m-%d %H:%M") if task.base else "未知"
                daily_text = (
                    f"{source}: {boundary_label} {base_text} → "
                    f"{task.target.strftime('%m-%d %H:%M')} · {status}"
                )
            else:
                daily_text = (
                    f"{source}: 固定时间 → {task.target.strftime('%m-%d %H:%M')} · "
                    f"{status}"
                )
            lines.append(f"  {daily_text}")
            daily_items.append(daily_text)

            if status == "待触发":
                diff_sec = max(0.0, (task.target - now).total_seconds())
                diff_min = int(diff_sec / 60)
                # Shorten activity names in pending list: extract 【tag】 labels only
                if task.source_type == "activity" and task.activity:
                    tags = re.findall(r"【([^】]+)】", task.activity)
                    short_source = "/".join(tags) if tags else task.activity[:8]
                else:
                    short_source = source
                pending.append(
                    (
                        diff_sec,
                        f"  {short_source} {task.target.strftime('%H:%M')} → 约 {diff_min} 分钟后",
                    )
                )

        issue_labels = {
            "timeline_unavailable": "时间线接口不可用",
            "timeline_error": "时间线读取失败",
            "no_schedule": "当天无日程",
            "not_matched": "未匹配",
            "invalid_boundary": "边界无效",
            "no_keywords": "未配置关键词",
            "invalid_time": "固定时间无效",
        }
        for issue in visible_issues:
            label = issue_labels.get(issue.status, issue.status)
            activity = f"{issue.activity} | " if issue.activity else ""
            daily_text = (
                f"每日问候 {issue.slot_num + 1}: {activity}{label}（{issue.detail}）"
            )
            lines.append(f"  {daily_text}")
            daily_items.append(daily_text)

        if pending:
            pending.sort(key=lambda x: x[0])
            lines.append("待触发任务:")
            lines.extend(text for _, text in pending)
        else:
            lines.append("待触发任务: 无")

        try:
            png = self._status_image_renderer.render(
                ProactiveStatusImageData(
                    subscription=subscription_label,
                    subscribed=bool(profile and profile.subscribed),
                    is_busy=is_busy,
                    judge_enabled=bool(judge_enabled),
                    heat_label=heat_label,
                    heat_value=heat_val,
                    facts=tuple(facts),
                    daily_items=tuple(daily_items),
                    pending_items=tuple(text.strip() for _, text in pending),
                ),
                now=now,
                mode=self._get_cfg(
                    "basic_settings", "status_image_theme", "自动切换"
                ),
            )
            yield event.chain_result([Image.fromBytes(png)])
        except Exception as exc:  # noqa: BLE001
            logger.exception("[Spark] Proactive status image rendering failed: %s", exc)
            yield event.plain_result("\n".join(lines))

    # 主动帮助
    @filter.command("主动帮助")
    @filter.command("灵犀帮助")
    async def _cmd_proactive_help(self, event: AstrMessageEvent):
        """显示灵犀 · 主动对话的完整帮助信息。"""
        yield event.plain_result(self._help_text())

    def _help_text(self) -> str:
        """Chinese help text."""
        return (
            "--- 灵犀 · 主动对话 帮助 ---\n"
            "让 AI 像真人一样主动找你聊天，\n"
            "通过大模型智能判断何时该开口、何时该沉默。\n\n"
            "== 基本命令 ==\n"
            "/灵犀 订阅 - 订阅主动对话，AI 会在你沉寂后主动找你\n"
            "/灵犀 退订 - 退订主动对话\n"
            "/立即主动 - 立即触发一次主动问候，跳过智能判断\n"
            "/主动状态 - 查看订阅状态、免打扰时段、每日问候等\n"
            "/主动帮助 - 显示本帮助\n\n"
            "== 管理员命令 ==\n"
            "/灵犀 开启 - 全局启用灵犀插件\n"
            "/灵犀 关闭 - 全局停用灵犀插件\n"
            "/灵犀 设置 免打扰 HH:MM-HH:MM - 设置免打扰时段\n"
            "/灵犀 设置 沉寂 <小时> - 设置沉寂多久后触发主动问候\n"
            "/灵犀 设置 定时1 HH:MM - 设置第一个每日问候时间\n"
            "/灵犀 帮助 - 显示本帮助"
        )

    # 对话增强（短期随机追回复）

    def _should_trigger_enhancement(self, umo: str) -> bool:
        """判断是否应该触发对话增强"""
        try:
            if not self.cfg.get("enable", True):
                logger.debug("[Spark] 对话增强跳过: 插件已禁用")
                return False

            enable_val = self._get_cfg("enhancement", "enable_enhancement", False)
            if not bool(enable_val):
                logger.debug(
                    f"[Spark] 对话增强跳过: enable_enhancement={enable_val} (raw cfg enhancement={self.cfg.get('enhancement')})"
                )
                return False

            # 对话增强仅私聊生效
            if "GroupMessage" in umo:
                logger.debug("[Spark] 对话增强跳过: 群聊不触发")
                return False

            profile = self._user_profiles.get(umo)
            if not profile or not profile.subscribed:
                logger.debug(
                    f"[Spark] 对话增强跳过: 用户未订阅 (profile={profile is not None}, subscribed={profile.subscribed if profile else 'N/A'})"
                )
                return False

            # 调度时不检查免打扰（用户刚发了消息说明在线），执行时再检查

            # 已有待执行的增强任务
            if (
                umo in self._enhancement_tasks
                and not self._enhancement_tasks[umo].done()
            ):
                logger.debug("[Spark] 对话增强跳过: 已有待执行任务")
                return False

            # Calculate trigger probability
            base_prob = min(
                max(
                    self._get_int_cfg("enhancement", "enhancement_probability", 20),
                    0,
                ),
                100,
            )
            st = self._states.get(umo)
            if not st:
                logger.debug("[Spark] 对话增强跳过: 无 SessionState")
                return False
            if st.last_proactive_reply_ts > st.last_user_reply_ts:
                logger.debug("[Spark] 对话增强跳过: 用户尚未回应上次主动消息")
                return False

            roll = random.random() * 100
            triggered = roll < base_prob

            if triggered:
                logger.info(
                    f"[Spark] 对话增强触发: {umo} (概率={base_prob}%, roll={roll:.2f})"
                )
            else:
                logger.debug(
                    f"[Spark] 对话增强未触发: {umo} (概率={base_prob}%, roll={roll:.2f})"
                )

            return triggered
        except Exception as e:
            logger.error(f"[Spark] 对话增强判断出错: {e}")
            return False

    def _schedule_enhancement(self, umo: str, current_round: dict | None = None):
        """Schedule a heat-scaled or fixed-policy follow-up."""
        enhancement_cfg = self.cfg.get("enhancement") or {}
        if not isinstance(enhancement_cfg, dict):
            enhancement_cfg = {}
        mode = str(enhancement_cfg.get("mode") or "对话热度").strip()
        st = self._states.get(umo)
        now_ts = _now_tz(
            self._get_cfg("basic_settings", "timezone") or None
        ).timestamp()

        if mode == "固定时间":
            base_delay = max(
                1.0,
                float(
                    enhancement_cfg.get(
                        "fixed_delay_seconds",
                        self._get_int_cfg("enhancement", "enhancement_min_delay", 45),
                    )
                    or 45
                ),
            )
            policy = parse_policy(
                enhancement_cfg,
                legacy_jitter_key="fixed_random_fluctuation_seconds",
            )
            result = apply_seconds_policy(
                base_delay,
                policy,
                seed=f"enhancement:{umo}:{now_ts:.0f}",
            )
            if result.minutes is None:
                logger.info(f"[Spark] 对话增强跳过: {umo} (固定等待策略结果无效)")
                if st:
                    st.next_enhancement_ts = 0.0
                return
            delay = result.minutes
            detail = f"mode={mode}, delay={delay:.1f}s"
        else:
            hot_delay = self._get_cfg(
                "enhancement", "enhancement_hot_delay_seconds", None
            )
            cold_delay = self._get_cfg(
                "enhancement", "enhancement_cold_delay_seconds", None
            )
            hot_delay = (
                int(hot_delay)
                if hot_delay not in (None, "")
                else self._get_int_cfg("enhancement", "enhancement_min_delay", 45)
            )
            cold_delay = (
                int(cold_delay)
                if cold_delay not in (None, "")
                else self._get_int_cfg("enhancement", "enhancement_max_delay", 600)
            )
            heat = self._calc_heat(st, now_ts) if st else 0.0
            delay = float(heat_scaled_delay_seconds(hot_delay, cold_delay, heat))
            detail = f"mode={mode}, heat={heat:.2f}, delay={delay:.1f}s"

        gen = self._enhancement_gen.get(umo, 0)
        logger.info(f"[Spark] 已调度对话增强: {umo}, {detail}")
        task = asyncio.create_task(
            self._delayed_enhancement(umo, delay, gen, current_round=current_round)
        )
        self._enhancement_tasks[umo] = task
        if st:
            st.next_enhancement_ts = now_ts + delay

    async def _delayed_enhancement(
        self,
        umo: str,
        delay: float,
        gen: int,
        current_round: dict | None = None,
    ):
        """延迟执行对话增强回复"""
        try:
            logger.info(f"[Spark] 增强任务开始: {umo}, 等待{delay}秒")
            st = self._states.get(umo)
            if not st:
                logger.info(f"[Spark] 增强任务退出: {umo} (无SessionState)")
                return
            trigger_chat_ts = max(
                st.last_user_reply_ts, st.last_ai_reply_ts, st.last_proactive_reply_ts
            )
            logger.info(
                f"[Spark] 增强任务sleep: {umo}, trigger_chat_ts={trigger_chat_ts}"
            )

            await asyncio.sleep(delay)

            logger.info(f"[Spark] 增强任务醒来: {umo}")
            # If a newer task was scheduled after us, abort.
            if self._enhancement_gen.get(umo, 0) != gen:
                logger.info(f"[Spark] 对话增强取消: {umo} (已被新任务替换, gen={gen})")
                return

            # 检查订阅状态
            profile = self._user_profiles.get(umo)
            if not profile or not profile.subscribed:
                logger.info(f"[Spark] 增强任务退出: {umo} (未订阅)")
                return

            tz = self._get_cfg("basic_settings", "timezone") or None

            # 执行时检查免打扰（延迟期间可能已进入免打扰时段）
            now = _now_tz(tz)
            latest_chat_ts = max(
                st.last_user_reply_ts, st.last_ai_reply_ts, st.last_proactive_reply_ts
            )
            if latest_chat_ts > trigger_chat_ts:
                logger.info(f"[Spark] 增强任务退出: {umo} (等待期间已有新聊天)")
                return
            quiet = self._get_cfg("basic_settings", "quiet_hours", "") or ""
            user_quiet = profile.quiet_hours if profile.quiet_hours else quiet
            if _in_quiet(now, user_quiet):
                logger.info(f"[Spark] 增强任务退出: {umo} (免打扰时段)")
                return

            # busy_schedule 快速退出场景等同免打扰：先触发一次即时状态刷新，再读标记
            _force = getattr(self.context, "_busy_schedule_force_check", None)
            if _force:
                try:
                    await _force()
                except Exception:
                    pass
            is_busy_flag = getattr(self.context, "_busy_schedule_is_busy", False)
            if is_busy_flag:
                logger.info(
                    f"[Spark] 增强任务退出: {umo} (忙碌时段, flag={is_busy_flag})"
                )
                return

            # 选择提示词模板
            prompts = self._get_cfg("enhancement", "enhancement_prompt_templates") or []
            if not prompts:
                logger.info(f"[Spark] 增强任务退出: {umo} (无提示词模板)")
                return
            prompt_template = random.choice(prompts)

            logger.info(f"[Spark] 执行对话增强回复: {umo}")
            ok = await self._proactive_reply(
                umo,
                tz,
                prompt_template,
                skip_judge=bool(self._get_cfg("enhancement", "ignore_judge", False)),
                judge_current_round=current_round,
                source="conversation_enhancement",
            )
            if ok:
                logger.info(f"[Spark] 对话增强回复成功: {umo}")

        except asyncio.CancelledError:
            logger.debug(f"[Spark] 对话增强任务被取消: {umo}")
        except Exception as e:
            logger.error(f"[Spark] 对话增强执行出错({umo}): {e}")
        finally:
            current_task = asyncio.current_task()
            if self._enhancement_tasks.get(umo) is current_task:
                self._enhancement_tasks.pop(umo, None)
                st = self._states.get(umo)
                if st:
                    st.next_enhancement_ts = 0.0

    # 调度器

    async def _scheduler_loop(self):
        """后台调度循环任务，每30秒检查一次是否需要触发主动回复"""
        try:
            while not self._stopped:
                await asyncio.sleep(30)
                if self._stopped:
                    break
                await self._tick()
        except asyncio.CancelledError:
            pass  # 正常取消，不需要日志
        except Exception as e:
            logger.error(f"[Spark] Scheduler error: {e}")
        finally:
            logger.info("[Spark] Scheduler stopped.")

    async def _tick(self):
        """
        单次调度检查（每30秒执行一次）

        检查逻辑：
        1. 如果插件被停用，直接返回
        2. 从配置同步订阅状态（实现配置热重载）
        3. 遍历所有已订阅的会话，检查是否需要主动回复
        4. 检查是否在免打扰时间段内
        5. 检查是否需要自动退订
        """
        # 检查插件是否已停止（框架禁用插件时会调用terminate设置此标志）
        if self._stopped:
            return

        if not self.cfg.get("enable", True):
            return

        # 从配置同步订阅状态（实现配置热重载，静默模式，只在有变化时打印日志）
        self._sync_subscribed_users_from_config(silent=True)

        tz = self._get_cfg("basic_settings", "timezone") or None
        now = _now_tz(tz)
        quiet = self._get_cfg("basic_settings", "quiet_hours", "") or ""
        reply_interval = int(
            self._get_cfg("basic_settings", "reply_interval_seconds") or 10
        )

        # Project daily greeting tasks from fixed times and schedule activities
        daily_projection = self._parse_daily_slots(now)
        daily_slots = daily_projection.tasks

        # Refresh busy state once before the per-user loop
        _force = getattr(self.context, "_busy_schedule_force_check", None)
        if _force:
            try:
                await _force()
            except Exception:
                pass

        # Per-user loop with error isolation
        for umo, profile in list(self._user_profiles.items()):
            try:
                if not profile.subscribed:
                    continue

                user_quiet = profile.quiet_hours if profile.quiet_hours else quiet
                is_in_dnd = _in_quiet(now, user_quiet)
                is_busy = getattr(self.context, "_busy_schedule_is_busy", False)

                st = self._states.get(umo)
                if st and await self._should_auto_unsubscribe(umo, profile, st, now):
                    continue

                # DND stays fully silent. Busy periods may run the timing judge, but
                # the idle state machine still blocks delivery until the period ends.
                if not is_in_dnd:
                    await self._check_idle_greeting(
                        umo,
                        st,
                        now,
                        tz,
                        reply_interval,
                        is_busy=is_busy,
                    )

                # Daily greetings: ignore_dnd items bypass DND/busy
                await self._check_daily_greetings(
                    umo,
                    st,
                    profile,
                    now,
                    daily_slots,
                    tz,
                    reply_interval,
                    is_in_dnd=is_in_dnd,
                    is_busy=is_busy,
                )
            except Exception as e:
                logger.error(
                    f"[Spark] 处理用户 {umo} 的 tick 任务时发生错误: {e}", exc_info=True
                )
                continue  # 继续处理下一个用户，不影响整体调度

        # 调度器结束时使用去抖保存，减少磁盘I/O
        await self._debounced_save_session_data()

    def _coerce_schedule_datetime(
        self, value: object, now: datetime
    ) -> datetime | None:
        if not isinstance(value, datetime):
            return None
        if now.tzinfo and value.tzinfo is None:
            return value.replace(tzinfo=now.tzinfo)
        if not now.tzinfo and value.tzinfo:
            return value.replace(tzinfo=None)
        return value

    def _busy_timeline(self, now: datetime) -> list[tuple[datetime, datetime, str]]:
        get_timeline = getattr(self.context, "_busy_schedule_get_timeline", None)
        if not callable(get_timeline):
            return []
        try:
            raw_timeline = get_timeline()
        except Exception as exc:
            logger.debug(f"[Spark] 读取忙碌时间线失败: {exc}")
            return []

        periods: list[tuple[datetime, datetime, str]] = []
        for item in raw_timeline or []:
            if not isinstance(item, dict) or not item.get("valid", True):
                continue
            start = self._coerce_schedule_datetime(item.get("start"), now)
            end = self._coerce_schedule_datetime(item.get("end"), now)
            if not start or not end or end <= start:
                continue
            periods.append((start, end, str(item.get("activity") or "忙碌活动")))
        return sorted(periods, key=lambda item: item[0])

    def _format_busy_periods(self, now: datetime) -> str:
        periods = self._busy_timeline(now)
        if not periods:
            return "无可用忙碌时间线"
        return "\n".join(
            f"- {start.strftime('%Y-%m-%d %H:%M')} ~ "
            f"{end.strftime('%Y-%m-%d %H:%M')}：{activity}"
            for start, end, activity in periods
        )

    def _defer_out_of_busy_periods(self, due_ts: float, now: datetime) -> float:
        """Move a candidate to five minutes after any overlapping busy periods."""
        target = datetime.fromtimestamp(due_ts, tz=now.tzinfo)
        periods = self._busy_timeline(now)
        for _ in range(len(periods) + 1):
            overlap = next(
                (
                    (start, end)
                    for start, end, _activity in periods
                    if start <= target < end
                ),
                None,
            )
            if not overlap:
                break
            target = overlap[1] + timedelta(minutes=5)
        return target.timestamp()

    def _daily_task(
        self,
        *,
        slot_num: int,
        source_date: date,
        base: datetime,
        item: dict,
        occurrence: int = 0,
        source_type: str = "fixed",
        activity: str = "",
        timeline_index: int = -1,
        boundary: str = "",
    ) -> DailyGreetingTask:
        policy = parse_policy(item, legacy_jitter_key="jitter_minutes")
        configured_task_id = re.sub(
            r"[^A-Za-z0-9_.-]+", "_", str(item.get("task_id") or "").strip()
        )[:64]
        task_identity = configured_task_id or str(slot_num)
        seed = f"daily:{source_date.isoformat()}:{task_identity}:{occurrence}"
        target = apply_datetime_policy(base, policy, seed=seed)
        tag = (
            f"daily_{task_identity}_{occurrence}@{source_date.isoformat()}"
            f"->{target.strftime('%Y-%m-%d %H:%M')}"
        )
        return DailyGreetingTask(
            slot_num=slot_num,
            greeting_id=task_identity,
            target=target,
            tag=tag,
            prompt=item.get("prompt", ""),
            ignore_dnd=bool(item.get("ignore_dnd", False)),
            ignore_judge=bool(item.get("ignore_judge", False)),
            cooldown_minutes=max(0, int(item.get("cooldown_minutes", 0) or 0)),
            activity_trigger_interval_minutes=max(
                0, int(item.get("activity_trigger_interval_minutes", 0) or 0)
            ),
            source_date=source_date,
            source_type=source_type,
            activity=activity,
            occurrence=occurrence,
            timeline_index=timeline_index,
            boundary=boundary,
            base=base,
        )

    def _activity_daily_projection(
        self, slot_num: int, item: dict, source_date: date, now: datetime
    ) -> DailyGreetingProjection:
        get_timeline = getattr(self.context, "_busy_schedule_get_timeline", None)
        if not callable(get_timeline):
            if now.timestamp() - self._timeline_warning_at >= 300:
                logger.warning(
                    "[Spark] 忙碌日程结构化时间线接口不可用；"
                    "活动每日问候将在接口注册后自动恢复"
                )
                self._timeline_warning_at = now.timestamp()
            return DailyGreetingProjection(
                tasks=[],
                issues=[
                    DailyGreetingIssue(
                        slot_num=slot_num,
                        source_date=source_date,
                        status="timeline_unavailable",
                        detail="忙碌日程时间线接口不可用",
                    )
                ],
            )

        keywords = [
            str(keyword).strip()
            for keyword in item.get("activity_keywords", [])
            if str(keyword).strip()
        ]
        if not keywords:
            return DailyGreetingProjection(
                tasks=[],
                issues=[
                    DailyGreetingIssue(
                        slot_num=slot_num,
                        source_date=source_date,
                        status="no_keywords",
                        detail="未配置活动关键词",
                    )
                ],
            )
        try:
            timeline = get_timeline(source_date)
        except Exception as exc:
            logger.warning(f"[Spark] 获取结构化日程失败({source_date}): {exc}")
            return DailyGreetingProjection(
                tasks=[],
                issues=[
                    DailyGreetingIssue(
                        slot_num=slot_num,
                        source_date=source_date,
                        status="timeline_error",
                        detail=str(exc),
                    )
                ],
            )

        if not timeline:
            return DailyGreetingProjection(
                tasks=[],
                issues=[
                    DailyGreetingIssue(
                        slot_num=slot_num,
                        source_date=source_date,
                        status="no_schedule",
                        detail="当天没有可用日程",
                    )
                ],
            )

        boundary_value = str(item.get("activity_boundary", "活动开始") or "活动开始")
        boundary = "end" if boundary_value in {"end", "活动结束"} else "start"
        candidate_projection = project_activity_candidates(timeline, keywords, boundary)
        tasks = []
        issues = [
            DailyGreetingIssue(
                slot_num=slot_num,
                source_date=source_date,
                status=issue.status,
                detail=issue.detail,
                activity=issue.activity,
                occurrence=issue.occurrence,
            )
            for issue in candidate_projection.issues
        ]
        for issue in candidate_projection.issues:
            if issue.status == "invalid_boundary":
                logger.warning(
                    f"[Spark] 跳过无效日程活动: {issue.activity} ({issue.detail})"
                )

        for candidate in candidate_projection.candidates:
            base = self._coerce_schedule_datetime(candidate.boundary, now)
            if base is None:
                continue
            tasks.append(
                self._daily_task(
                    slot_num=slot_num,
                    source_date=source_date,
                    base=base,
                    item=item,
                    occurrence=candidate.occurrence,
                    source_type="activity",
                    activity=candidate.activity,
                    timeline_index=candidate.timeline_index,
                    boundary=boundary,
                )
            )
        return DailyGreetingProjection(tasks=tasks, issues=issues)

    def _parse_daily_slots(self, now: datetime) -> DailyGreetingProjection:
        """Project fixed-time and schedule-driven greetings onto concrete datetimes."""
        daily = self.cfg.get("daily_prompts") or {}
        greetings = daily.get("daily_greetings", [])
        tasks: list[DailyGreetingTask] = []
        issues: list[DailyGreetingIssue] = []
        source_dates = [
            now.date() - timedelta(days=1),
            now.date(),
            now.date() + timedelta(days=1),
        ]

        if isinstance(greetings, list) and greetings:
            activity_tasks: list[tuple[tuple[date, int], int, DailyGreetingTask]] = []

            for idx, item in enumerate(greetings):
                if not isinstance(item, dict) or not item.get("enable", False):
                    continue
                if item.get("trigger_source", "固定时间") in {
                    "activity",
                    "日程活动",
                }:
                    priority = max(0, int(item.get("priority", 0) or 0))
                    for source_date in source_dates:
                        projection = self._activity_daily_projection(
                            idx, item, source_date, now
                        )
                        for task in projection.tasks:
                            activity_tasks.append(
                                (
                                    (task.source_date, task.timeline_index),
                                    priority,
                                    task,
                                )
                            )
                        issues.extend(projection.issues)
                    continue
                parsed = _parse_hhmm(str(item.get("time", "")))
                if not parsed:
                    issues.append(
                        DailyGreetingIssue(
                            slot_num=idx,
                            source_date=now.date(),
                            status="invalid_time",
                            detail="固定时间格式无效",
                        )
                    )
                    continue
                for source_date in source_dates:
                    base = datetime.combine(
                        source_date, time(*parsed), tzinfo=now.tzinfo
                    )
                    tasks.append(
                        self._daily_task(
                            slot_num=idx,
                            source_date=source_date,
                            base=base,
                            item=item,
                        )
                    )

            tasks.extend(select_highest_priority(activity_tasks))

            return DailyGreetingProjection(tasks=tasks, issues=issues)

        for slot_num in [1, 2, 3]:
            slot_cfg = daily.get(f"slot{slot_num}", {})
            enabled = (
                bool(slot_cfg.get("enable", False))
                if slot_cfg
                else bool(daily.get(f"daily{slot_num}_enable", False))
            )
            time_str = (
                slot_cfg.get("time", "")
                if slot_cfg
                else daily.get(f"time{slot_num}", "")
            )
            prompt = (
                slot_cfg.get("prompt", "")
                if slot_cfg
                else daily.get(f"prompt{slot_num}", "")
            )
            parsed = _parse_hhmm(str(time_str))
            if not enabled or not parsed:
                continue
            for source_date in source_dates:
                base = datetime.combine(source_date, time(*parsed), tzinfo=now.tzinfo)
                tasks.append(
                    self._daily_task(
                        slot_num=slot_num,
                        source_date=source_date,
                        base=base,
                        item={"prompt": prompt},
                    )
                )
        return DailyGreetingProjection(tasks=tasks, issues=issues)

    async def _judge_idle_delay_minutes(self, umo: str, tz: str | None) -> int | None:
        result = await self._judge_should_reply(umo, tz, delay_protocol=True)
        return (
            result if isinstance(result, int) and not isinstance(result, bool) else None
        )

    def _initialize_idle_cycle(
        self,
        st: SessionState,
        profile: UserProfile,
        now_ts: float,
    ) -> None:
        anchor_ts = self._latest_chat_ts(st) or float(st.last_ts or 0) or now_ts
        st.idle_judge_cycle += 1
        st.idle_judge_checked_cycle = -1
        st.idle_judge_inflight_cycle = -1
        st.idle_judge_task_ts = 0.0
        st.idle_judge_anchor_ts = anchor_ts
        st.idle_retry_after_ts = 0.0
        st.idle_schedule_mode = self._idle_mode()
        delay_m = self._calc_fluctuated_idle_delay(st, anchor_ts, profile)
        st.next_idle_ts = anchor_ts + delay_m * 60 if delay_m else 0.0

    async def _check_idle_greeting(
        self,
        umo: str,
        st: SessionState | None,
        now: datetime,
        tz: str | None,
        reply_interval: int,
        *,
        is_busy: bool = False,
    ):
        """Advance one durable idle-greeting cycle."""
        if not bool(self._get_cfg("idle_greetings", "enable_idle_greetings", True)):
            return
        profile = self._user_profiles.get(umo)
        if not st or not profile or not profile.subscribed:
            return

        now_ts = now.timestamp()
        mode = self._idle_mode()
        if st.idle_schedule_mode != mode:
            self._initialize_idle_cycle(st, profile, now_ts)
            self._save_session_data()

        latest_activity_ts = self._latest_chat_ts(st)
        if (
            mode == "大模型判断"
            and st.idle_judge_anchor_ts > 0
            and latest_activity_ts > st.idle_judge_anchor_ts
        ):
            self._reset_idle_schedule(st, latest_activity_ts, profile)
            self._save_session_data()

        if (
            bool(
                self._get_cfg(
                    "idle_greetings", "require_user_reply_before_idle_greeting", True
                )
            )
            and st.last_proactive_reply_ts > st.last_user_reply_ts
        ):
            st.next_idle_ts = 0.0
            st.idle_judge_task_ts = 0.0
            logger.info(f"[Spark] 沉寂问候跳过: {umo} (用户尚未回应上次主动消息)")
            self._save_session_data()
            return

        if st.idle_retry_after_ts < 0:
            return
        if st.idle_retry_after_ts and now_ts < st.idle_retry_after_ts:
            return

        if mode == "大模型判断" and st.idle_judge_task_ts <= 0:
            if st.idle_judge_checked_cycle == st.idle_judge_cycle:
                return
            if st.next_idle_ts <= 0:
                anchor_ts = st.idle_judge_anchor_ts or latest_activity_ts or now_ts
                st.next_idle_ts = anchor_ts + self._idle_judge_after_minutes() * 60
                self._save_session_data()
            if now_ts < st.next_idle_ts:
                return

            cycle = st.idle_judge_cycle
            anchor_ts = st.idle_judge_anchor_ts
            st.idle_judge_checked_cycle = cycle
            st.idle_judge_inflight_cycle = cycle
            st.next_idle_ts = 0.0
            self._save_session_data()
            delay_minutes = await self._judge_idle_delay_minutes(umo, tz)

            if (
                st.idle_judge_cycle != cycle
                or st.idle_judge_anchor_ts != anchor_ts
                or self._latest_chat_ts(st) > anchor_ts
            ):
                logger.info(f"[Spark] 沉寂延迟判断结果失效: {umo} (判断期间已有新聊天)")
                return

            st.idle_judge_inflight_cycle = -1
            if delay_minutes is None:
                retry_minutes = max(
                    1,
                    self._get_int_cfg("idle_greetings", "judge_retry_minutes", 5),
                )
                st.idle_judge_checked_cycle = -1
                st.idle_retry_after_ts = now_ts + retry_minutes * 60
                st.next_idle_ts = st.idle_retry_after_ts
                logger.warning(
                    f"[Spark] 沉寂延迟判断技术失败: {umo}, {retry_minutes} 分钟后重试"
                )
                self._save_session_data()
                return

            st.idle_retry_after_ts = 0.0
            if delay_minutes == 0:
                logger.info(f"[Spark] 沉寂延迟判断结束当前周期: {umo} (返回 0)")
                self._save_session_data()
                return

            decided_at_dt = _now_tz(tz)
            decided_at = decided_at_dt.timestamp()
            desired_due = anchor_ts + delay_minutes * 60
            candidate_due = max(decided_at, desired_due)
            candidate_due = self._defer_out_of_busy_periods(
                candidate_due, decided_at_dt
            )
            st.idle_judge_task_ts = candidate_due
            st.next_idle_ts = st.idle_judge_task_ts
            logger.info(
                f"[Spark] 沉寂延迟判断已创建一次性任务: {umo}, "
                f"total_idle={delay_minutes}m, desired={desired_due:.0f}, "
                f"due={st.idle_judge_task_ts:.0f}"
            )
            self._save_session_data()
            return

        due_ts = st.idle_judge_task_ts if mode == "大模型判断" else st.next_idle_ts
        if due_ts <= 0:
            self._initialize_idle_cycle(st, profile, now_ts)
            self._save_session_data()
            return
        if now_ts < due_ts:
            return

        if is_busy:
            deferred_due = self._defer_out_of_busy_periods(max(due_ts, now_ts), now)
            if deferred_due <= now_ts:
                # A temporary/manual busy state may not appear in the exported
                # timeline. Rechecking every tick keeps delivery at least five
                # minutes behind the last observed busy state.
                deferred_due = now_ts + 5 * 60
            if mode == "大模型判断":
                st.idle_judge_task_ts = deferred_due
            st.next_idle_ts = deferred_due
            logger.info(
                f"[Spark] 沉寂问候因忙碌顺延: {umo}, due={deferred_due:.0f}"
            )
            self._save_session_data()
            return

        cooldown_minutes = max(
            0, self._get_int_cfg("idle_greetings", "cooldown_minutes", 10)
        )
        cooldown_until = cooldown_deadline(latest_activity_ts, cooldown_minutes)
        if now_ts < cooldown_until:
            if mode == "大模型判断":
                st.idle_judge_task_ts = cooldown_until
            st.next_idle_ts = cooldown_until
            self._save_session_data()
            return

        idle_prompts = self._get_cfg("idle_greetings", "idle_prompt_templates") or []
        if not idle_prompts:
            logger.warning(f"[Spark] 沉寂问候跳过: {umo} (未配置问候模板)")
            return

        tag = f"idle@{now.strftime('%Y-%m-%d %H:%M')}"
        if st.has_fired(tag):
            return
        skip_judge = mode == "大模型判断" or bool(
            self._get_cfg("idle_greetings", "ignore_judge", False)
        )
        logger.info(f"[Spark] 触发沉寂问候: {umo}, mode={mode}")
        ok = await self._proactive_reply(
            umo,
            tz,
            random.choice(idle_prompts),
            skip_judge=skip_judge,
            source="silence_greeting",
        )
        st.mark_fired(tag)
        if ok:
            st.next_idle_ts = 0.0
            st.idle_judge_task_ts = 0.0
            self._save_session_data()
            if reply_interval > 0:
                await asyncio.sleep(reply_interval)
            return

        st.consecutive_no_reply_count += 1
        retry_minutes = max(
            1, self._get_int_cfg("idle_greetings", "judge_retry_minutes", 5)
        )
        retry_at = _now_tz(tz).timestamp() + retry_minutes * 60
        if mode == "大模型判断":
            st.idle_judge_task_ts = retry_at
            st.next_idle_ts = retry_at
        else:
            retry_delay_m = self._calc_fluctuated_idle_delay(st, now_ts, profile)
            st.next_idle_ts = now_ts + retry_delay_m * 60 if retry_delay_m else retry_at
        self._save_session_data()

    async def _check_daily_greetings(
        self,
        umo: str,
        st: SessionState | None,
        profile: UserProfile,
        now: datetime,
        daily_slots: list[DailyGreetingTask],
        tz: str | None,
        reply_interval: int,
        is_in_dnd: bool = False,
        is_busy: bool = False,
    ):
        """Execute concrete daily tasks with durable, technical-only retries."""
        if (
            not bool(self.cfg.get("enable_daily_greetings", True))
            or not profile.daily_reminders_enabled
        ):
            return
        if not st:
            return

        now_ts = now.timestamp()
        initial_grace_seconds = max(
            30,
            self._get_int_cfg("daily_prompts", "initial_trigger_grace_seconds", 90),
        )
        retry_window_seconds = max(
            60,
            self._get_int_cfg("daily_prompts", "technical_retry_window_minutes", 15)
            * 60,
        )
        retry_interval_seconds = max(
            30,
            self._get_int_cfg("daily_prompts", "technical_retry_interval_seconds", 60),
        )
        max_attempts = max(
            1,
            self._get_int_cfg("daily_prompts", "technical_retry_max_attempts", 3),
        )

        for task in sorted(daily_slots, key=lambda item: item.target):
            if task.target.timestamp() > now_ts:
                continue
            state = st.daily_state(task)
            decision, planned = plan_task(
                state,
                now_ts=now_ts,
                initial_grace_seconds=initial_grace_seconds,
                max_attempts=max_attempts,
            )
            if planned != state:
                st.set_daily_state(planned)
                self._save_session_data()
            if decision != "attempt":
                continue

            interval_deadline = (
                success_interval_deadline(
                    st.daily_greeting_success_times,
                    task.greeting_id,
                    task.activity_trigger_interval_minutes,
                )
                if task.source_type == "activity"
                else 0.0
            )
            if interval_deadline and now_ts < interval_deadline:
                logger.info(
                    f"[Spark] 每日问候跳过: {umo} "
                    f"(同问候间隔 {task.activity_trigger_interval_minutes} 分钟)"
                )
                st.set_daily_state(
                    terminal_state(
                        planned,
                        "skipped_interval",
                        now_ts=now_ts,
                        reason="activity_success_interval",
                    )
                )
                self._save_session_data()
                continue

            latest_chat_ts = max(
                st.last_user_reply_ts,
                st.last_ai_reply_ts,
                st.last_proactive_reply_ts,
            )
            if (
                task.cooldown_minutes > 0
                and latest_chat_ts > 0
                and now_ts - latest_chat_ts < task.cooldown_minutes * 60
            ):
                logger.info(
                    f"[Spark] 每日问候跳过: {umo} "
                    f"(聊天冷却 {task.cooldown_minutes} 分钟)"
                )
                st.set_daily_state(
                    terminal_state(
                        planned,
                        "skipped_cooldown",
                        now_ts=now_ts,
                        reason="chat_cooldown",
                    )
                )
                self._save_session_data()
                continue

            if not task.ignore_dnd and is_in_dnd:
                logger.info(f"[Spark] 每日问候跳过: {umo} (免打扰时段)")
                st.set_daily_state(
                    terminal_state(
                        planned,
                        "skipped_dnd",
                        now_ts=now_ts,
                        reason="dnd",
                    )
                )
                self._save_session_data()
                continue
            if not task.ignore_dnd and is_busy:
                logger.info(f"[Spark] 每日问候跳过: {umo} (忙碌时段)")
                st.set_daily_state(
                    terminal_state(
                        planned,
                        "skipped_busy",
                        now_ts=now_ts,
                        reason="busy",
                    )
                )
                self._save_session_data()
                continue

            prompt_raw = task.prompt
            prompt_template = (
                random.choice(prompt_raw)
                if isinstance(prompt_raw, list) and prompt_raw
                else prompt_raw
            )
            if not prompt_template:
                st.set_daily_state(
                    terminal_state(
                        planned,
                        "failed",
                        now_ts=now_ts,
                        reason="empty_prompt_config",
                    )
                )
                self._save_session_data()
                continue

            if not task.ignore_judge:
                should_send = await self._judge_should_reply(umo, tz)
                if not should_send:
                    logger.info(f"[Spark] 每日问候跳过: {umo} (智能判断为否)")
                    st.set_daily_state(
                        terminal_state(
                            planned,
                            "skipped_judge",
                            now_ts=now_ts,
                            reason="judge_rejected",
                        )
                    )
                    self._save_session_data()
                    continue

            logger.info(
                f"[Spark] 触发每日定时{task.slot_num}回复 {umo} "
                f"(ignore_dnd={task.ignore_dnd}, ignore_judge={task.ignore_judge}, "
                f"attempt={planned.attempts + 1})"
            )
            if task.ignore_dnd and is_busy:
                flush_delay = int(
                    self._get_cfg("daily_prompts", "ignore_busy_flush_delay_seconds")
                    or 10
                )
                wake_fn = getattr(self.context, "_busy_schedule_wake_and_flush", None)
                if wake_fn:
                    try:
                        await wake_fn(umo)
                    except Exception as exc:
                        logger.warning(f"[Spark] wake_and_flush 失败: {exc}")
                await asyncio.sleep(flush_delay)

            sending = begin_attempt(
                planned,
                now_ts=now_ts,
                retry_window_seconds=retry_window_seconds,
            )
            st.set_daily_state(sending)
            self._save_session_data()
            ok = await self._proactive_reply(
                umo,
                tz,
                prompt_template,
                skip_judge=True,
                source="daily_greeting",
            )
            delivery_completed_at = _now_tz(tz).timestamp()
            if ok:
                st.set_daily_state(
                    terminal_state(
                        sending,
                        "sent",
                        now_ts=now_ts,
                        reason="delivered",
                    )
                )
                if task.source_type == "activity":
                    st.daily_greeting_success_times = record_success_time(
                        st.daily_greeting_success_times,
                        task.greeting_id,
                        delivery_completed_at,
                    )
                self._save_session_data()
                if reply_interval > 0:
                    await asyncio.sleep(reply_interval)
                continue

            st.consecutive_no_reply_count += 1
            st.set_daily_state(
                technical_failure(
                    sending,
                    now_ts=now_ts,
                    max_attempts=max_attempts,
                    retry_interval_seconds=retry_interval_seconds,
                )
            )
            self._save_session_data()

    async def _should_auto_unsubscribe(
        self, umo: str, profile: UserProfile, st: SessionState, now: datetime
    ) -> bool:
        """检查是否需要自动退订（根据用户无回复天数）"""
        # 手动退订的用户不会被自动退订逻辑处理
        if profile.manual_unsubscribe:
            return False

        max_days = int(self._get_cfg("basic_settings", "max_no_reply_days") or 0)
        if max_days <= 0:
            return False

        if st.last_user_reply_ts > 0:
            last_reply = datetime.fromtimestamp(st.last_user_reply_ts, tz=now.tzinfo)
            days_since_reply = (now - last_reply).days

            if days_since_reply >= max_days:
                profile.subscribed = False
                profile.auto_unsubscribed = True  # 标记为自动退订
                profile.manual_unsubscribe = False  # 确保不是手动退订状态
                logger.info(
                    f"[Spark] 自动退订 {umo}：用户{days_since_reply}天未回复（可自动重新激活）"
                )
                self._save_user_data()
                self._sync_subscribed_users_to_config()  # 同步到配置文件
                return True

        return False

    # 主动回复

    def _format_contexts_for_prompt(self, contexts: list, limit: int = 12) -> str:
        lines = []
        for msg in contexts[-limit:]:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            content = self._extract_history_text(msg.get("content", ""))
            if not content:
                continue
            speaker = "Mando" if role == "user" else "AI"
            lines.append(f"{speaker}: {content[:500]}")
        return "\n".join(lines)

    async def _refresh_realtime_context(self):
        force = getattr(self.context, "_busy_schedule_force_check", None)
        if callable(force):
            try:
                await force()
            except Exception as e:
                logger.debug(f"[Spark] 刷新 busy_schedule 状态失败: {e}")

    @staticmethod
    def _provider_id(provider) -> str:
        config = getattr(provider, "provider_config", None)
        if not isinstance(config, dict):
            return ""
        return str(config.get("id") or "").strip()

    def _resolve_provider_chain(
        self,
        *,
        umo: str,
        primary_id: object,
        fallback_ids: object,
        purpose: str,
    ) -> list:
        normalized_primary_id = str(primary_id or "").strip()
        try:
            current_provider = self.context.get_using_provider(umo=umo)
        except (TypeError, ValueError) as exc:
            logger.warning(f"[Spark] 获取当前对话模型失败({umo}): {exc}")
            current_provider = None

        providers, missing_ids = resolve_provider_chain(
            primary_id=normalized_primary_id,
            fallback_ids=fallback_ids,
            get_provider=self.context.get_provider_by_id,
            current_provider=current_provider,
        )
        for provider_id in missing_ids:
            role = "主模型" if provider_id == normalized_primary_id else "回退模型"
            suffix = "，使用当前对话模型" if role == "主模型" else "，已跳过"
            logger.warning(f"[Spark] {purpose}{role} '{provider_id}' 不存在{suffix}")
        return providers

    def _astrbot_generation_fallback_ids(self, umo: str) -> list[str]:
        astrbot_config = self.context.get_config(umo=umo)
        provider_settings = (
            astrbot_config.get("provider_settings", {}) if astrbot_config else {}
        )
        if not isinstance(provider_settings, dict):
            return []
        return normalize_provider_ids(provider_settings.get("fallback_chat_models", []))

    def _get_judge_providers(self, umo: str, *, delay_protocol: bool = False) -> list:
        if delay_protocol:
            group_key = "idle_greetings"
            primary_key = "idle_judge_provider"
            fallback_key = "idle_judge_fallback_providers"
            purpose = "沉寂判断"
        else:
            group_key = "proactive_settings"
            primary_key = "proactive_judge_provider"
            fallback_key = "proactive_judge_fallback_providers"
            purpose = "判断"
        return self._resolve_provider_chain(
            umo=umo,
            primary_id=self._get_cfg(group_key, primary_key, ""),
            fallback_ids=self._get_cfg(group_key, fallback_key, []),
            purpose=purpose,
        )

    def _get_gen_providers(self, umo: str) -> list:
        mode = self._get_cfg(
            "proactive_settings", "generation_fallback_mode", "插件独立回退"
        )
        fallback_ids = select_generation_fallback_ids(
            mode,
            self._get_cfg("proactive_settings", "generation_fallback_providers", []),
            self._astrbot_generation_fallback_ids(umo),
        )
        if str(mode or "").strip() == ASTRBOT_FALLBACK_MODE:
            logger.debug(f"[Spark] 生成模型回退跟随 AstrBot 当前会话配置: {umo}")
        return self._resolve_provider_chain(
            umo=umo,
            primary_id=self._get_cfg("proactive_settings", "fixed_provider", ""),
            fallback_ids=fallback_ids,
            purpose="生成",
        )

    async def _get_emotion_judge_context(self, umo: str) -> str:
        callback = getattr(self.context, "_emotion_state_get_prompt_context", None)
        if not callable(callback):
            logger.debug(f"[Spark] 判断上下文未包含 Emotion: {umo} (接口不可用)")
            return ""
        try:
            result = callback(umo)
            result = await result if asyncio.iscoroutine(result) else result
            context = str(result or "").strip()
            marker_complete = (
                "<emotion_state_rules>" in context
                and "</emotion_state_rules>" in context
                and "<emotion_state_snapshot>" in context
                and "</emotion_state_snapshot>" in context
            )
            logger.info(
                f"[Spark] 判断 Emotion 诊断: session={umo}, "
                f"marker_complete={marker_complete}, chars={len(context)}"
            )
            return context
        except Exception as exc:
            logger.warning(f"[Spark] 读取 Emotion 判断上下文失败({umo}): {exc}")
            return ""

    async def _judge_should_reply(
        self,
        umo: str,
        tz: str | None,
        current_round: dict | None = None,
        *,
        delay_protocol: bool = False,
    ) -> bool | int | None:
        """Run the shared judge context with yes/no or strict delay output."""
        try:
            await self._refresh_realtime_context()
            now = _now_tz(tz)
            time_fmt = (
                self._get_cfg("basic_settings", "time_format") or "%Y-%m-%d %H:%M"
            )
            now_str = now.strftime(time_fmt)

            st = self._states.get(umo)
            time_since_last_chat = "未知"
            last_chat_at = "未知"
            idle_elapsed_minutes = 0
            if st:
                _last_chat_ts = max(
                    st.last_user_reply_ts,
                    st.last_proactive_reply_ts,
                    st.last_ai_reply_ts,
                )
                if delay_protocol and st.idle_judge_anchor_ts > 0:
                    _last_chat_ts = st.idle_judge_anchor_ts
                if _last_chat_ts > 0:
                    time_since_last_chat = _format_time_delta(
                        now.timestamp() - _last_chat_ts
                    )
                    last_chat_at = datetime.fromtimestamp(
                        _last_chat_ts, tz=now.tzinfo
                    ).strftime(time_fmt)
                    idle_elapsed_minutes = max(
                        0, int((now.timestamp() - _last_chat_ts) // 60)
                    )

            last_user, last_ai = await self._get_last_messages(umo)

            configured_judge_rounds = self._get_cfg(
                "proactive_settings", "judge_history_rounds", 3
            )
            try:
                judge_history_rounds = max(0, int(configured_judge_rounds))
            except (TypeError, ValueError):
                judge_history_rounds = 3
            raw_judge_contexts = await self._get_conversation_contexts(
                umo,
                judge_history_rounds,
                preserve_round_boundaries=True,
                include_datetime=True,
            )
            current_round_source = "official"
            if current_round:
                pending_user = self._sanitize_retrieval_user_content(
                    self._extract_history_text(current_round.get("user", ""))
                )
                pending_ai = self._extract_history_text(
                    current_round.get("assistant", "")
                )
                official_rounds = self._project_complete_history_rounds(
                    raw_judge_contexts,
                    include_datetime=True,
                )
                latest_official = official_rounds[-1] if official_rounds else None
                already_present = bool(
                    latest_official
                    and not latest_official["proactive"]
                    and latest_official["semantic_user"] == pending_user
                    and "\n".join(latest_official["assistant"]) == pending_ai
                )
                if pending_user and pending_ai and not already_present:
                    pending_reminder = self._fallback_datetime_reminder(umo, tz)
                    raw_judge_contexts.extend(
                        [
                            {
                                "role": "user",
                                "content": build_user_content_with_datetime(
                                    pending_user,
                                    pending_reminder,
                                ),
                            },
                            {"role": "assistant", "content": pending_ai},
                        ]
                    )
                    current_round_source = "pending"
            judge_contexts, selected_judge_rounds = self._select_recent_round_contexts(
                raw_judge_contexts,
                judge_history_rounds,
                include_datetime=True,
            )
            proactive_rounds = sum(turn["proactive"] for turn in selected_judge_rounds)
            newest_type = (
                "proactive"
                if selected_judge_rounds and selected_judge_rounds[-1]["proactive"]
                else "normal"
                if selected_judge_rounds
                else "none"
            )
            logger.info(
                f"[Spark] Judge contexts for {umo}: "
                f"configured_rounds={judge_history_rounds}, "
                f"selected_rounds={len(selected_judge_rounds)}, "
                f"normal={len(selected_judge_rounds) - proactive_rounds}, "
                f"proactive={proactive_rounds}, newest={newest_type}, "
                f"current_round={current_round_source}, "
                f"messages={len(judge_contexts)}; "
                f"content={self._format_context_tail_for_log(judge_contexts, limit=len(judge_contexts))}"
            )

            judge_group = "idle_greetings" if delay_protocol else "proactive_settings"
            judge_prompt_key = (
                "idle_judge_prompt" if delay_protocol else "proactive_judge_prompt"
            )
            judge_rules_key = (
                "idle_judge_rules" if delay_protocol else "proactive_judge_rules"
            )
            judge_template = self._get_cfg(judge_group, judge_prompt_key) or ""
            if not judge_template:
                judge_template = (
                    "当前时间：{now}\n最后聊天时间：{last_chat_at}\n"
                    "已沉寂：{idle_elapsed_minutes} 分钟（{time_since_last_chat}）\n"
                    "日程：{today_schedule}\n当前活动：{current_activity}\n"
                    "下一个活动：{next_activity}\n忙碌状态：{busy_status}\n"
                    "忙碌时间线：\n{busy_periods}\n"
                    "用户节律：{time_period_prompt}\n内心世界：\n{emotion_state}\n"
                    "最近用户消息：{last_user}\n最近AI回复：{last_ai}\n\n"
                    "先根据聊天记录、日程和忙碌时间线确定理想的绝对联系时刻，"
                    "再输出从最后聊天时间起算的总沉寂分钟数。"
                    if delay_protocol
                    else (
                        "日程：{today_schedule}\n当前活动：{current_activity}\n"
                        "用户节律：{time_period_prompt}\n内心世界：\n{emotion_state}\n"
                        "距上次聊天：{time_since_last_chat}\n"
                        "最近用户消息：{last_user}\n最近AI回复：{last_ai}"
                    )
                )
            elif "{emotion_state}" not in judge_template:
                judge_template = (
                    f"{judge_template.rstrip()}\n内心世界：\n{{emotion_state}}"
                )
            judge_rules = self._get_cfg(judge_group, judge_rules_key) or ""
            if not judge_rules:
                judge_rules = (
                    "只能输出一个非负整数分钟数，不得输出任何其他内容。"
                    if delay_protocol
                    else '！！必须遵守！！：你只能输出一个字："是"或"否"，不允许输出任何其他字。'
                )
            today_schedule = getattr(self.context, "_busy_schedule_today_schedule", "")
            outfit = getattr(self.context, "_busy_schedule_outfit", "")
            current_activity = getattr(
                self.context, "_busy_schedule_current_activity", ""
            )
            next_activity = getattr(self.context, "_busy_schedule_next_activity", "")
            busy_status = "忙碌中" if getattr(
                self.context, "_busy_schedule_is_busy", False
            ) else "当前不忙碌"
            busy_periods = self._format_busy_periods(now)
            custom_prompt = getattr(self.context, "_busy_schedule_custom_prompt", "")
            _get_prompt = getattr(self.context, "_time_period_get_prompt", None)
            time_period_prompt = (
                _get_prompt()
                if callable(_get_prompt)
                else getattr(self.context, "_time_period_current_prompt", "")
            )
            emotion_state = await self._get_emotion_judge_context(umo)

            # Compute heat_level label for judge templates
            if st:
                _heat_val = self._calc_heat(st, now.timestamp())
            else:
                _heat_val = 0.0
            if _heat_val >= 0.6:
                heat_level = "热"
            elif _heat_val >= 0.2:
                heat_level = "温"
            else:
                heat_level = "冷"

            prompt_values = {
                "now": now_str,
                "last_chat_at": last_chat_at,
                "idle_elapsed_minutes": idle_elapsed_minutes,
                "judge_after_minutes": self._idle_judge_after_minutes(),
                "idle_min_total_minutes": self._idle_judge_bounds()[0],
                "idle_max_total_minutes": self._idle_judge_bounds()[1],
                "last_user": last_user,
                "last_ai": last_ai,
                "time_since_last_chat": time_since_last_chat,
                "umo": umo,
                "today_schedule": today_schedule,
                "outfit": outfit,
                "current_activity": current_activity,
                "next_activity": next_activity,
                "busy_status": busy_status,
                "busy_periods": busy_periods,
                "custom_prompt": custom_prompt,
                "time_period_prompt": time_period_prompt,
                "emotion_state": emotion_state,
                "heat_level": heat_level,
            }
            judge_prompt = _format_template(judge_template, prompt_values)
            judge_rules = _format_template(judge_rules, prompt_values)

            providers = self._get_judge_providers(
                umo,
                delay_protocol=delay_protocol,
            )
            if not providers:
                message = "判断模型链为空"
                logger.warning(f"[Spark] {message}({umo})")
                return None if delay_protocol else True

            judge_persona = self._resolve_persona(
                "proactive_settings", "judge_persona_id"
            )
            if not judge_persona:
                judge_persona = await self._get_current_persona_prompt(umo)
            if not judge_persona:
                judge_persona = (
                    "你是一个严格的对话判断助手，只输出非负整数分钟数"
                    if delay_protocol
                    else "你是一个对话判断助手，只回复是或否"
                )

            judge_retries = 3
            last_err = None
            for provider_index, provider in enumerate(providers):
                provider_id = self._provider_id(provider) or "<unknown>"
                if provider_index:
                    logger.warning(
                        f"[Spark] 判断模型切换到回退供应商: {provider_id} ({umo})"
                    )
                for attempt in range(judge_retries):
                    try:
                        llm_resp = await provider.text_chat(
                            prompt=None,
                            contexts=judge_contexts
                            + [
                                {"role": "user", "content": judge_prompt},
                                {"role": "user", "content": judge_rules},
                            ],
                            system_prompt=judge_persona,
                        )
                        response = (
                            llm_resp.completion_text
                            if hasattr(llm_resp, "completion_text")
                            else ""
                        ).strip()
                        if not response:
                            raise ValueError("Empty completion text")

                        if delay_protocol:
                            minimum, maximum = self._idle_judge_bounds()
                            delay_minutes = self._parse_delay_minutes(
                                response, minimum, maximum
                            )
                            if delay_minutes is None:
                                raise ValueError(
                                    f"Invalid delay protocol response: {response[:40]!r}"
                                )
                            logger.info(
                                f"[Spark] Judge delay for {umo} via {provider_id}: "
                                f"{delay_minutes} minutes"
                            )
                            return delay_minutes

                        should_reply = "是" in response[:10]
                        result = "YES" if should_reply else "NO"
                        logger.info(
                            f"[Spark] Judge {result} for {umo} via {provider_id}: "
                            f"'{response[:20]}'"
                        )
                        return should_reply

                    except Exception as e:
                        last_err = e
                        err_str = str(e).lower()
                        is_retryable = (
                            any(code in err_str for code in ("502", "503", "504"))
                            or "no usable output" in err_str
                            or "empty" in err_str
                            or "invalid delay protocol" in err_str
                            or "timeout" in err_str
                            or "connect" in err_str
                        )
                        if is_retryable and attempt < judge_retries - 1:
                            wait = 2 ** (attempt + 1)
                            logger.warning(
                                f"[Spark] Judge retry {attempt + 1}/{judge_retries} "
                                f"via {provider_id} for {umo}: {e}, waiting {wait}s"
                            )
                            await asyncio.sleep(wait)
                            continue
                        logger.warning(
                            f"[Spark] 判断模型 {provider_id} 调用失败({umo}): {e}"
                        )
                        break

            logger.error(f"[Spark] 所有判断模型均失败({umo}): {last_err}")
            return None if delay_protocol else True

        except Exception as e:
            logger.error(f"[Spark] Judge unexpected error({umo}): {e}")
            return None if delay_protocol else True

    def _resolve_persona(self, *config_keys) -> str:
        """Resolve persona_id from config to system_prompt via persona_mgr."""
        persona_id = self._get_cfg(*config_keys) or ""
        if not persona_id:
            return ""
        try:
            persona_mgr = self.context.persona_manager
            if persona_mgr:
                for p in persona_mgr.personas:
                    if p.persona_id == persona_id:
                        return p.system_prompt if p.system_prompt else ""
        except Exception as e:
            logger.warning(f"[Spark] Failed to resolve persona '{persona_id}': {e}")
        return ""

    def _get_gen_provider(self, umo: str):
        """Resolve the primary LLM provider for compatibility with older callers."""
        providers = self._get_gen_providers(umo)
        return providers[0] if providers else None

    async def _get_gen_persona(self, umo: str = "") -> str:
        """Resolve the system persona for the generate step via persona_mgr.

        Falls back to the current conversation's persona if no gen persona is configured.
        """
        persona = self._resolve_persona("proactive_settings", "gen_persona_id")
        if not persona and umo:
            persona = await self._get_current_persona_prompt(umo)
        return persona

    def _build_proactive_prompt_envelope(
        self,
        *,
        prompt: str,
        now_str: str,
        time_since_last_chat: str,
        last_user: str,
        last_ai: str,
    ) -> str:
        mode = str(
            self._get_cfg(
                "proactive_settings", "proactive_fact_envelope_mode", "minimal"
            )
            or "minimal"
        ).lower()
        if mode == "off":
            return prompt

        if mode != "full":
            stance = [
                "[主动回复姿态]",
                "本轮是 AI 主动开口，不是 Mando 新发来消息后等待回复。",
                "最近对话只作为历史上下文；不要把最近一条 Mando 消息当成刚刚收到的提问、催促或要求。",
                f"距上次聊天：{time_since_last_chat}",
                "回复要像你自己想起话题后主动找 Mando，而不是回答 Mando 刚刚问你。",
                "[/主动回复姿态]",
            ]
            return "\n".join(stance) + "\n\n" + prompt

        today_schedule = getattr(self.context, "_busy_schedule_today_schedule", "")
        outfit = getattr(self.context, "_busy_schedule_outfit", "")
        current_activity = getattr(self.context, "_busy_schedule_current_activity", "")
        next_activity = getattr(self.context, "_busy_schedule_next_activity", "")
        custom_prompt = getattr(self.context, "_busy_schedule_custom_prompt", "")
        _get_prompt = getattr(self.context, "_time_period_get_prompt", None)
        time_period_prompt = (
            _get_prompt()
            if callable(_get_prompt)
            else getattr(self.context, "_time_period_current_prompt", "")
        )
        facts = [
            "[主动回复实时事实]",
            "本轮是 AI 主动开口，不是 Mando 新发来消息后等待回复。",
            "最近对话只作为历史上下文；不要把最近一条 Mando 消息当成刚刚收到的提问、催促或要求。",
            f"当前时间：{now_str}",
            f"距上次聊天：{time_since_last_chat}",
            f"最近Mando消息：{last_user or '无'}",
            f"最近AI回复：{last_ai or '无'}",
        ]
        if today_schedule:
            facts.append(f"今日日程：{today_schedule}")
        if current_activity:
            facts.append(f"当前活动：{current_activity}")
        if next_activity:
            facts.append(f"下一个活动：{next_activity}")
        if outfit:
            facts.append(f"当前穿搭：{outfit}")
        if time_period_prompt:
            facts.append(f"当前节律：{time_period_prompt}")
        if custom_prompt:
            facts.append(f"附加状态：{custom_prompt}")
        facts.append(
            "事实优先级：当前时间、当前活动、当前节律、最近真实对话优先于长期记忆和知识库。长期记忆/知识库只能作为背景补充，不能当作今天刚发生的事；如果与当前事实冲突，必须忽略旧内容。"
        )
        facts.append("[/主动回复实时事实]")
        return "\n".join(facts) + "\n\n" + prompt

    def _proactive_placeholder(self) -> str:
        return (
            self._get_cfg("proactive_settings", "proactive_user_placeholder")
            or "[用户本人未发送消息，本轮为 AI 主动对 Mando 发起对话]"
        )

    def _fallback_datetime_reminder(self, umo: str, tz: str | None = None) -> str:
        astrbot_config = self.context.get_config(umo=umo)
        provider_settings = astrbot_config.get("provider_settings", {})
        if not provider_settings.get("datetime_system_prompt"):
            return ""
        timezone = tz or astrbot_config.get("timezone")
        now = _now_tz(timezone)
        if now.tzinfo is None:
            now = now.astimezone()
        current_time = now.strftime("%Y-%m-%d %H:%M (%Z)")
        return f"<system_reminder>Current datetime: {current_time}</system_reminder>"

    def _is_proactive_placeholder(self, content) -> bool:
        semantic_content = self._extract_history_text(content, exclude_datetime=True)
        normalized = self._normalize_history_text(semantic_content)
        known_placeholders = {
            self._normalize_history_text(self._proactive_placeholder()),
            "[用户本人未发送消息，本轮为 AI 主动对 Mando 发起对话]",
            "[用户本人未说话，本轮为 AI 主动发起对话]",
            "【会话占位：用户未发送新消息；助手在图片任务完成后补充通知】",
        }
        return normalized in known_placeholders

    def _is_natural_retrieval_line(self, content: str) -> bool:
        stripped = content.strip()
        if self._is_proactive_placeholder(stripped):
            return True
        if not stripped or stripped == "[主动对话]" or stripped.startswith("[灵犀主动"):
            return False
        blocked_markers = (
            "[主动回复实时事实]",
            "[/主动回复实时事实]",
            "[主动回复姿态]",
            "[/主动回复姿态]",
            "主动回复请求",
            "正在发送主动消息",
            "Output your last task result",
        )
        if any(marker in stripped for marker in blocked_markers):
            return False
        if stripped.startswith("/"):
            return False
        if self._is_internal_history_noise("user", stripped):
            return False
        return True

    def _sanitize_retrieval_user_content(self, content: str) -> str:
        stripped = str(content or "").strip()
        if not stripped or self._is_proactive_placeholder(stripped):
            return stripped

        internal_block_starts = (
            "[节律：",
            "[主动回复实时事实]",
            "[主动回复姿态]",
        )
        cut_positions = []
        for marker in internal_block_starts:
            start = stripped.find(marker)
            if start == 0:
                return ""
            if start > 0:
                cut_positions.append(start)
        if cut_positions:
            stripped = stripped[: min(cut_positions)].rstrip()

        return "" if self._is_internal_history_noise("user", stripped) else stripped

    def _project_complete_history_rounds(
        self,
        contexts: list,
        *,
        include_datetime: bool = False,
    ) -> list[dict]:
        """Project raw history into complete user/assistant rounds."""
        candidate_rounds = []
        current_round = None
        for msg in contexts:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            raw_content = msg.get("content", "")
            semantic_content = self._extract_history_text(
                raw_content, exclude_datetime=True
            )
            if role not in ("user", "assistant") or not semantic_content:
                continue
            if role == "user":
                current_round = None
                semantic_content = self._sanitize_retrieval_user_content(
                    semantic_content
                )
                if not semantic_content or not self._is_natural_retrieval_line(
                    semantic_content
                ):
                    continue
                model_content = raw_content if include_datetime else semantic_content
                current_round = {
                    "proactive": self._is_proactive_placeholder(raw_content),
                    "user": model_content,
                    "semantic_user": semantic_content,
                    "assistant": [],
                }
                candidate_rounds.append(current_round)
            elif self._is_internal_history_noise(role, semantic_content):
                continue
            elif current_round is not None and self._is_natural_retrieval_line(
                semantic_content
            ):
                current_round["assistant"].append(semantic_content)

        return [turn for turn in candidate_rounds if turn["assistant"]]

    def _select_recent_round_contexts(
        self,
        contexts: list,
        rounds: int,
        *,
        include_datetime: bool = False,
    ) -> tuple[list, list[dict]]:
        """Return the newest complete rounds as protocol role messages."""
        if rounds <= 0:
            return [], []
        complete_rounds = self._project_complete_history_rounds(
            contexts,
            include_datetime=include_datetime,
        )
        selected_rounds = complete_rounds[-rounds:]
        projected = []
        for turn in selected_rounds:
            projected.append({"role": "user", "content": turn["user"]})
            projected.append(
                {
                    "role": "assistant",
                    "content": "\n".join(turn["assistant"]),
                }
            )
        return projected, selected_rounds

    def _build_proactive_retrieval_query(self, contexts: list, prompt: str) -> str:
        configured_budget = self._get_cfg(
            "proactive_settings",
            "retrieval_query_max_chars",
            800,
        )
        try:
            total_budget = max(100, min(4000, int(configured_budget)))
        except (TypeError, ValueError):
            total_budget = 800
        max_rounds = 4
        proactive_marker = "[主动轮] 用户未发送新消息"
        mode_labels = {
            "仅最近对话": "recent_context",
            "仅自定义检索提示词": "retrieval_hint",
            "最近对话 + 自定义检索提示词": "recent_context_and_hint",
            "最近对话 + 完整生成指令": "recent_context_and_instruction",
        }
        configured_mode = self._get_cfg(
            "proactive_settings",
            "retrieval_mode",
            "最近对话 + 自定义检索提示词",
        )
        mode = mode_labels.get(configured_mode, configured_mode)
        valid_modes = set(mode_labels.values())
        if mode not in valid_modes:
            mode = "recent_context_and_hint"

        include_history = mode != "retrieval_hint"
        retrieval_text = ""
        retrieval_label = ""
        if mode in ("retrieval_hint", "recent_context_and_hint"):
            retrieval_text = str(
                self._get_cfg(
                    "proactive_settings",
                    "retrieval_hint",
                    "用户最近的经历、近况、未完成的话题",
                )
                or "用户最近的经历、近况、未完成的话题"
            ).strip()
            retrieval_label = "检索提示"
        elif mode == "recent_context_and_instruction":
            for end_tag in (
                "[/主动回复姿态]\n\n",
                "[/主动回复实时事实]\n\n",
            ):
                idx = prompt.find(end_tag)
                if idx != -1:
                    retrieval_text = prompt[idx + len(end_tag) :].strip()
                    break
            if not retrieval_text:
                retrieval_text = prompt.strip()
            retrieval_label = "模板指令"

        retrieval_block = (
            f"[{retrieval_label}：{retrieval_text}]"
            if retrieval_text and retrieval_label
            else ""
        )
        retrieval_truncated = len(retrieval_block) > total_budget
        if retrieval_truncated:
            suffix = retrieval_block[-(total_budget - 8) :]
            retrieval_block = "[已截断]" + suffix

        complete_rounds = (
            self._project_complete_history_rounds(contexts) if include_history else []
        )
        recent_rounds = complete_rounds[-max_rounds:]

        def render_round(turn: dict) -> str:
            user_line = (
                proactive_marker if turn["proactive"] else f"用户：{turn['user']}"
            )
            assistant_text = "\n".join(turn["assistant"])
            return f"{user_line}\nAI：{assistant_text}"

        separator = "\n\n" if retrieval_block and recent_rounds else ""
        history_header = "最近聊天：\n"
        history_budget = max(
            0,
            total_budget - len(retrieval_block) - len(separator),
        )
        selected_newest_first = []
        used_chars = len(history_header)
        for turn in reversed(recent_rounds):
            rendered = render_round(turn)
            added_chars = len(rendered) + (2 if selected_newest_first else 0)
            if used_chars + added_chars <= history_budget:
                selected_newest_first.append(rendered)
                used_chars += added_chars
                continue
            if not selected_newest_first and history_budget > len(history_header):
                available = history_budget - len(history_header)
                user_line = (
                    proactive_marker if turn["proactive"] else f"用户：{turn['user']}"
                )
                assistant_prefix = "\nAI："
                truncation_marker = "…[前文已截断]"
                fixed_chars = len(user_line) + len(assistant_prefix)
                if fixed_chars >= available:
                    user_budget = max(0, available - len(assistant_prefix))
                    rendered = user_line[:user_budget] + assistant_prefix
                else:
                    assistant_budget = available - fixed_chars
                    assistant_text = "\n".join(turn["assistant"])
                    if len(assistant_text) > assistant_budget:
                        tail_budget = max(
                            0,
                            assistant_budget - len(truncation_marker),
                        )
                        assistant_text = (
                            truncation_marker + assistant_text[-tail_budget:]
                            if tail_budget
                            else truncation_marker[:assistant_budget]
                        )
                    rendered = user_line + assistant_prefix + assistant_text
                selected_newest_first.append(rendered)
            break

        history_block = ""
        if selected_newest_first:
            history_block = history_header + "\n\n".join(
                reversed(selected_newest_first)
            )
        elif include_history and not retrieval_block:
            history_block = "延续最近对话"

        query_parts = [part for part in (history_block, retrieval_block) if part]
        query = "\n\n".join(query_parts) or "延续最近对话"
        selected_rounds = (
            recent_rounds[-len(selected_newest_first) :]
            if selected_newest_first
            else []
        )
        proactive_count = sum(turn["proactive"] for turn in selected_rounds)
        newest_type = (
            (
                "proactive"
                if recent_rounds and recent_rounds[-1]["proactive"]
                else "normal"
            )
            if recent_rounds
            else "none"
        )
        logger.debug(
            f"[Spark] 主动检索构造: mode={mode}, budget={total_budget}, "
            f"hint_chars={len(retrieval_text)}, history_budget={history_budget}, "
            f"candidates={len(complete_rounds)}, selected={len(selected_newest_first)}, "
            f"normal={len(selected_newest_first) - proactive_count}, "
            f"proactive={proactive_count}, newest={newest_type}, "
            f"query_chars={len(query)}, truncated={retrieval_truncated}"
        )
        return query

    async def _proactive_reply(
        self,
        umo: str,
        tz: str | None,
        prompt_template: str,
        skip_judge: bool = False,
        judge_current_round: dict | None = None,
        slash_triggered: bool = False,
        source: str = "silence_greeting",
    ) -> bool:
        """
        执行主动回复的核心方法

        v3 改造：通过官方 CronMessageEvent + build_main_agent 走合规 Agent Pipeline，
        支持完整的工具调用、人格注入、历史管理。
        当框架 API 不可用时降级到旧的 provider.text_chat 方式。

        Args:
            skip_judge: 为 True 时跳过 LLM 判断步骤，必定触发回复（用于每日问候等定时任务）
            slash_triggered: 为 True 时保留发送和历史，但不更新聊天时间或主动证据
            source: 自然主动消息来源，用于成功送达后的结构化证据
        """
        try:
            # Step 1: Judge whether to reply (skip for daily greetings etc.)
            if not skip_judge and self._get_cfg(
                "proactive_settings", "proactive_judge_enable", True
            ):
                if not await self._judge_should_reply(
                    umo,
                    tz,
                    current_round=judge_current_round,
                ):
                    return False

            # Step 2: Format prompt
            now = _now_tz(tz)
            time_fmt = (
                self._get_cfg("basic_settings", "time_format") or "%Y-%m-%d %H:%M"
            )
            now_str = now.strftime(time_fmt)

            st = self._states.get(umo)
            time_since_last_chat = "未知"
            if st:
                _last_chat_ts = max(
                    st.last_user_reply_ts,
                    st.last_proactive_reply_ts,
                    st.last_ai_reply_ts,
                )
                if _last_chat_ts > 0:
                    time_since_last_chat = _format_time_delta(
                        now.timestamp() - _last_chat_ts
                    )

            last_user, last_ai = await self._get_last_messages(umo)

            if prompt_template:
                today_schedule = getattr(
                    self.context, "_busy_schedule_today_schedule", ""
                )
                outfit = getattr(self.context, "_busy_schedule_outfit", "")
                current_activity = getattr(
                    self.context, "_busy_schedule_current_activity", ""
                )
                next_activity = getattr(
                    self.context, "_busy_schedule_next_activity", ""
                )
                custom_prompt = getattr(
                    self.context, "_busy_schedule_custom_prompt", ""
                )
                _get_prompt = getattr(self.context, "_time_period_get_prompt", None)
                time_period_prompt = (
                    _get_prompt()
                    if callable(_get_prompt)
                    else getattr(self.context, "_time_period_current_prompt", "")
                )

                # Compute heat_level label for generation templates
                if st:
                    _heat_val = self._calc_heat(st, now.timestamp())
                else:
                    _heat_val = 0.0
                if _heat_val >= 0.6:
                    _heat_level = "热"
                elif _heat_val >= 0.2:
                    _heat_level = "温"
                else:
                    _heat_level = "冷"

                try:
                    prompt = prompt_template.format(
                        now=now_str,
                        last_user=last_user,
                        last_ai=last_ai,
                        umo=umo,
                        time_since_last_chat=time_since_last_chat,
                        today_schedule=today_schedule,
                        outfit=outfit,
                        current_activity=current_activity,
                        next_activity=next_activity,
                        custom_prompt=custom_prompt,
                        time_period_prompt=time_period_prompt,
                        heat_level=_heat_level,
                    )
                except KeyError as e:
                    logger.warning(f"[Spark] prompt format error: {e}")
                    prompt = prompt_template
            else:
                prompt = "请自然地延续对话，与用户继续交流。"

            prompt = self._build_proactive_prompt_envelope(
                prompt=prompt,
                now_str=now_str,
                time_since_last_chat=time_since_last_chat,
                last_user=last_user,
                last_ai=last_ai,
            )

            logger.info(f"[Spark] 准备主动回复 {umo}")

            # Step 3: Generate via LLM (with gen provider/persona)
            gen_provider = self._get_gen_provider(umo)
            gen_persona = await self._get_gen_persona(umo)

            if HAS_AGENT_PIPELINE:
                delivery = await self._run_agent_pipeline(
                    umo,
                    prompt,
                    tz,
                    provider=gen_provider,
                    persona=gen_persona,
                    slash_triggered=slash_triggered,
                )
            else:
                logger.error(
                    f"[Spark] Agent Pipeline 不可用，主动对话退回旧路径；"
                    f"知识库、记忆 hook 与框架上下文裁剪不会生效。"
                    f"import_error={AGENT_PIPELINE_IMPORT_ERROR}"
                )
                response_text = await self._run_legacy_llm(
                    umo, prompt, provider=gen_provider, persona=gen_persona
                )
                delivery = (
                    AgentDeliveryResult(
                        response_text=response_text,
                        history_text=response_text,
                    )
                    if response_text
                    else None
                )

            if delivery is None:
                return False

            if not delivery.already_delivered:
                if not await self._send_text(umo, delivery.response_text):
                    return False
                logger.info(
                    f"[Spark] 已发送主动回复给 {umo}: {delivery.response_text[:50]}..."
                )
            else:
                logger.info(
                    f"[Spark] 工具已直接发送主动回复给 {umo}: "
                    f"kind={delivery.delivery_kind}, "
                    f"history={delivery.history_text[:50]}..."
                )

            # Save history only for legacy path; agent pipeline saves history in _run_agent_pipeline.
            if not HAS_AGENT_PIPELINE:
                try:
                    conv_mgr = self.context.conversation_manager
                    curr_cid = await conv_mgr.get_curr_conversation_id(umo)
                    if curr_cid:
                        reminder = self._fallback_datetime_reminder(umo, tz)
                        await self._add_message_pair_to_history(
                            umo,
                            curr_cid,
                            None,
                            build_proactive_user_content(
                                self._proactive_placeholder(), reminder
                            ),
                            delivery.history_text,
                        )
                except Exception as e:
                    logger.warning(f"[Spark] 保存主动回复历史失败: {e}")

            # Commit evidence only after the complete message was delivered.
            if not slash_triggered:
                sent_at = _now_tz(tz).timestamp()
                self._record_proactive_delivery(
                    umo,
                    source=source,
                    sent_at=sent_at,
                    response_text=delivery.history_text,
                )
                await self._debounced_save_session_data()

            return True

        except Exception as e:
            logger.error(f"[Spark] proactive error({umo}): {e}", exc_info=True)
            return False

    def _extract_history_text(self, content, *, exclude_datetime: bool = False) -> str:
        return extract_history_text(content, exclude_datetime=exclude_datetime)

    def _dedupe_contexts(
        self,
        contexts: list,
        *,
        preserve_content: bool = False,
    ) -> list:
        # Iterate in reverse so the latest occurrence of duplicate content is kept,
        # not the oldest. This matters when proactive placeholders repeat identically.
        seen = set()
        result = []
        for msg in reversed(contexts):
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            raw_content = msg.get("content", "")
            content = self._extract_history_text(raw_content)
            if role not in ("user", "assistant") or not content:
                continue
            if self._is_internal_history_noise(role, content):
                continue
            key = (role, re.sub(r"\s+", " ", content).strip())
            if key in seen:
                continue
            seen.add(key)
            result.append(
                {
                    "role": role,
                    "content": raw_content if preserve_content else content,
                }
            )
        result.reverse()
        return result

    def _format_context_tail_for_log(self, contexts: list, limit: int = 4) -> str:
        lines = []
        for msg in contexts[-limit:]:
            role = msg.get("role", "") if isinstance(msg, dict) else ""
            content = (
                self._extract_history_text(msg.get("content", ""))
                if isinstance(msg, dict)
                else ""
            )
            if not content:
                continue
            if role == "user" and self._is_proactive_placeholder(
                msg.get("content", "")
            ):
                speaker = "系统占位"
            else:
                speaker = "Mando" if role == "user" else "AI"
            lines.append(f"{speaker}: {content[:160]}")
        return " | ".join(lines) if lines else "无"

    def _normalize_history_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "")).strip()

    def _is_internal_history_noise(self, role: str, content: str) -> bool:
        stripped = str(content or "").strip()
        if not stripped:
            return True
        norm = self._normalize_history_text(stripped)
        if role == "assistant":
            return False
        if role != "user":
            return False
        blocked_markers = (
            "[灵犀主动",
            "主动对话",
            "[主动对话]",
            "[主动回复请求",
            "[节律：",
            "Output your last task result below.",
            "正在发送主动消息",
        )
        if stripped.startswith(blocked_markers):
            return True
        if "以上关于Mando的节律信息" in stripped:
            return True
        if "本轮是 AI 主动开口" in stripped:
            return True
        if "最近聊天：" in stripped and (
            "[用户本人未说话" in stripped or "[用户本人未发送消息" in stripped
        ):
            return True
        if norm.startswith("最近聊天：") and "AI:" in norm and "Mando:" in norm:
            return True
        return False

    def _parse_conversation_history(self, conversation) -> list:
        if not conversation or not getattr(conversation, "history", None):
            return []
        try:
            parsed = (
                json.loads(conversation.history)
                if isinstance(conversation.history, str)
                else conversation.history
            )
            return parsed if isinstance(parsed, list) else []
        except Exception as e:
            logger.warning(f"[Spark] 解析对话历史失败: {e}")
            return []

    async def _remove_internal_history_tail(
        self,
        umo: str,
        conversation_id: str,
        before_len: int | None = None,
        assistant_response: str = "",
    ) -> int:
        if not conversation_id:
            return 0
        conv_mgr = self.context.conversation_manager
        conversation = await conv_mgr.get_conversation(umo, conversation_id)
        history = self._parse_conversation_history(conversation)
        if not history:
            return 0

        response_key = self._normalize_history_text(assistant_response)
        tail_start = before_len if before_len is not None else max(0, len(history) - 30)
        tail_start = max(0, min(tail_start, len(history)))
        cleaned = []
        removed = 0
        for idx, msg in enumerate(history):
            if not isinstance(msg, dict):
                cleaned.append(msg)
                continue
            role = msg.get("role", "")
            content = self._extract_history_text(msg.get("content", ""))
            in_recent_tail = idx >= tail_start
            if in_recent_tail and self._is_internal_history_noise(role, content):
                removed += 1
                continue
            if (
                in_recent_tail
                and response_key
                and role == "assistant"
                and self._normalize_history_text(content) == response_key
            ):
                removed += 1
                continue
            cleaned.append(msg)

        if removed:
            await conv_mgr.update_conversation(umo, conversation_id, history=cleaned)
        return removed

    async def _save_standard_proactive_history(
        self,
        umo: str,
        conversation_id: str,
        assistant_response: str,
        baseline_len: int,
        datetime_reminder: str = "",
    ) -> None:
        assistant_response = self._clean_output_text(assistant_response)
        if not conversation_id or not assistant_response:
            return
        placeholder = self._proactive_placeholder()
        user_content = build_proactive_user_content(
            placeholder,
            datetime_reminder,
        )
        conv_mgr = self.context.conversation_manager
        removed = await self._remove_internal_history_tail(
            umo,
            conversation_id,
            before_len=baseline_len,
            assistant_response=assistant_response,
        )
        conversation = await conv_mgr.get_conversation(umo, conversation_id)
        history = self._parse_conversation_history(conversation)

        expected_tail = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_response},
        ]
        if history[-2:] != expected_tail:
            history.extend(expected_tail)
            await conv_mgr.update_conversation(umo, conversation_id, history=history)
        logger.debug(
            f"[Spark] 已写入标准主动历史: {conversation_id}, "
            f"baseline={baseline_len}, cleaned={removed}"
        )

    @staticmethod
    def _proactive_prompt_diagnostics(req) -> dict[str, object]:
        system_prompt = str(getattr(req, "system_prompt", "") or "")
        return {
            "emotion": "<!-- EMOTION_STATE_BEGIN -->" in system_prompt
            and "<!-- /EMOTION_STATE_END -->" in system_prompt,
            "emotion_anchor": "<!-- EMOTION_STATE_ANCHOR -->" in system_prompt,
            "busy_schedule": "<!-- BUSY_SCHEDULE_CACHE -->" in system_prompt,
            "system_prompt_chars": len(system_prompt),
            "extra_user_parts": len(
                getattr(req, "extra_user_content_parts", None) or []
            ),
        }

    async def _apply_proactive_request_hooks(
        self,
        event,
        req,
        *,
        umo: str,
        pipeline: str,
    ) -> bool:
        if not HAS_REQUEST_HOOKS:
            logger.warning(
                f"[Spark] 主动请求钩子不可用: {umo}, "
                f"pipeline={pipeline}, import_error={REQUEST_HOOK_IMPORT_ERROR}"
            )
            return False

        stopped = await call_event_hook(event, EventType.OnLLMRequestEvent, req)
        diagnostics = self._proactive_prompt_diagnostics(req)
        event.set_extra("spark_prompt_diagnostics", diagnostics)
        logger.info(
            f"[Spark] 主动请求装配诊断: session={umo}, pipeline={pipeline}, "
            f"emotion={diagnostics['emotion']}, "
            f"emotion_anchor={diagnostics['emotion_anchor']}, "
            f"busy_schedule={diagnostics['busy_schedule']}, "
            f"system_chars={diagnostics['system_prompt_chars']}, "
            f"extra_user_parts={diagnostics['extra_user_parts']}, stopped={stopped}"
        )
        return stopped

    async def _run_agent_pipeline(
        self,
        umo: str,
        prompt: str,
        tz: str | None = None,
        provider=None,
        persona: str = "",
        slash_triggered: bool = False,
    ) -> AgentDeliveryResult | None:
        """通过官方 CronMessageEvent + build_main_agent 执行 Agent Pipeline"""
        self._last_cron_event_sent = False

        session = MessageSession.from_str(umo)
        cron_event = CronMessageEvent(
            context=self.context,
            session=session,
            message=prompt,
        )

        astrbot_config = self.context.get_config(umo=umo)
        config = build_main_agent_config(
            astrbot_config,
            timezone=tz or astrbot_config.get("timezone"),
            streaming_response=False,
        )

        generation_providers = self._get_gen_providers(umo)
        if provider:
            generation_providers = dedupe_provider_chain(
                [provider, *generation_providers]
            )
        if not generation_providers:
            logger.warning(f"[Spark] 生成模型链为空: {umo}")
            return None
        provider = generation_providers[0]
        config.provider_settings = dict(config.provider_settings)
        config.provider_settings["fallback_chat_models"] = [
            provider_id
            for candidate in generation_providers[1:]
            if (provider_id := self._provider_id(candidate))
        ]

        req = ProviderRequest()
        req.prompt = prompt
        hook_history_len = None
        curr_cid = None

        # 获取会话并设置到 req.conversation，使 _ensure_persona_and_skills 能正常工作
        # 这样主动对话的人设注入方式就和正常对话相同，能共享 KV Cache
        try:
            conv_mgr = self.context.conversation_manager
            curr_cid = await conv_mgr.get_curr_conversation_id(umo)
            if curr_cid:
                conversation = await conv_mgr.get_conversation(umo, curr_cid)
                if conversation:
                    req.conversation = conversation
                    req.contexts = self._parse_conversation_history(conversation)
                    hook_history_len = len(req.contexts)
        except Exception as e:
            logger.warning(f"[Spark] 获取会话失败: {e}")

        retrieval_contexts = await self._get_conversation_contexts(
            umo,
            10,
            preserve_round_boundaries=True,
        )
        retrieval_query = self._build_proactive_retrieval_query(
            retrieval_contexts, prompt
        )
        logger.info(
            f"[Spark] Generation recent contexts for {umo}: {self._format_context_tail_for_log(retrieval_contexts)}"
        )
        logger.info(f"[Spark] Generation retrieval query for {umo}: {retrieval_query}")
        generation_prompt = prompt
        req.prompt = retrieval_query
        cron_event.set_extra("spark_proactive_retrieval", True)
        cron_event.set_extra("spark_slash_triggered", slash_triggered)

        result = await build_main_agent(
            event=cron_event,
            plugin_context=self.context,
            config=config,
            provider=provider,
            req=req,
            apply_reset=False,
        )
        datetime_reminder = find_datetime_reminder(req.extra_user_content_parts)

        if not result or not result.agent_runner:
            logger.warning(f"[Spark] build_main_agent 返回空结果: {umo}")
            return None

        runner = result.agent_runner

        cron_event.message_str = generation_prompt
        # hook 期间 req.prompt 保持 retrieval_query（真实聊天+模板指令），供 livingmemory/knowledge_base 做检索
        # 与正常 pipeline 顺序一致：OnLLMRequestEvent 触发后各插件注入记忆/知识库/节律等
        if await self._apply_proactive_request_hooks(
            cron_event,
            req,
            umo=umo,
            pipeline="agent",
        ):
            return None
        req.prompt = generation_prompt

        if curr_cid and hook_history_len is not None:
            try:
                removed = await self._remove_internal_history_tail(
                    umo,
                    curr_cid,
                    before_len=hook_history_len,
                )
                if removed:
                    conversation = (
                        await self.context.conversation_manager.get_conversation(
                            umo, curr_cid
                        )
                    )
                    req.conversation = conversation
                    req.contexts = self._parse_conversation_history(conversation)
                    logger.debug(
                        f"[Spark] 已清理本轮 hook 临时历史: {curr_cid}, "
                        f"baseline={hook_history_len}, cleaned={removed}"
                    )
            except Exception as e:
                logger.warning(f"[Spark] 清理主动 hook 临时历史失败: {e}")

        logger.info(
            f"[Spark] Main context for {umo}: messages={len(req.contexts)}, "
            f"max_turns={config.max_context_length}, "
            f"placeholder={any(self._is_proactive_placeholder(msg.get('content', '')) for msg in req.contexts if isinstance(msg, dict))}"
        )
        if result.reset_coro:
            await result.reset_coro

        delivered_texts = await self._consume_agent_responses(umo, runner)

        llm_resp = runner.get_final_llm_resp()
        delivery = resolve_agent_delivery(
            self._clean_output_text(
                getattr(llm_resp, "completion_text", "") if llm_resp else ""
            ),
            has_send_operation=getattr(cron_event, "_has_send_oper", False),
            direct_history_text=self._clean_output_text(
                cron_event.get_extra(DIRECT_DELIVERY_TEXT_EXTRA, "")
            ),
            direct_delivery_kind=cron_event.get_extra(DIRECT_DELIVERY_KIND_EXTRA, ""),
            delivered_texts=tuple(delivered_texts),
        )
        if delivery is None:
            logger.debug(f"[Spark] Agent 无响应且未发送消息: {umo}")
            return None

        # Store proactive replies as one ordinary-looking conversation round:
        # user placeholder + assistant response. This keeps future history reads
        # aligned with normal chat instead of retaining internal scheduler prompts.
        try:
            cid = getattr(req.conversation, "cid", None) if req.conversation else None
            if not cid and req.conversation:
                cid = getattr(req.conversation, "conversation_id", None)
            if not cid:
                conv_mgr = self.context.conversation_manager
                cid = await conv_mgr.get_curr_conversation_id(umo)
            if cid:
                await self._save_standard_proactive_history(
                    umo,
                    cid,
                    delivery.history_text,
                    hook_history_len or 0,
                    datetime_reminder,
                )
            else:
                logger.warning(f"[Spark] 保存主动回复历史失败: 未找到当前会话ID {umo}")
        except Exception as e:
            logger.warning(f"[Spark] 保存对话历史失败: {e}")

        return delivery

    async def _consume_agent_responses(self, umo: str, runner) -> list[str]:
        delivered_texts: list[str] = []
        async for agent_response in runner.step_until_done(30):
            await self._send_visible_agent_text(umo, agent_response, delivered_texts)
        return delivered_texts

    async def _send_visible_agent_text(
        self,
        umo: str,
        agent_response,
        delivered_texts: list[str],
    ) -> None:
        if getattr(agent_response, "type", "") != "llm_result":
            return

        response_data = getattr(agent_response, "data", {}) or {}
        chain = response_data.get("chain")
        if chain is None or getattr(chain, "type", None) == "reasoning":
            return

        text = self._clean_output_text(chain.get_plain_text())
        if not text or text in delivered_texts:
            return

        if await self._send_text(umo, text):
            delivered_texts.append(text)
            logger.info(f"[Spark] 已提前发送 Agent 文本给 {umo}: {text[:50]}...")

    async def _run_legacy_llm(
        self, umo: str, prompt: str, provider=None, persona: str = ""
    ) -> str | None:
        """Fallback: direct provider.text_chat() for older framework versions."""
        providers = self._get_gen_providers(umo)
        if provider:
            providers = dedupe_provider_chain([provider, *providers])
        if not providers:
            logger.warning(f"[Spark] provider missing for {umo}")
            return None

        if not persona:
            persona = await self._get_gen_persona(umo)

        self._last_cron_event_sent = False

        contexts = await self._get_conversation_contexts(
            umo,
            10,
            include_datetime=True,
        )
        logger.info(
            f"[Spark] Legacy generation recent contexts for {umo}: "
            f"{self._format_context_tail_for_log(contexts)}"
        )

        req = None
        if HAS_REQUEST_HOOKS:
            session = MessageSession.from_str(umo)
            cron_event = CronMessageEvent(
                context=self.context,
                session=session,
                message=prompt,
            )
            req = ProviderRequest(
                prompt=prompt,
                contexts=contexts,
                system_prompt=persona,
            )
            if await self._apply_proactive_request_hooks(
                cron_event,
                req,
                umo=umo,
                pipeline="legacy",
            ):
                return None

        last_err = None
        for provider_index, candidate in enumerate(providers):
            provider_id = self._provider_id(candidate) or "<unknown>"
            if provider_index:
                logger.warning(
                    f"[Spark] 旧生成路径切换到回退供应商: {provider_id} ({umo})"
                )
            try:
                llm_resp = await candidate.text_chat(
                    prompt=req.prompt if req else None,
                    contexts=req.contexts
                    if req
                    else [{"role": "user", "content": prompt}] + contexts,
                    system_prompt=req.system_prompt if req else persona,
                    extra_user_content_parts=(
                        req.extra_user_content_parts if req else None
                    ),
                )
                text = (
                    llm_resp.completion_text
                    if hasattr(llm_resp, "completion_text")
                    else ""
                )
                text = self._clean_output_text(text)
                if text:
                    return text
                raise ValueError("Empty completion text")
            except Exception as exc:
                last_err = exc
                logger.warning(
                    f"[Spark] 旧生成路径模型 {provider_id} 调用失败({umo}): {exc}"
                )

        logger.error(f"[Spark] 所有生成模型均失败({umo}): {last_err}")
        return None

    async def _add_message_pair_to_history(
        self,
        umo: str,
        conversation_id: str,
        conversation,
        user_prompt,
        assistant_response: str,
    ):
        """
        将消息对添加到对话历史（使用官方 API）

        注意：走 build_main_agent 的主动回复会在 _run_agent_pipeline 中保存历史，
        此方法仅用于降级路径或其他需要手动追加历史的场景。
        """
        assistant_response = self._clean_output_text(assistant_response)
        if not assistant_response:
            return
        try:
            if not conversation_id:
                logger.warning("[Spark] conversation_id 为空，无法更新历史")
                return

            conv_mgr = self.context.conversation_manager

            if HAS_NEW_MESSAGE_API:
                try:
                    user_parts = (
                        [TextPart(text=part["text"]) for part in user_prompt]
                        if isinstance(user_prompt, list)
                        else [TextPart(text=user_prompt)]
                    )
                    user_msg = UserMessageSegment(content=user_parts)
                    assistant_msg = AssistantMessageSegment(
                        content=[TextPart(text=assistant_response)]
                    )
                    await conv_mgr.add_message_pair(
                        cid=conversation_id,
                        user_message=user_msg,
                        assistant_message=assistant_msg,
                    )
                    logger.debug(f"[Spark] 已添加消息对到历史: {conversation_id}")
                    return
                except Exception as e:
                    logger.warning(f"[Spark] add_message_pair 失败: {e}")

            # 降级：使用 dict 格式
            await conv_mgr.add_message_pair(
                cid=conversation_id,
                user_message={"role": "user", "content": user_prompt},
                assistant_message={"role": "assistant", "content": assistant_response},
            )
            logger.debug(f"[Spark] 已添加消息对到历史(dict): {conversation_id}")

        except Exception as e:
            logger.error(f"[Spark] 添加消息对到历史失败: {e}", exc_info=True)

    async def _get_last_messages(self, umo: str) -> tuple[str, str]:
        """从官方 conversation 历史中获取最近的 user 和 assistant 消息（供占位符使用）"""
        last_user = ""
        last_ai = ""
        try:
            conv_mgr = self.context.conversation_manager
            curr_cid = await conv_mgr.get_curr_conversation_id(umo)
            if not curr_cid:
                return last_user, last_ai

            conversation = await conv_mgr.get_conversation(umo, curr_cid)
            if not conversation or not conversation.history:
                return last_user, last_ai

            history = (
                json.loads(conversation.history)
                if isinstance(conversation.history, str)
                else conversation.history
            )
            if not isinstance(history, list):
                return last_user, last_ai

            for msg in reversed(history):
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role", "")
                raw_content = msg.get("content", "")
                content = self._extract_history_text(raw_content, exclude_datetime=True)
                if self._is_proactive_placeholder(raw_content):
                    continue
                content = content[:200]
                if self._is_internal_history_noise(role, content):
                    continue
                if role == "user" and not last_user:
                    last_user = content
                elif role == "assistant" and not last_ai:
                    last_ai = content
                if last_user and last_ai:
                    break
        except Exception as e:
            logger.debug(f"[Spark] 获取最近消息失败: {e}")
        return last_user, last_ai

    async def _get_current_persona_prompt(self, umo: str) -> str:
        """Get the current conversation's persona system_prompt as fallback."""
        try:
            conv_mgr = self.context.conversation_manager
            curr_cid = await conv_mgr.get_curr_conversation_id(umo)
            if not curr_cid:
                return ""
            conversation = await conv_mgr.get_conversation(umo, curr_cid)
            if not conversation or not conversation.persona_id:
                return ""
            persona_mgr = self.context.persona_manager
            if persona_mgr:
                for p in persona_mgr.personas:
                    if p.persona_id == conversation.persona_id and p.system_prompt:
                        return p.system_prompt
        except Exception as e:
            logger.warning(f"[Spark] Failed to get current persona for {umo}: {e}")
        return ""

    async def _get_conversation_contexts(
        self,
        umo: str,
        rounds: int,
        preserve_round_boundaries: bool = False,
        *,
        include_datetime: bool = False,
    ) -> list:
        """Fetch recent history projected for model context or semantic retrieval."""
        if rounds <= 0:
            return []

        msgs = []
        try:
            conv_mgr = self.context.conversation_manager
            curr_cid = await conv_mgr.get_curr_conversation_id(umo)
            if curr_cid:
                conversation = await conv_mgr.get_conversation(umo, curr_cid)
                if conversation and conversation.history:
                    history = (
                        json.loads(conversation.history)
                        if isinstance(conversation.history, str)
                        else conversation.history
                    )
                    if isinstance(history, list):
                        source_history = (
                            history
                            if preserve_round_boundaries
                            else history[-rounds * 4 :]
                        )
                        for msg in source_history:
                            if not isinstance(msg, dict):
                                continue
                            role = msg.get("role", "")
                            if role not in ("user", "assistant"):
                                continue
                            raw_content = msg.get("content", "")
                            content = (
                                raw_content
                                if include_datetime
                                else self._extract_history_text(
                                    raw_content,
                                    exclude_datetime=True,
                                )
                            )
                            if content:
                                msgs.append({"role": role, "content": content})
        except Exception as e:
            logger.warning(
                f"[Spark] Failed to get conversation contexts for {umo}: {e}"
            )

        if preserve_round_boundaries:
            return msgs
        msgs = self._dedupe_contexts(
            msgs,
            preserve_content=include_datetime,
        )
        return msgs[-rounds * 2 :]

    def _apply_segmentation(self, text: str) -> list[str]:
        """应用分段回复逻辑（模拟 AstrBot 的分段正则处理）

        Returns:
            分段后的文本列表
        """
        try:
            # 获取分段配置
            seg_config = (
                self.context.get_config()
                .get("platform_settings", {})
                .get("segmented_reply", {})
            )

            # 检查是否启用分段
            if not seg_config.get("enable", False):
                return [text]

            # 获取配置参数
            words_threshold = int(seg_config.get("words_count_threshold", 1000))
            regex_pattern = seg_config.get("regex", r"[^。！？\n]+[。！？\n]?")
            cleanup_rule = seg_config.get("content_cleanup_rule", "")

            # 如果文本过长，不分段（与 AstrBot 逻辑一致）
            if len(text) > words_threshold:
                return [text]

            # 应用分段正则
            segments = re.findall(regex_pattern, text, re.DOTALL | re.MULTILINE)

            if not segments:
                return [text]

            # 清理并过滤空段落
            result = []
            for seg in segments:
                if cleanup_rule:
                    seg = re.sub(cleanup_rule, "", seg)
                if seg.strip():
                    result.append(seg)

            return result if result else [text]

        except Exception as e:
            logger.warning(f"[Spark] 分段处理失败，使用原始文本: {e}")
            return [text]

    def _clean_output_text(self, value: object) -> str:
        text = str(value or "").strip()
        context = getattr(self, "context", None)
        cleaner = getattr(context, "_thinking_cleaner_clean_text", None)
        if not callable(cleaner):
            return text
        try:
            return str(cleaner(text) or "").strip()
        except Exception as exc:
            logger.warning(
                f"[Spark] Thinking cleanup failed; using original text: {exc}"
            )
            return text

    async def _send_text(self, umo: str, text: str) -> bool:
        """发送主动回复消息到指定会话"""
        text = self._clean_output_text(text)
        if not text:
            return False
        try:
            # 检查 umo 是否缺少 session_id（例如：platform:MessageType:None）
            # 如果是，尝试从 conversation_manager 获取完整的 umo
            if umo.endswith(":None") or ":None" in umo:
                try:
                    conv_mgr = self.context.conversation_manager
                    # 尝试获取当前会话ID
                    curr_cid = await conv_mgr.get_curr_conversation_id(umo)
                    if curr_cid:
                        # 重新构造完整的 umo
                        # umo 格式通常是 platform:MessageType:session_id
                        parts = umo.split(":")
                        if len(parts) >= 2:
                            # 使用获取到的 conversation_id 替换 None
                            umo = f"{parts[0]}:{parts[1]}:{curr_cid}"
                            logger.debug(f"[Spark] 修复 umo: {umo}")
                except Exception as e:
                    logger.warning(f"[Spark] 尝试修复 umo 失败: {e}")

            # 应用分段逻辑
            segments = self._apply_segmentation(text)

            # 发送每个分段
            for segment in segments:
                message_chain = MessageChain().message(segment)
                await self.context.send_message(umo, message_chain)
                logger.debug(f"[Spark] ✅ 消息片段已发送: {segment[:50]}...")

                # 如果有多个分段，添加短暂延迟（模拟分段回复的间隔）
                if len(segments) > 1:
                    await asyncio.sleep(1.5)
            return True

        except Exception as e:
            logger.error(f"[Spark] ❌ 发送消息失败({umo}): {e}")
            return False

    # 生命周期管理
    async def terminate(self):
        """插件销毁"""
        self._stopped = True  # 设置停止标志，让调度器循环退出
        if (
            getattr(self.context, "_spark_get_proactive_state", None)
            == self._get_proactive_state
        ):
            delattr(self.context, "_spark_get_proactive_state")

        if self._loop_task and not self._loop_task.done():
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass  # 预期的取消异常

        # 取消所有对话增强任务
        for task in list(self._enhancement_tasks.values()):
            if task and not task.done():
                task.cancel()
        self._enhancement_tasks.clear()

        logger.info("[Spark] Performing final data save before termination...")
        if self._save_user_data_task and not self._save_user_data_task.done():
            self._save_user_data_task.cancel()
        if self._save_session_data_task and not self._save_session_data_task.done():
            self._save_session_data_task.cancel()
        self._save_user_data()
        self._save_session_data()

        logger.info("[Spark] 插件已停止")
