
from __future__ import annotations

import asyncio
import json
import math
import os
import random
import re
from dataclasses import dataclass
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple

from astrbot.api import logger, AstrBotConfig
from astrbot.api.event import filter, AstrMessageEvent, MessageChain
from astrbot.api.star import Context, Star, register

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
        UserMessageSegment,
        TextPart,
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

# 导入官方 Agent Pipeline API（用于主动回复走合规调用）
try:
    from astrbot.core.cron.events import CronMessageEvent
    from astrbot.core.astr_main_agent import (
        build_main_agent,
        build_main_agent_config,
    )
    from astrbot.core.provider.entities import ProviderRequest
    from astrbot.core.platform.message_session import MessageSession
    from astrbot.core.pipeline.context import call_event_hook
    from astrbot.core.star.star_handler import EventType
    HAS_AGENT_PIPELINE = True
    AGENT_PIPELINE_IMPORT_ERROR = ""
except ImportError as exc:
    HAS_AGENT_PIPELINE = False
    AGENT_PIPELINE_IMPORT_ERROR = repr(exc)

# 工具函数
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


def _compute_heat(msg_timestamps: list, now_ts: float, window_minutes: float, full_score_messages: float = 10.0) -> float:
    """Compute a [0.0, 1.0] conversation heat score using exponential decay."""
    if not msg_timestamps:
        return 0.0
    window_minutes = max(float(window_minutes or 1), 1.0)
    full_score_messages = max(float(full_score_messages or 10), 1.0)
    window_sec = window_minutes * 60.0
    total = sum(
        math.exp(-3.0 * (now_ts - t) / window_sec)
        for t in msg_timestamps
        if 0 <= now_ts - t <= window_sec
    )
    return min(total / full_score_messages, 1.0)



def _parse_hhmm(s: str) -> Optional[Tuple[int, int]]:
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

# 数据类定义
@dataclass
class UserProfile:
    """用户订阅信息和个性化设置"""
    subscribed: bool = False
    idle_after_minutes: int | None = None  
    daily_reminders_enabled: bool = True
    daily_reminder_count: int = 3
    quiet_hours: str | None = None  # 用户专属免打扰时间 "HH:MM-HH:MM"
    manual_unsubscribe: bool = False  # 标记是否是手动退订（强开关）
    auto_unsubscribed: bool = False  # 标记是否是自动退订（用于自动重新激活判断）

    def to_dict(self):
        return {
            "subscribed": self.subscribed,
            "idle_after_minutes": self.idle_after_minutes,
            "daily_reminders_enabled": self.daily_reminders_enabled,
            "daily_reminder_count": self.daily_reminder_count,
            "quiet_hours": self.quiet_hours,
            "manual_unsubscribe": self.manual_unsubscribe,
            "auto_unsubscribed": self.auto_unsubscribed
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            subscribed=data.get("subscribed", False),
            idle_after_minutes=data.get("idle_after_minutes"),
            daily_reminders_enabled=data.get("daily_reminders_enabled", True),
            daily_reminder_count=data.get("daily_reminder_count", 3),
            quiet_hours=data.get("quiet_hours"),
            manual_unsubscribe=data.get("manual_unsubscribe", False),
            auto_unsubscribed=data.get("auto_unsubscribed", False)
        )

@dataclass
class SessionState:
    """运行时会话状态（内存中维护）"""
    last_ts: float = 0.0
    last_fired_tag: str = ""  # 保留用于向后兼容
    last_fired_tags: dict = None  # 改为字典：{tag: timestamp}，支持过期清理
    last_user_reply_ts: float = 0.0
    consecutive_no_reply_count: int = 0
    next_idle_ts: float = 0.0
    last_proactive_reply_ts: float = 0.0  # 最近一次主动回复时间戳
    last_ai_reply_ts: float = 0.0  # 最近一次 AI 普通回复时间戳（用于对话增强取消判断）
    msg_timestamps: list = None  # rolling window of user message timestamps for heat computation
    proactive_recent_messages: list = None  # Deprecated; kept only for old session data compatibility
    next_enhancement_ts: float = 0.0  # scheduled enhancement fire time (runtime only, not persisted)

    def __post_init__(self):
        """初始化后处理"""
        if self.last_fired_tags is None:
            self.last_fired_tags = {}
            # 迁移旧数据
            if self.last_fired_tag:
                self.last_fired_tags[self.last_fired_tag] = _now_tz(None).timestamp()
        if self.msg_timestamps is None:
            self.msg_timestamps = []
        if self.proactive_recent_messages is None:
            self.proactive_recent_messages = []

    def to_dict(self):
        return {
            "last_ts": self.last_ts,
            "last_fired_tag": self.last_fired_tag,  # 保留用于向后兼容
            "last_fired_tags": self.last_fired_tags if self.last_fired_tags else {},
            "last_user_reply_ts": self.last_user_reply_ts,
            "consecutive_no_reply_count": self.consecutive_no_reply_count,
            "next_idle_ts": self.next_idle_ts,
            "last_proactive_reply_ts": self.last_proactive_reply_ts,
            "last_ai_reply_ts": self.last_ai_reply_ts,
            "msg_timestamps": self.msg_timestamps if self.msg_timestamps else [],
        }

    @classmethod
    def from_dict(cls, data: dict):
        tags_dict = data.get("last_fired_tags", {})
        if not isinstance(tags_dict, dict):
            tags_dict = {}
        msg_ts = data.get("msg_timestamps", [])
        if not isinstance(msg_ts, list):
            msg_ts = []
        proactive_recent = []

        return cls(
            last_ts=data.get("last_ts", 0.0),
            last_fired_tag=data.get("last_fired_tag", ""),
            last_fired_tags=tags_dict,
            last_user_reply_ts=data.get("last_user_reply_ts", 0.0),
            consecutive_no_reply_count=data.get("consecutive_no_reply_count", 0),
            next_idle_ts=data.get("next_idle_ts", 0.0),
            last_proactive_reply_ts=data.get("last_proactive_reply_ts", 0.0),
            last_ai_reply_ts=data.get("last_ai_reply_ts", 0.0),
            msg_timestamps=msg_ts,
            proactive_recent_messages=proactive_recent,
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
        expired_tags = [t for t, ts in self.last_fired_tags.items() if now_ts - ts > 7 * 86400]
        for t in expired_tags:
            del self.last_fired_tags[t]


@dataclass
class Reminder:
    """用户设置的提醒事项"""
    id: str
    umo: str
    content: str
    at: str  # "YYYY-MM-DD HH:MM" 或 "HH:MM|daily"
    created_at: float

    def to_dict(self):
        return {
            "id": self.id,
            "umo": self.umo,
            "content": self.content,
            "at": self.at,
            "created_at": self.created_at
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            id=data.get("id"),
            umo=data.get("umo"),
            content=data.get("content"),
            at=data.get("at"),
            created_at=data.get("created_at")
        )

# 灵犀 · 主动对话插件
# 灵感参考：astrbot_plugin_Conversa v3.0.0 (Luna-channel)
@register("astrbot_plugin_Spark", "灵犀 · 主动对话", "让 AI 像真人一样主动找你聊天——通过大模型智能判断何时该开口、何时该沉默，支持忙碌时段免打扰、独立判断/生成双模型、无限定时问候", "1.4.0", "https://github.com/gongzhudeng/astrbot_plugin_Spark")
class Spark(Star):

    # 初始化
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.cfg: AstrBotConfig = config
        self._loop_task: Optional[asyncio.Task] = None
        self._stopped: bool = False  # 插件停止标志
        
        # 运行时数据
        self._states: Dict[str, SessionState] = {}
        self._user_profiles: Dict[str, UserProfile] = {}
        self._reminders: Dict[str, Reminder] = {}
        
        # 文件保存去抖相关
        self._save_user_data_task: Optional[asyncio.Task] = None
        self._save_session_data_task: Optional[asyncio.Task] = None
        self._save_delay_seconds = 2.0  # 去抖延迟：2秒
        
        # 对话增强相关
        self._enhancement_tasks: Dict[str, asyncio.Task] = {}
        self._enhancement_gen: Dict[str, int] = {}  # generation counter per umo
        self._heat_event_marker = "_spark_heat_counted"
        
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
                if hasattr(context, 'get_data_path') or hasattr(self, 'get_data_path'):
                    data_path_func = getattr(context, 'get_data_path', None) or getattr(self, 'get_data_path', None)
                    if data_path_func:
                        base_path = data_path_func()
                        self._data_dir = _ensure_dir(os.path.join(base_path, "astrbot_plugin_conversa"))
                    else:
                        raise AttributeError
                else:
                    raise AttributeError
            except (AttributeError, TypeError):
                # 最终后备：基于当前工作目录，但添加警告
                import warnings
                warnings.warn("[Spark] 无法使用 StarTools，使用 os.getcwd() 作为后备方案")
                root = os.getcwd()
                self._data_dir = _ensure_dir(os.path.join(root, "data", "plugin_data", "astrbot_plugin_conversa"))
        
        self._user_data_path = os.path.join(self._data_dir, "user_data.json")
        self._session_data_path = os.path.join(self._data_dir, "session_data.json")
        
        # 加载数据
        self._load_user_data()
        self._load_session_data()
        self._sync_subscribed_users_from_config()
        self._migrate_config()
        self._migrate_daily_greetings()

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
                    greetings.append({
                        "enable": slot_cfg.get("enable", False),
                        "time": slot_cfg.get("time", ""),
                        "prompt": slot_cfg.get("prompt", ""),
                        "ignore_dnd": False,
                    })

            # 如果没有 slot 格式，检查扁平格式（daily1_enable/time1/prompt1）
            if not any(g.get("time") for g in greetings):
                greetings = []
                for n in [1, 2, 3]:
                    if daily.get(f"daily{n}_enable", False) or daily.get(f"time{n}", ""):
                        greetings.append({
                            "enable": daily.get(f"daily{n}_enable", False),
                            "time": daily.get(f"time{n}", ""),
                            "prompt": daily.get(f"prompt{n}", ""),
                            "ignore_dnd": False,
                        })

            if greetings:
                daily["daily_greetings"] = greetings
                self.cfg["daily_prompts"] = daily
                self.cfg.save_config()
                logger.info(f"[Spark] 已迁移旧每日问候配置到新列表格式（{len(greetings)} 个时段）")
        except Exception as e:
            logger.warning(f"[Spark] 每日问候配置迁移失败: {e}")

    def _migrate_config(self):
        """One-time config migration: old locations -> new locations"""
        try:
            changed = False
            proactive = self.cfg.get("proactive_settings") or {}
            basic = self.cfg.get("basic_settings") or {}
            advanced = self.cfg.get("advanced") or {}
            heat = self.cfg.get("heat_settings") or {}
            enhancement = self.cfg.get("enhancement") or {}

            if isinstance(heat, dict):
                if "heat_window_minutes" not in heat:
                    heat["heat_window_minutes"] = int(float(heat.get("heat_window_hours", 4) or 4) * 60)
                    changed = True
                    logger.info("[Spark] migrated heat_window_hours -> heat_window_minutes")
                if "heat_messages_for_full_score" not in heat:
                    heat["heat_messages_for_full_score"] = 10
                    changed = True
                self.cfg["heat_settings"] = heat

            if isinstance(enhancement, dict) and "enhancement_cooldown_minutes" not in enhancement:
                enhancement["enhancement_cooldown_minutes"] = 12
                self.cfg["enhancement"] = enhancement
                changed = True
            if advanced.get("fixed_provider") and not proactive.get("fixed_provider"):
                proactive["fixed_provider"] = advanced["fixed_provider"]
                changed = True
                logger.info("[Spark] migrated advanced.fixed_provider -> proactive_settings.fixed_provider")
            if advanced.get("history_depth") and not proactive.get("history_depth"):
                proactive["history_depth"] = advanced["history_depth"]
                changed = True
                logger.info("[Spark] migrated advanced.history_depth -> proactive_settings.history_depth")
            if advanced.get("persona_override") and not proactive.get("gen_persona_id"):
                proactive["persona_override_legacy"] = advanced["persona_override"]
                changed = True

            # special.provider -> proactive_settings.fixed_provider
            special = self.cfg.get("special")
            if isinstance(special, dict) and special.get("provider"):
                if not proactive.get("fixed_provider"):
                    proactive["fixed_provider"] = special["provider"]
                    changed = True
                    logger.info("[Spark] migrated special.provider -> proactive_settings.fixed_provider")

            # basic_settings.fixed_provider -> proactive_settings.fixed_provider
            if basic.get("fixed_provider") and not proactive.get("fixed_provider"):
                proactive["fixed_provider"] = basic["fixed_provider"]
                changed = True
                logger.info("[Spark] migrated basic_settings.fixed_provider -> proactive_settings.fixed_provider")

            if changed:
                self.cfg["proactive_settings"] = proactive
                self.cfg.save_config()
        except Exception as e:
            logger.debug(f"[Spark] config migration: {e}")

    async def _migrate_reminders_to_cron(self) -> str:
        """将旧版提醒迁移到 AstrBot 原生 cron 系统（幂等，重复执行安全）"""
        if not self._reminders:
            return "没有需要迁移的提醒。"
        cron_mgr = getattr(self.context, "cron_manager", None)
        if not cron_mgr:
            return "❌ cron_manager 不可用，无法迁移。"

        # 预取已有 jobs 用于幂等检查
        existing_jobs = await cron_mgr.list_jobs()
        existing_map = {j.name: j for j in existing_jobs}

        migrated = 0
        failed = 0
        for rid, reminder in list(self._reminders.items()):
            try:
                at = reminder.at
                umo = reminder.umo
                content = reminder.content

                job_name = f"spark_migrate_{rid}"
                template = self._get_cfg("reminders_settings", "reminder_prompt_template") or "提醒内容：{reminder_content}"
                note = template.replace("{reminder_content}", content)

                # 构建正确的 cron 表达式或一次性参数
                is_daily = "|daily" in at
                cron_expr = None
                run_at = None
                if is_daily:
                    hhmm = at.split("|", 1)[0]
                    t = _parse_hhmm(hhmm)
                    if not t:
                        failed += 1
                        continue
                    cron_expr = f"{t[1]} {t[0]} * * *"
                else:
                    try:
                        run_at = datetime.strptime(at, "%Y-%m-%d %H:%M")
                    except ValueError:
                        failed += 1
                        continue

                # 幂等检查：同名 job 已存在则删除重建（覆盖）
                existing = existing_map.get(job_name)
                if existing:
                    await cron_mgr.delete_job(existing.job_id)

                if is_daily:
                    await cron_mgr.add_active_job(
                        name=job_name,
                        cron_expression=cron_expr,
                        payload={"session": umo, "note": note, "origin": "spark_migrate"},
                        description=f"[Spark迁移] {content[:60]}",
                        run_once=False,
                    )
                else:
                    await cron_mgr.add_active_job(
                        name=job_name,
                        cron_expression=None,
                        payload={"session": umo, "note": note, "origin": "spark_migrate"},
                        description=f"[Spark迁移] {content[:60]}",
                        run_once=True,
                        run_at=run_at,
                    )
                migrated += 1

            except Exception as e:
                logger.error(f"[Spark] 迁移提醒 {rid} 失败: {e}")
                failed += 1

        parts = []
        if migrated > 0:
            parts.append(f"✅ 已迁移 {migrated} 个提醒到 AstrBot 原生定时任务")
            parts.append("旧数据已保留，可继续通过 /conversa remind 管理")
        if failed > 0:
            parts.append(f"❌ {failed} 个迁移失败")
        if not parts:
            parts.append("没有需要迁移的提醒。")
        return "\n".join(parts)

    async def initialize(self):
        """插件激活时的初始化方法（框架生命周期）"""
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

    def _get_int_cfg(self, group_key: str, sub_key: str, default: int) -> int:
        value = self._get_cfg(group_key, sub_key, None)
        if value is None or value == "":
            return int(default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    def _get_heat_args(self) -> tuple[float, float]:
        heat_cfg = self.cfg.get("heat_settings") or {}
        if not isinstance(heat_cfg, dict):
            heat_cfg = {}
        window_minutes = heat_cfg.get("heat_window_minutes")
        if window_minutes is None:
            window_minutes = float(heat_cfg.get("heat_window_hours", 4) or 4) * 60.0
        full_score_messages = float(heat_cfg.get("heat_messages_for_full_score", 10) or 10)
        return max(float(window_minutes or 1), 1.0), max(full_score_messages, 1.0)

    def _calc_heat(self, st: "SessionState", now_ts: float) -> float:
        window_m, full_score_messages = self._get_heat_args()
        return _compute_heat(st.msg_timestamps or [], now_ts, window_m, full_score_messages)

    def _calc_idle_delay(self, st: "SessionState", now_ts: float, profile: "UserProfile") -> float:
        """Return the base idle-greeting delay in minutes, applying heat scaling when enabled.

        When heat is disabled, falls back to the user/global idle_after_minutes setting.
        Fluctuation is applied by the caller.
        """
        heat_enabled = bool(self._get_cfg("heat_settings", "enable_heat", True))
        if heat_enabled:
            hot_m = float(self._get_cfg("heat_settings", "hot_delay_minutes", 30) or 30)
            cold_m = float(self._get_cfg("heat_settings", "cold_delay_minutes", 1200) or 1200)
            heat = self._calc_heat(st, now_ts)
            delay_m = hot_m + (cold_m - hot_m) * (1.0 - heat)
            logger.debug(f"[Spark] 热度计算: heat={heat:.2f}, delay={delay_m:.0f}m")
            return delay_m

        # heat disabled — use fixed setting
        if profile.idle_after_minutes is not None:
            return float(profile.idle_after_minutes)
        return float(self._get_cfg("idle_greetings", "idle_after_minutes", 45) or 45)

    # 数据持久化
    def _load_user_data(self):
        """加载用户配置和提醒事项（从 user_data.json）"""
        if not os.path.exists(self._user_data_path):
            return
        try:
            with open(self._user_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                profiles_data = data.get("profiles", {})
                for user_id, profile_dict in profiles_data.items():
                    self._user_profiles[user_id] = UserProfile.from_dict(profile_dict)
                logger.debug(f"[Spark] Loaded {len(self._user_profiles)} user profiles.")
                
                reminders_data = data.get("reminders", {})
                for reminder_id, reminder_dict in reminders_data.items():
                    self._reminders[reminder_id] = Reminder.from_dict(reminder_dict)
                logger.debug(f"[Spark] Loaded {len(self._reminders)} reminders.")
        
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"[Spark] Failed to load user data: {e}")
        except (IOError, OSError) as e:
            logger.error(f"[Spark] Failed to read user data file: {e}")
    
    def _save_user_data(self):
        """保存用户配置和提醒事项（到 user_data.json）"""
        try:
            profiles_dict = {uid: profile.to_dict() for uid, profile in self._user_profiles.items()}
            reminders_dict = {rid: reminder.to_dict() for rid, reminder in self._reminders.items()}
            data = {
                "profiles": profiles_dict,
                "reminders": reminders_dict
            }
            with open(self._user_data_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        except (IOError, OSError) as e:
            logger.error(f"[Spark] Failed to write user data file: {e}")
        except (TypeError, ValueError) as e:
            logger.error(f"[Spark] Failed to serialize user data: {e}")
    
    def _load_session_data(self):
        """加载运行时状态（从 session_data.json）"""
        if not os.path.exists(self._session_data_path):
            return
        try:
            with open(self._session_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                states_data = data.get("states", {})
                for conv_id, state_dict in states_data.items():
                    self._states[conv_id] = SessionState.from_dict(state_dict)
                logger.debug(f"[Spark] Loaded {len(self._states)} session states.")
        
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"[Spark] Failed to load session data: {e}")
        except (IOError, OSError) as e:
            logger.error(f"[Spark] Failed to read session data file: {e}")
    
    def _save_session_data(self):
        """保存运行时状态（到 session_data.json）"""
        try:
            states_dict = {cid: state.to_dict() for cid, state in self._states.items()}
            data = {"states": states_dict}
            with open(self._session_data_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        except (IOError, OSError) as e:
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
            config_subscribed_ids = self._get_cfg("basic_settings", "subscribed_users") or []
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
                        profile.manual_unsubscribe = True  # 标记为手动退订（WebUI操作视为手动）
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
                    logger.debug(f"[Spark] 已从配置同步 {len(config_subscribed_ids)} 个订阅用户ID")
                    subscribed_sessions = [user_id for user_id, profile in self._user_profiles.items() if profile.subscribed]
                    logger.debug(f"[Spark] 当前已订阅的会话数: {len(subscribed_sessions)}")
            
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
    
    @filter.event_message_type(filter.EventMessageType.ALL)
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

        # Determine if this is a real chat message vs. a slash command.
        # We cannot rely on startswith('/') because AstrBot strips wake_prefix
        # (which is '/') from event.message_str before passing to plugins.
        # Instead, check activated_handlers: if this event activated any of
        # Spark's own command handlers, it is a command, not real chat.
        message_text = (
            event.message_str.strip()
            if hasattr(event, "message_str") and event.message_str
            else ""
        )
        _activated = event.get_extra("activated_handlers") or []
        _spark_module = __name__  # 'data.plugins.astrbot_plugin_Spark.main'
        _is_spark_cmd = any(
            getattr(h, "handler_module_path", "") == _spark_module
            and getattr(h, "handler_name", "").startswith("_cmd_")
            for h in _activated
        )
        is_real_message = bool(message_text) and not _is_spark_cmd

        # Enhancement task cancellation is handled inside _delayed_enhancement
        # via last_user_reply_ts check. Do NOT cancel here — it would kill tasks
        # that are already past the sleep phase and executing LLM calls.

        # 保存旧的 last_user_reply_ts 用于判断是否是老用户
        old_last_user_reply_ts = st.last_user_reply_ts

        # 更新时间戳
        now_ts = _now_tz(self._get_cfg("basic_settings", "timezone") or None).timestamp()
        st.last_ts = now_ts
        if is_real_message:
            st.last_user_reply_ts = now_ts
            if not event.get_extra(self._heat_event_marker, False):
                if st.msg_timestamps is None:
                    st.msg_timestamps = []
                st.msg_timestamps.append(now_ts)
                if len(st.msg_timestamps) > 100:
                    st.msg_timestamps = st.msg_timestamps[-100:]
                event.set_extra(self._heat_event_marker, True)
        st.consecutive_no_reply_count = 0

        # 自动订阅模式：仅在首次创建用户时自动订阅
        if (self._get_cfg("basic_settings", "subscribe_mode") or "manual") == "auto":
            # 只在用户第一次发消息时（old_last_user_reply_ts == 0）自动订阅
            if old_last_user_reply_ts == 0 and not profile.manual_unsubscribe:
                profile.subscribed = True
                profile.auto_unsubscribed = False  # 清除自动退订标记
                logger.info(f"[Spark] 自动订阅模式：新用户 {umo} 已自动订阅")
                self._sync_subscribed_users_to_config()  # 同步到配置文件
        
        # 自动重新激活：仅对"被自动退订"的用户生效，手动退订的用户不会被自动重新激活
        if not profile.subscribed and profile.auto_unsubscribed and not profile.manual_unsubscribe:
            auto_resubscribe = bool(self._get_cfg("basic_settings", "auto_resubscribe", True))
            if auto_resubscribe:
                # 用户主动发消息，重新激活订阅
                profile.subscribed = True
                profile.auto_unsubscribed = False  # 清除自动退订标记
                logger.info(f"[Spark] 自动重新激活订阅: {umo} (用户在自动退订后主动聊天)")
                self._sync_subscribed_users_to_config()  # 同步到配置文件


        # 计算下一次延时问候触发时间（仅真实聊天消息触发，命令不重置倒计时）
        if is_real_message:
            try:
                if profile.subscribed and bool(self._get_cfg("idle_greetings", "enable_idle_greetings", True)):
                    delay_m = self._calc_idle_delay(st, now_ts, profile)
                    fluctuation_m = int(self._get_cfg("idle_greetings", "idle_random_fluctuation_minutes") or 15)
                    fluctuation_m = min(fluctuation_m, max(0, int(delay_m) - 1))
                    delay_m = max(1, delay_m + random.randint(-fluctuation_m, fluctuation_m))
                    st.next_idle_ts = now_ts + delay_m * 60
                    logger.debug(f"[Spark] 沉寂计时刷新(消息): {umo}, delay={delay_m:.0f}m, next={st.next_idle_ts:.0f}")
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
            umo = event.unified_msg_origin
            st = self._states.get(umo)
            if not st:
                return
            if event.get_extra(self._heat_event_marker, False):
                return
            now_ts = _now_tz(self._get_cfg("basic_settings", "timezone") or None).timestamp()
            st.last_user_reply_ts = now_ts
            if st.msg_timestamps is None:
                st.msg_timestamps = []
            st.msg_timestamps.append(now_ts)
            if len(st.msg_timestamps) > 100:
                st.msg_timestamps = st.msg_timestamps[-100:]
            event.set_extra(self._heat_event_marker, True)
            profile = self._user_profiles.get(umo)
            if profile and profile.subscribed and bool(self._get_cfg("idle_greetings", "enable_idle_greetings", True)):
                delay_m = self._calc_idle_delay(st, now_ts, profile)
                fluctuation_m = int(self._get_cfg("idle_greetings", "idle_random_fluctuation_minutes") or 15)
                fluctuation_m = min(fluctuation_m, max(0, int(delay_m) - 1))
                delay_m = max(1, delay_m + random.randint(-fluctuation_m, fluctuation_m))
                st.next_idle_ts = now_ts + delay_m * 60
                logger.debug(f"[Spark] 沉寂计时刷新(llm_request补偿): {umo}, delay={delay_m:.0f}m, next={st.next_idle_ts:.0f}")
            await self._debounced_save_session_data()
        except Exception as e:
            logger.debug(f"[Spark] _on_llm_request_update_ts 异常: {e}")

    @filter.on_llm_response()
    async def _on_llm_response_enhancement(self, event: AstrMessageEvent, _response=None):
        """对话增强：LLM 回复后检查是否应触发短期追回复"""
        try:
            # Skip proactive replies triggered by this plugin itself (CronMessageEvent)
            if HAS_AGENT_PIPELINE and isinstance(event, CronMessageEvent):
                return
            umo = event.unified_msg_origin
            st = self._states.get(umo)
            now_ts = _now_tz(self._get_cfg("basic_settings", "timezone") or None).timestamp()
            if st:
                st.last_ai_reply_ts = now_ts
                # Refresh idle greeting timer on every AI response
                profile = self._user_profiles.get(umo)
                if profile and profile.subscribed and bool(self._get_cfg("idle_greetings", "enable_idle_greetings", True)):
                    delay_m = self._calc_idle_delay(st, now_ts, profile)
                    fluctuation_m = int(self._get_cfg("idle_greetings", "idle_random_fluctuation_minutes") or 15)
                    fluctuation_m = min(fluctuation_m, max(0, int(delay_m) - 1))
                    delay_m = max(1, delay_m + random.randint(-fluctuation_m, fluctuation_m))
                    st.next_idle_ts = now_ts + delay_m * 60
                    logger.debug(f"[Spark] 沉寂计时刷新(AI回复): {umo}, delay={delay_m:.0f}m, next={st.next_idle_ts:.0f}")
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
        '''管理主动对话功能。当用户希望你能主动找他聊天、保持联系时开启；当用户明确不需要时关闭。

        Args:
            action(string): "on" 开启主动对话, "off" 关闭主动对话
        '''
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
        command_parts = text.lstrip('/').split()
        if not command_parts:
            return
        
        # 提取真实命令和参数
        args_str = " ".join(command_parts[1:]) if len(command_parts) > 1 else ""
        
        # 将参数字符串分割成子命令和值
        args = args_str.split()
        sub_command = args[0] if args else ""

        # Chinese-to-English sub-command aliases
        _sub_alias = {
            "订阅": "watch", "退订": "unwatch",
            "开启": "on", "关闭": "off",
            "设置": "set",
            "帮助": "help",
        }
        sub_command = _sub_alias.get(sub_command, sub_command)

        # Chinese target aliases for "set" sub-command
        if sub_command == "set" and len(args) >= 2:
            _target_alias = {
                "免打扰": "quiet", "沉寂": "after",
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
            debug_info.append(f"用户订阅状态: {profile.subscribed if profile else False}")
            
            # 显示订阅/退订状态标记
            if profile:
                if profile.manual_unsubscribe:
                    debug_info.append("退订类型: 手动退订（强制，不会自动重新激活）")
                elif profile.auto_unsubscribed:
                    debug_info.append("退订类型: 自动退订（可自动重新激活）")
                elif profile.subscribed:
                    debug_info.append("订阅类型: 正常订阅")
            
            debug_info.append(f"用户专属免打扰: {profile.quiet_hours if profile and profile.quiet_hours else '未设置(使用全局)'}")
            debug_info.append(f"全局免打扰时间: {self._get_cfg('basic_settings', 'quiet_hours', '未设置')}")
            debug_info.append(f"延时基准: {self._get_cfg('idle_greetings', 'idle_after_minutes', 0)}分钟")
            debug_info.append(f"最大无回复天数: {self._get_cfg('basic_settings', 'max_no_reply_days', 0)}")
            debug_info.append(f"自动重新激活: {bool(self._get_cfg('basic_settings', 'auto_resubscribe', True))}")

            # Heat debug info
            st_debug = self._states.get(umo)
            heat_enabled = bool(self._get_cfg("heat_settings", "enable_heat", True))
            if heat_enabled and st_debug:
                _window_m, _full_score_messages = self._get_heat_args()
                _heat_val = self._calc_heat(st_debug, _now_tz(self._get_cfg("basic_settings", "timezone") or None).timestamp())
                _hot_m = float(self._get_cfg("heat_settings", "hot_delay_minutes", 30) or 30)
                _cold_m = float(self._get_cfg("heat_settings", "cold_delay_minutes", 1200) or 1200)
                _next_delay = _hot_m + (_cold_m - _hot_m) * (1.0 - _heat_val)
                if _heat_val >= 0.6:
                    _heat_label = "热"
                elif _heat_val >= 0.2:
                    _heat_label = "温"
                else:
                    _heat_label = "冷"
                debug_info.append(f"对话热度: {_heat_label}({_heat_val:.2f}) → 下次触发延迟约 {_next_delay:.0f} 分钟")
                debug_info.append(f"热度窗口: {int(_window_m)} 分钟，满热约需 {_full_score_messages:.0f} 条消息，记录消息数: {len(st_debug.msg_timestamps or [])}")
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
                        greetings.append({"enable": False, "time": "", "prompt": "", "ignore_dnd": False})
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
                    if self._is_admin(event) and len(args) > 3 and args[3].lower() == "global":
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
                yield reply("🧵 历史条数已废弃，请使用「判断轮数」和「生成轮数」配置项替代。")
                return
            
            yield reply(f"❌ 未知的 set 目标 '{target}'。可用: after, daily[1-3], quiet。")
            return

        # migrate-reminders 命令（管理员）
        if sub_command == "migrate-reminders":
            if not self._is_admin(event):
                yield reply("错误：此命令仅限管理员使用。")
                return
            result = await self._migrate_reminders_to_cron()
            yield reply(result)
            return

        # remind 命令（旧功能，推荐使用 AstrBot 原生定时提醒）
        if sub_command == "remind":
            if not bool(self._get_cfg("reminders_settings", "enable_reminders", True)):
                yield reply("提醒功能已被管理员禁用。\n💡 推荐直接对 AI 说「提醒我...」使用 AstrBot 原生定时提醒。")
                return
            
            remind_sub_command = args[1].lower() if len(args) > 1 else ""

            if remind_sub_command == "list":
                list_text = self._remind_list_text(event.unified_msg_origin)
                yield reply(f"{list_text}\n\n💡 提示：推荐直接对 AI 说「提醒我...」使用 AstrBot 原生定时提醒。")
                return
            
            if remind_sub_command == "del" and len(args) >= 3:
                # 支持通过序号或 ID 删除
                identifier = args[2].strip()
                umo = event.unified_msg_origin
                
                # 尝试解析为序号（整数）
                try:
                    index = int(identifier)
                    # 获取用户的提醒列表并排序
                    user_reminders = self._get_user_reminders_sorted(umo)
                    if 1 <= index <= len(user_reminders):
                        rid = user_reminders[index - 1].id  # 序号从 1 开始
                        del self._reminders[rid]
                        self._save_user_data()
                        yield reply(f"🗑️ 已删除提醒 #{index}")
                    else:
                        yield reply(f"❌ 序号超出范围，当前共有 {len(user_reminders)} 个提醒")
                    return
                except ValueError:
                    # 不是数字，尝试作为 ID 删除（向后兼容）
                    rid = identifier
                    if rid in self._reminders and self._reminders[rid].umo == umo:
                        del self._reminders[rid]
                        self._save_user_data()
                        yield reply(f"🗑️ 已删除提醒 {rid}")
                    else:
                        yield reply("❌ 未找到该提醒，请使用 `/conversa remind list` 查看可用序号")
                return
            
            if remind_sub_command == "add":
                remind_content = " ".join(args[2:])
                # 匹配 HH:MM 格式
                m_daily = re.match(r"^(\d{1,2}:\d{2})\s+(.+)$", remind_content)
                # 匹配 YYYY-MM-DD HH:MM 格式
                m_once = re.match(r"^(\d{4}-\d{2}-\d{2}\s+\d{1,2}:\d{2})\s+(.+)$", remind_content)
                
                rid = f"R{int(datetime.now().timestamp())}"
                
                if m_once:
                    at_time, content = m_once.groups()
                    self._reminders[rid] = Reminder(
                        id=rid,
                        umo=event.unified_msg_origin,
                        content=content.strip(),
                        at=at_time.strip(),
                        created_at=datetime.now().timestamp()
                    )
                    self._save_user_data()
                    yield reply(f"⏰ 已添加一次性提醒 {rid}\n💡 提示：推荐直接对 AI 说「提醒我...」使用 AstrBot 原生定时提醒。")
                    return
                elif m_daily:
                    hhmm, content = m_daily.groups()
                    self._reminders[rid] = Reminder(
                        id=rid,
                        umo=event.unified_msg_origin,
                        content=content.strip(),
                        at=f"{hhmm}|daily",
                        created_at=datetime.now().timestamp()
                    )
                    self._save_user_data()
                    yield reply(f"⏰ 已添加每日提醒 {rid}\n💡 提示：推荐直接对 AI 说「提醒我...」使用 AstrBot 原生定时提醒。")
                    return
            
            yield reply(self._help_text())
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
            yield event.plain_result("未配置沉寂问候模板，请先在设置中配置 idle_prompt_templates")
            return

        yield event.plain_result("正在发送主动消息...")
        tz = self._get_cfg("basic_settings", "timezone") or None
        await self._proactive_reply(umo, tz, random.choice(idle_prompts), skip_judge=True)

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

        if profile:
            sub_status = "已订阅" if profile.subscribed else "未订阅"
            lines.append(f"订阅状态: {sub_status}")
            if profile.manual_unsubscribe:
                lines.append("退订类型: 手动退订")
            elif profile.auto_unsubscribed:
                lines.append("退订类型: 自动退订（可自动恢复）")
            if profile.quiet_hours:
                lines.append(f"专属免打扰: {profile.quiet_hours}")
        else:
            lines.append("订阅状态: 未订阅")

        global_quiet = self._get_cfg("basic_settings", "quiet_hours", "") or ""
        if global_quiet:
            lines.append(f"全局免打扰: {global_quiet}")

        is_busy = getattr(self.context, '_busy_schedule_is_busy', False)
        lines.append(f"忙碌时段: {'是' if is_busy else '否'}")

        heat_enabled = bool(self._get_cfg("heat_settings", "enable_heat", True))
        if heat_enabled:
            window_m, _full_score_messages = self._get_heat_args()
            now_ts = now.timestamp()
            heat_val = self._calc_heat(st, now_ts) if st else 0.0
            if heat_val >= 0.6:
                heat_label = "热"
            elif heat_val >= 0.2:
                heat_label = "温"
            else:
                heat_label = "冷"
            window_sec = float(window_m) * 60.0
            recent_msg_count = 0
            if st and st.msg_timestamps:
                recent_msg_count = sum(1 for ts in st.msg_timestamps if 0 <= now_ts - ts <= window_sec)
            lines.append(f"当前热度: {heat_label}({heat_val:.2f})，{int(window_m)}分钟内 {recent_msg_count} 条消息")
        else:
            lines.append("当前热度: 已关闭（使用固定沉寂延迟）")

        if st and st.last_user_reply_ts > 0:
            delta = now.timestamp() - st.last_user_reply_ts
            lines.append(f"距上次聊天: {_format_time_delta(delta)}")

        judge_enabled = self._get_cfg("proactive_settings", "proactive_judge_enable", True)
        lines.append(f"智能判断: {'开启' if judge_enabled else '关闭'}")

        # --- 待触发任务 ---
        pending = []
        if st and st.next_idle_ts > 0:
            remaining = st.next_idle_ts - now.timestamp()
            if remaining > 0:
                pending.append(f"  沉寂问候 → 约 {int(remaining / 60)} 分钟后")
            else:
                pending.append("  沉寂问候 → 等待触发条件")

        if st and st.next_enhancement_ts > 0:
            remaining_enh = st.next_enhancement_ts - now.timestamp()
            if remaining_enh > 0:
                pending.append(f"  对话增强 → 约 {int(remaining_enh / 60)} 分钟后")

        daily_slots = self._parse_daily_slots(now)
        for idx, actual_time, tag, slot_cfg in daily_slots:
            if st and st.has_fired(tag):
                continue
            slot_dt = now.replace(hour=actual_time[0], minute=actual_time[1], second=0, microsecond=0)
            diff_sec = (slot_dt - now).total_seconds()
            if diff_sec > 0:
                diff_min = int(diff_sec / 60)
                pending.append(f"  每日问候 {actual_time[0]:02d}:{actual_time[1]:02d} → 约 {diff_min} 分钟后")
            else:
                pending.append(f"  每日问候 {actual_time[0]:02d}:{actual_time[1]:02d} → 等待触发条件")

        if pending:
            lines.append("待触发任务:")
            lines.extend(pending)
        else:
            lines.append("待触发任务: 无")

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
            "/灵犀 提醒 添加 <时间> <内容> - 添加提醒\n"
            "/灵犀 提醒 列表 - 查看提醒列表\n"
            "/灵犀 提醒 删除 <序号> - 删除提醒\n"
            "/灵犀 帮助 - 显示本帮助"
        )

    def _get_user_reminders_sorted(self, umo: str) -> List[Reminder]:
        """获取指定用户的提醒列表并排序"""
        arr = [r for r in self._reminders.values() if r.umo == umo]
        arr.sort(key=lambda x: x.created_at)
        return arr
    
    def _remind_list_text(self, umo: str) -> str:
        """生成指定用户的提醒列表文本（显示序号）"""
        arr = self._get_user_reminders_sorted(umo)
        if not arr:
            return "暂无提醒"
        lines = []
        for idx, r in enumerate(arr, start=1):
            # 格式化时间显示
            time_display = r.at.replace("|daily", " (每日)")
            lines.append(f"{idx}. {time_display} | {r.content}")
        # 使用换行符连接，确保每个提醒单独一行
        # 提示信息放在末尾，避免某些消息平台过滤括号内容
        return "提醒列表：\n" + "\n".join(lines)

    # 对话增强（短期随机追回复）

    def _should_trigger_enhancement(self, umo: str) -> bool:
        """判断是否应该触发对话增强"""
        try:
            if not self.cfg.get("enable", True):
                logger.debug("[Spark] 对话增强跳过: 插件已禁用")
                return False
            
            enable_val = self._get_cfg("enhancement", "enable_enhancement", False)
            if not bool(enable_val):
                logger.debug(f"[Spark] 对话增强跳过: enable_enhancement={enable_val} (raw cfg enhancement={self.cfg.get('enhancement')})")
                return False
            
            # 对话增强仅私聊生效
            if "GroupMessage" in umo:
                logger.debug("[Spark] 对话增强跳过: 群聊不触发")
                return False
            
            profile = self._user_profiles.get(umo)
            if not profile or not profile.subscribed:
                logger.debug(f"[Spark] 对话增强跳过: 用户未订阅 (profile={profile is not None}, subscribed={profile.subscribed if profile else 'N/A'})")
                return False
            
            # 调度时不检查免打扰（用户刚发了消息说明在线），执行时再检查
            
            # 已有待执行的增强任务
            if umo in self._enhancement_tasks and not self._enhancement_tasks[umo].done():
                logger.debug("[Spark] 对话增强跳过: 已有待执行任务")
                return False
            
            # Calculate trigger probability
            base_prob = int(self._get_cfg("enhancement", "enhancement_probability") or 20)
            st = self._states.get(umo)
            if not st:
                logger.debug("[Spark] 对话增强跳过: 无 SessionState")
                return False

            roll = random.random() * 100
            triggered = roll < base_prob

            if triggered:
                logger.info(f"[Spark] 对话增强触发: {umo} (概率={base_prob}%, roll={roll:.2f})")
            else:
                logger.debug(f"[Spark] 对话增强未触发: {umo} (概率={base_prob}%, roll={roll:.2f})")
            
            return triggered
        except Exception as e:
            logger.error(f"[Spark] 对话增强判断出错: {e}")
            return False

    def _schedule_enhancement(self, umo: str, current_round: Optional[dict] = None):
        """调度一个延迟的对话增强任务"""
        min_delay = self._get_int_cfg("enhancement", "enhancement_min_delay", 30)
        max_delay = min(self._get_int_cfg("enhancement", "enhancement_max_delay", 1800), 1800)
        min_delay = max(min_delay, 0)
        max_delay = max(max_delay, 0)
        if min_delay > max_delay:
            min_delay = max_delay
        delay = random.randint(min_delay, max_delay)
        
        gen = self._enhancement_gen.get(umo, 0)
        logger.info(f"[Spark] 已调度对话增强: {umo}, {delay}秒后执行")
        task = asyncio.create_task(
            self._delayed_enhancement(umo, delay, gen, current_round=current_round)
        )
        self._enhancement_tasks[umo] = task
        st = self._states.get(umo)
        if st:
            import time as _time
            st.next_enhancement_ts = _time.time() + delay

    async def _delayed_enhancement(
        self,
        umo: str,
        delay: int,
        gen: int,
        current_round: Optional[dict] = None,
    ):
        """延迟执行对话增强回复"""
        try:
            logger.info(f"[Spark] 增强任务开始: {umo}, 等待{delay}秒")
            st = self._states.get(umo)
            if not st:
                logger.info(f"[Spark] 增强任务退出: {umo} (无SessionState)")
                return
            trigger_chat_ts = max(st.last_user_reply_ts, st.last_ai_reply_ts, st.last_proactive_reply_ts)
            logger.info(f"[Spark] 增强任务sleep: {umo}, trigger_chat_ts={trigger_chat_ts}")

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
            now_ts = now.timestamp()
            latest_chat_ts = max(st.last_user_reply_ts, st.last_ai_reply_ts, st.last_proactive_reply_ts)
            if latest_chat_ts > trigger_chat_ts + 1.0:
                logger.info(f"[Spark] 增强任务退出: {umo} (等待期间已有新聊天)")
                return
            cooldown = max(self._get_int_cfg("enhancement", "enhancement_cooldown_minutes", 12), 0) * 60
            if cooldown > 0 and latest_chat_ts > 0 and now_ts - latest_chat_ts < cooldown:
                logger.info(f"[Spark] 增强任务退出: {umo} (距最近聊天不足冷却时间, {int(now_ts - latest_chat_ts)}s < {cooldown}s)")
                return
            quiet = self._get_cfg("basic_settings", "quiet_hours", "") or ""
            user_quiet = profile.quiet_hours if profile.quiet_hours else quiet
            if _in_quiet(now, user_quiet):
                logger.info(f"[Spark] 增强任务退出: {umo} (免打扰时段)")
                return
            
            # busy_schedule 快速退出场景等同免打扰：先触发一次即时状态刷新，再读标记
            _force = getattr(self.context, '_busy_schedule_force_check', None)
            if _force:
                try:
                    await _force()
                except Exception:
                    pass
            is_busy_flag = getattr(self.context, '_busy_schedule_is_busy', False)
            if is_busy_flag:
                logger.info(f"[Spark] 增强任务退出: {umo} (忙碌时段, flag={is_busy_flag})")
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
                judge_current_round=current_round,
            )
            if ok:
                logger.info(f"[Spark] 对话增强回复成功: {umo}")
            
        except asyncio.CancelledError:
            logger.debug(f"[Spark] 对话增强任务被取消: {umo}")
        except Exception as e:
            logger.error(f"[Spark] 对话增强执行出错({umo}): {e}")
        finally:
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
        6. 检查并触发提醒事项
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
        reply_interval = int(self._get_cfg("basic_settings", "reply_interval_seconds") or 10)

        # 解析每日定时配置（修复：使用 slot1/slot2/slot3 而非 time1/time2/time3）
        daily_slots = self._parse_daily_slots(now)

        # Refresh busy state once before the per-user loop
        _force = getattr(self.context, '_busy_schedule_force_check', None)
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
                is_busy = getattr(self.context, '_busy_schedule_is_busy', False)

                st = self._states.get(umo)
                if st and await self._should_auto_unsubscribe(umo, profile, st, now):
                    continue

                # Idle greetings: skip during DND/busy
                if not is_in_dnd and not is_busy:
                    await self._check_idle_greeting(umo, st, now, tz, reply_interval)

                # Daily greetings: ignore_dnd items bypass DND/busy
                await self._check_daily_greetings(
                    umo, st, profile, now, daily_slots, tz, reply_interval,
                    is_in_dnd=is_in_dnd, is_busy=is_busy,
                )
            except Exception as e:
                logger.error(f"[Spark] 处理用户 {umo} 的 tick 任务时发生错误: {e}", exc_info=True)
                continue  # 继续处理下一个用户，不影响整体调度

        # 检查提醒
        await self._check_reminders(now, tz, reply_interval)
        # 调度器结束时使用去抖保存，减少磁盘I/O
        await self._debounced_save_session_data()

    def _parse_daily_slots(self, now: datetime) -> List[Tuple[int, Optional[Tuple[int, int]], str, dict]]:
        """
        Parse daily greetings config. Supports:
        1. New list format: daily_greetings = [{enable, time, prompt, ignore_dnd}, ...]
        2. Legacy slot format: slot1/slot2/slot3 or daily1_enable/time1/prompt1
        Returns: [(slot_num, time_tuple, tag, slot_cfg), ...]
        """
        daily = self.cfg.get("daily_prompts") or {}
        slots_info = []

        # New list format takes priority
        greetings_list = daily.get("daily_greetings", [])
        if isinstance(greetings_list, list) and greetings_list:
            for idx, item in enumerate(greetings_list):
                if not isinstance(item, dict) or not item.get("enable", False):
                    continue
                time_str = item.get("time", "")
                prompt_str = item.get("prompt", "")
                ignore_dnd = item.get("ignore_dnd", False)
                jitter_minutes = max(0, int(item.get("jitter_minutes", 0) or 0))
                time_tuple = _parse_hhmm(time_str)
                if time_tuple:
                    if jitter_minutes > 0:
                        # Stable per-day offset: same seed → same offset every minute of the day
                        _rng = random.Random(now.toordinal() * 1000 + idx)
                        offset = _rng.randint(-jitter_minutes, jitter_minutes)
                        total = max(0, min(time_tuple[0] * 60 + time_tuple[1] + offset, 23 * 60 + 59))
                        actual_time = (total // 60, total % 60)
                    else:
                        actual_time = time_tuple
                    tag = f"daily_{idx}@{now.strftime('%Y-%m-%d')} {actual_time[0]:02d}:{actual_time[1]:02d}"
                    slots_info.append((idx, actual_time, tag, {
                        "prompt": prompt_str, "ignore_dnd": ignore_dnd,
                    }))
            return slots_info

        # Legacy slot format (slot1/slot2/slot3)
        for slot_num in [1, 2, 3]:
            slot_cfg = daily.get(f"slot{slot_num}", {})
            if slot_cfg:
                if slot_cfg.get("enable", False):
                    time_str = slot_cfg.get("time", "")
                    prompt_str = slot_cfg.get("prompt", "")
                    time_tuple = _parse_hhmm(time_str)
                    if time_tuple:
                        tag = f"daily{slot_num}@{now.strftime('%Y-%m-%d')} {time_tuple[0]:02d}:{time_tuple[1]:02d}"
                        slots_info.append((slot_num, time_tuple, tag, {"prompt": prompt_str}))
            else:
                enable_key = f"daily{slot_num}_enable"
                time_key = f"time{slot_num}"
                prompt_key = f"prompt{slot_num}"
                if daily.get(enable_key, False):
                    time_str = daily.get(time_key, "")
                    prompt_str = daily.get(prompt_key, "")
                    time_tuple = _parse_hhmm(time_str)
                    if time_tuple:
                        tag = f"daily{slot_num}@{now.strftime('%Y-%m-%d')} {time_tuple[0]:02d}:{time_tuple[1]:02d}"
                        slots_info.append((slot_num, time_tuple, tag, {"prompt": prompt_str}))

        return slots_info

    async def _check_idle_greeting(self, umo: str, st: Optional[SessionState], now: datetime, 
                                   tz: Optional[str], reply_interval: int):
        """检查并触发延时问候"""
        if not bool(self._get_cfg("idle_greetings", "enable_idle_greetings", True)):
            logger.debug(f"[Spark] 沉寂问候跳过: {umo} (enable_idle_greetings=False)")
            return
        
        if not st:
            logger.debug(f"[Spark] 沉寂问候跳过: {umo} (无SessionState)")
            return
        
        # 向后兼容：如果 next_idle_ts 未设置或为0，自动初始化
        if not st.next_idle_ts or st.next_idle_ts <= 0:
            profile = self._user_profiles.get(umo)
            if profile and profile.subscribed:
                delay_m = profile.idle_after_minutes
                if delay_m is None:
                    base_delay_m = int(self._get_cfg("idle_greetings", "idle_after_minutes") or 45)
                    fluctuation_m = int(self._get_cfg("idle_greetings", "idle_random_fluctuation_minutes") or 15)
                    fluctuation_m = min(fluctuation_m, max(0, base_delay_m - 1))
                    delay_m = max(1, base_delay_m + random.randint(-fluctuation_m, fluctuation_m))
                
                # Base on last activity time, but never set a timestamp already in the past
                base_ts = st.last_ts if st.last_ts > 0 else now.timestamp()
                computed = base_ts + delay_m * 60
                st.next_idle_ts = computed if computed > now.timestamp() else now.timestamp() + delay_m * 60
                logger.info(f"[Spark] 沉寂问候初始化计时: {umo}, delay={delay_m}m, 将在 {st.next_idle_ts:.0f} 触发")
                await self._debounced_save_session_data()
                return  # 本次不触发，等下次检查
        
        if now.timestamp() < st.next_idle_ts:
            return
        
        tag = f"idle@{now.strftime('%Y-%m-%d %H:%M')}"
        if st.has_fired(tag):
            logger.debug(f"[Spark] 沉寂问候跳过: {umo} (本分钟已触发 tag={tag})")
            return
        
        idle_prompts = self._get_cfg("idle_greetings", "idle_prompt_templates") or []
        if not idle_prompts:
            logger.warning(f"[Spark] 沉寂问候跳过: {umo} (idle_prompt_templates 未配置)")
            return
        
        prompt_template = random.choice(idle_prompts)
        logger.info(f"[Spark] 触发延时问候 {umo}")
        ok = await self._proactive_reply(umo, tz, prompt_template)
        st.mark_fired(tag)
        if ok:
            st.next_idle_ts = 0.0
            if reply_interval > 0:
                await asyncio.sleep(reply_interval)
        else:
            st.consecutive_no_reply_count += 1
            st.next_idle_ts = 0.0  # judge 拒绝也视为本次任务结束，等下轮沉寂重新计时

    async def _check_daily_greetings(self, umo: str, st: Optional[SessionState], profile: UserProfile,
                                     now: datetime, daily_slots: List[Tuple],
                                     tz: Optional[str], reply_interval: int,
                                     is_in_dnd: bool = False, is_busy: bool = False):
        """检查并触发每日定时问候（支持 ignore_dnd 跳过免打扰/忙碌）"""
        if not bool(self.cfg.get("enable_daily_greetings", True)) or not profile.daily_reminders_enabled:
            return
        
        if not st:
            return
        
        for slot_num, slot_time, tag, slot_cfg in daily_slots:
            if slot_time and now.hour == slot_time[0] and now.minute == slot_time[1]:
                if st.has_fired(tag):
                    continue
                
                # ignore_dnd=true 的时段不受免打扰/忙碌限制
                if not slot_cfg.get("ignore_dnd", False):
                    if is_in_dnd or is_busy:
                        continue

                _prompt_raw = slot_cfg.get("prompt", "")
                if isinstance(_prompt_raw, list):
                    import random as _random
                    prompt_template = _random.choice(_prompt_raw) if _prompt_raw else ""
                else:
                    prompt_template = _prompt_raw
                if prompt_template:
                    logger.info(f"[Spark] 触发每日定时{slot_num}回复 {umo} (ignore_dnd={slot_cfg.get('ignore_dnd', False)})")
                    # When ignore_dnd overrides busy state: wake AI, flush queued messages first
                    if slot_cfg.get("ignore_dnd", False) and is_busy:
                        flush_delay = int(self._get_cfg("daily_prompts", "ignore_busy_flush_delay_seconds") or 10)
                        wake_fn = getattr(self.context, "_busy_schedule_wake_and_flush", None)
                        if wake_fn:
                            try:
                                await wake_fn(umo)
                            except Exception as e:
                                logger.warning(f"[Spark] wake_and_flush 失败: {e}")
                        await asyncio.sleep(flush_delay)
                    ok = await self._proactive_reply(umo, tz, prompt_template, skip_judge=True)
                    if ok:
                        st.mark_fired(tag)
                        if reply_interval > 0:
                            await asyncio.sleep(reply_interval)
                    else:
                        st.consecutive_no_reply_count += 1
                break  # 同一分钟只触发一个定时任务

    async def _should_auto_unsubscribe(self, umo: str, profile: UserProfile, st: SessionState, now: datetime) -> bool:
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
                logger.info(f"[Spark] 自动退订 {umo}：用户{days_since_reply}天未回复（可自动重新激活）")
                self._save_user_data()
                self._sync_subscribed_users_to_config()  # 同步到配置文件
                return True

        return False

    async def _check_reminders(self, now: datetime, tz: Optional[str], reply_interval: int):
        """检查并触发到期的提醒事项"""
        if not bool(self._get_cfg("reminders_settings", "enable_reminders", True)):
            return
        
        fired_ids = []
        for rid, r in list(self._reminders.items()):
            try:
                # 检查用户订阅状态
                profile = self._user_profiles.get(r.umo)
                if not profile or not profile.subscribed:
                    continue
                
                st = self._states.get(r.umo)
                if not st:
                    logger.warning(f"[Spark] Reminder check skipped for {r.umo}: no session state found.")
                    continue

                if "|daily" in r.at:
                    hhmm = r.at.split("|", 1)[0]
                    t = _parse_hhmm(hhmm)
                    if not t:
                        continue
                    
                    if now.hour == t[0] and now.minute == t[1]:
                        # 为每日提醒创建唯一标记（每天一个）
                        tag = f"remind_daily_{r.id}@{now.strftime('%Y-%m-%d')}"
                        if not st.has_fired(tag):
                            logger.info(f"[Spark] Firing daily reminder {r.id} for {r.umo}")
                            ok = await self._proactive_reminder_reply(r.umo, r.content)
                            if ok:
                                st.mark_fired(tag)  # 记录已触发
                                if reply_interval > 0:
                                    await asyncio.sleep(reply_interval)
                else:
                    # 一次性提醒：比较时间字符串（精确到分钟）
                    try:
                        # 使用字符串比较，避免时区问题
                        reminder_time_str = r.at  # 格式: "YYYY-MM-DD HH:MM"
                        now_time_str = now.strftime("%Y-%m-%d %H:%M")
                        
                        # 使用字符串比较，当前时间 >= 提醒时间即触发
                        if now_time_str >= reminder_time_str:
                            # 为一次性提醒创建唯一标记（防止重复）
                            tag = f"remind_once_{r.id}@{reminder_time_str}"
                            if not st.has_fired(tag):
                                logger.info(f"[Spark] Firing one-time reminder {r.id} for {r.umo} (due: {r.at}, now: {now_time_str})")
                                ok = await self._proactive_reminder_reply(r.umo, r.content)
                                # 无论发送成功与否，一次性提醒都应该被删除，避免无限重试
                                st.mark_fired(tag)
                                fired_ids.append(rid)
                                if not ok:
                                    logger.warning(f"[Spark] One-time reminder {r.id} failed to send, but will be deleted to prevent infinite retry")
                                if reply_interval > 0:
                                    await asyncio.sleep(reply_interval)
                    except Exception as e:
                        logger.warning(f"[Spark] Error processing one-time reminder {r.id}: {e}")
                        continue
            except Exception as e:
                logger.error(f"[Spark] Error checking reminder {r.id}: {e}")
                continue
        
        if fired_ids:
            for rid in fired_ids:
                self._reminders.pop(rid, None)
            self._save_user_data()
    
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

    async def _judge_should_reply(
        self,
        umo: str,
        tz: Optional[str],
        current_round: Optional[dict] = None,
    ) -> bool:
        """Step 1: Lightweight LLM call to decide whether to send a proactive reply."""
        try:
            await self._refresh_realtime_context()
            now = _now_tz(tz)
            time_fmt = self._get_cfg("basic_settings", "time_format") or "%Y-%m-%d %H:%M"
            now_str = now.strftime(time_fmt)

            st = self._states.get(umo)
            time_since_last_chat = "未知"
            if st:
                _last_chat_ts = max(st.last_user_reply_ts, st.last_proactive_reply_ts, st.last_ai_reply_ts)
                if _last_chat_ts > 0:
                    time_since_last_chat = _format_time_delta(now.timestamp() - _last_chat_ts)

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
                    raw_judge_contexts
                )
                latest_official = official_rounds[-1] if official_rounds else None
                already_present = bool(
                    latest_official
                    and not latest_official["proactive"]
                    and latest_official["user"] == pending_user
                    and "\n".join(latest_official["assistant"]) == pending_ai
                )
                if pending_user and pending_ai and not already_present:
                    raw_judge_contexts.extend(
                        [
                            {"role": "user", "content": pending_user},
                            {"role": "assistant", "content": pending_ai},
                        ]
                    )
                    current_round_source = "pending"
            judge_contexts, selected_judge_rounds = (
                self._select_recent_round_contexts(
                    raw_judge_contexts,
                    judge_history_rounds,
                )
            )
            proactive_rounds = sum(
                turn["proactive"] for turn in selected_judge_rounds
            )
            newest_type = (
                "proactive"
                if selected_judge_rounds and selected_judge_rounds[-1]["proactive"]
                else "normal" if selected_judge_rounds else "none"
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

            judge_template = self._get_cfg("proactive_settings", "proactive_judge_prompt") or ""
            if not judge_template:
                judge_template = (
                    '日程：{today_schedule}\n当前活动：{current_activity}\n'
                    '用户节律：{time_period_prompt}\n距上次聊天：{time_since_last_chat}'
                )
            judge_rules = self._get_cfg("proactive_settings", "proactive_judge_rules") or ""
            if not judge_rules:
                judge_rules = '！！必须遵守！！：你只能输出一个字："是"或"否"，不允许输出任何其他字。'
            today_schedule = getattr(self.context, "_busy_schedule_today_schedule", "")
            outfit = getattr(self.context, "_busy_schedule_outfit", "")
            current_activity = getattr(self.context, "_busy_schedule_current_activity", "")
            next_activity = getattr(self.context, "_busy_schedule_next_activity", "")
            custom_prompt = getattr(self.context, "_busy_schedule_custom_prompt", "")
            _get_prompt = getattr(self.context, "_time_period_get_prompt", None)
            time_period_prompt = _get_prompt() if callable(_get_prompt) else getattr(self.context, "_time_period_current_prompt", "")

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

            try:
                judge_prompt = judge_template.format(
                    now=now_str, last_user=last_user, last_ai=last_ai,
                    time_since_last_chat=time_since_last_chat, umo=umo,
                    today_schedule=today_schedule, outfit=outfit,
                    current_activity=current_activity, next_activity=next_activity,
                    custom_prompt=custom_prompt, time_period_prompt=time_period_prompt,
                    heat_level=heat_level,
                )
            except KeyError as e:
                logger.warning(f"[Spark] Judge prompt format error: {e}")
                judge_prompt = judge_template

            judge_provider_id = self._get_cfg("proactive_settings", "proactive_judge_provider") or ""
            provider = None
            if judge_provider_id:
                provider = self.context.get_provider_by_id(judge_provider_id)
            if not provider:
                provider = self.context.get_using_provider(umo=umo)
            if not provider:
                return True

            judge_persona = self._resolve_persona("proactive_settings", "judge_persona_id")
            if not judge_persona:
                judge_persona = await self._get_current_persona_prompt(umo)
            if not judge_persona:
                judge_persona = "你是一个对话判断助手，只回复是或否"

            _JUDGE_RETRIES = 3
            _RETRYABLE = (502, 503, 504)
            last_err = None
            for attempt in range(_JUDGE_RETRIES):
                try:
                    llm_resp = await provider.text_chat(
                        prompt=None,
                        contexts=judge_contexts + [
                            {"role": "user", "content": judge_prompt},
                            {"role": "user", "content": judge_rules},
                        ],
                        system_prompt=judge_persona,
                    )
                    response = (llm_resp.completion_text if hasattr(llm_resp, "completion_text") else "").strip()
                    if not response:
                        raise ValueError("Empty completion text")

                    should_reply = "是" in response[:10]
                    if should_reply:
                        logger.info(f"[Spark] Judge YES for {umo}: '{response[:20]}'")
                    else:
                        logger.info(f"[Spark] Judge NO for {umo}: '{response[:20]}'")
                    return should_reply

                except Exception as e:
                    last_err = e
                    is_retryable = False
                    err_str = str(e)
                    if any(code in err_str for code in ("502", "503", "504")):
                        is_retryable = True
                    if "no usable output" in err_str.lower() or "empty" in err_str.lower():
                        is_retryable = True
                    if "timeout" in err_str.lower() or "connect" in err_str.lower():
                        is_retryable = True

                    if is_retryable and attempt < _JUDGE_RETRIES - 1:
                        wait = 2 ** (attempt + 1)
                        logger.warning(f"[Spark] Judge retry {attempt + 1}/{_JUDGE_RETRIES} for {umo}: {e}, waiting {wait}s")
                        await asyncio.sleep(wait)
                    else:
                        break

            logger.error(f"[Spark] Judge failed after {_JUDGE_RETRIES} attempts for {umo}: {last_err}, defaulting to allow")
            return True

        except Exception as e:
            logger.error(f"[Spark] Judge unexpected error({umo}): {e}, defaulting to allow")
            return True

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
        """Resolve the LLM provider for the generate step."""
        pid = self._get_cfg("proactive_settings", "fixed_provider", "") or ""
        if pid:
            p = self.context.get_provider_by_id(pid)
            if p:
                return p
        return self.context.get_using_provider(umo=umo)

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
        mode = str(self._get_cfg("proactive_settings", "proactive_fact_envelope_mode", "minimal") or "minimal").lower()
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
        time_period_prompt = _get_prompt() if callable(_get_prompt) else getattr(self.context, "_time_period_current_prompt", "")
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
        facts.append("事实优先级：当前时间、当前活动、当前节律、最近真实对话优先于长期记忆和知识库。长期记忆/知识库只能作为背景补充，不能当作今天刚发生的事；如果与当前事实冲突，必须忽略旧内容。")
        facts.append("[/主动回复实时事实]")
        return "\n".join(facts) + "\n\n" + prompt

    def _proactive_placeholder(self) -> str:
        return (
            self._get_cfg("proactive_settings", "proactive_user_placeholder")
            or "[用户本人未发送消息，本轮为 AI 主动对 Mando 发起对话]"
        )

    def _is_proactive_placeholder(self, content: str) -> bool:
        normalized = self._normalize_history_text(content)
        known_placeholders = {
            self._normalize_history_text(self._proactive_placeholder()),
            "[用户本人未发送消息，本轮为 AI 主动对 Mando 发起对话]",
            "[用户本人未说话，本轮为 AI 主动发起对话]",
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

    def _project_complete_history_rounds(self, contexts: list) -> list[dict]:
        """Project raw history into complete user/assistant rounds."""
        candidate_rounds = []
        current_round = None
        for msg in contexts:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            content = self._extract_history_text(msg.get("content", ""))
            if role not in ("user", "assistant") or not content:
                continue
            if role == "user":
                current_round = None
                content = self._sanitize_retrieval_user_content(content)
                if not content or not self._is_natural_retrieval_line(content):
                    continue
                current_round = {
                    "proactive": self._is_proactive_placeholder(content),
                    "user": content,
                    "assistant": [],
                }
                candidate_rounds.append(current_round)
            elif self._is_internal_history_noise(role, content):
                continue
            elif current_round is not None and self._is_natural_retrieval_line(content):
                current_round["assistant"].append(content)

        return [turn for turn in candidate_rounds if turn["assistant"]]

    def _select_recent_round_contexts(
        self, contexts: list, rounds: int
    ) -> tuple[list, list[dict]]:
        """Return the newest complete rounds as protocol role messages."""
        if rounds <= 0:
            return [], []
        complete_rounds = self._project_complete_history_rounds(contexts)
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
                proactive_marker
                if turn["proactive"]
                else f"用户：{turn['user']}"
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
                    proactive_marker
                    if turn["proactive"]
                    else f"用户：{turn['user']}"
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
            "proactive" if recent_rounds and recent_rounds[-1]["proactive"] else "normal"
        ) if recent_rounds else "none"
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
        tz: Optional[str],
        prompt_template: str,
        skip_judge: bool = False,
        judge_current_round: Optional[dict] = None,
    ) -> bool:
        """
        执行主动回复的核心方法
        
        v3 改造：通过官方 CronMessageEvent + build_main_agent 走合规 Agent Pipeline，
        支持完整的工具调用、人格注入、历史管理。
        当框架 API 不可用时降级到旧的 provider.text_chat 方式。
        
        Args:
            skip_judge: 为 True 时跳过 LLM 判断步骤，必定触发回复（用于每日问候等定时任务）
        """
        try:
            # Step 1: Judge whether to reply (skip for daily greetings etc.)
            if not skip_judge and self._get_cfg("proactive_settings", "proactive_judge_enable", True):
                if not await self._judge_should_reply(
                    umo,
                    tz,
                    current_round=judge_current_round,
                ):
                    return False

            # Step 2: Format prompt
            now = _now_tz(tz)
            time_fmt = self._get_cfg("basic_settings", "time_format") or "%Y-%m-%d %H:%M"
            now_str = now.strftime(time_fmt)

            st = self._states.get(umo)
            time_since_last_chat = "未知"
            if st:
                _last_chat_ts = max(st.last_user_reply_ts, st.last_proactive_reply_ts, st.last_ai_reply_ts)
                if _last_chat_ts > 0:
                    time_since_last_chat = _format_time_delta(now.timestamp() - _last_chat_ts)

            last_user, last_ai = await self._get_last_messages(umo)

            if prompt_template:
                today_schedule = getattr(self.context, "_busy_schedule_today_schedule", "")
                outfit = getattr(self.context, "_busy_schedule_outfit", "")
                current_activity = getattr(self.context, "_busy_schedule_current_activity", "")
                next_activity = getattr(self.context, "_busy_schedule_next_activity", "")
                custom_prompt = getattr(self.context, "_busy_schedule_custom_prompt", "")
                _get_prompt = getattr(self.context, "_time_period_get_prompt", None)
                time_period_prompt = _get_prompt() if callable(_get_prompt) else getattr(self.context, "_time_period_current_prompt", "")

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
                response_text = await self._run_agent_pipeline(
                    umo, prompt, tz, provider=gen_provider, persona=gen_persona
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

            if not response_text:
                return False

            # Send message (if Agent didn't send via tool)
            if not getattr(self, '_last_cron_event_sent', False):
                await self._send_text(umo, response_text)
            logger.info(f"[Spark] 已发送主动回复给 {umo}: {response_text[:50]}...")

            # Save history only for legacy path; agent pipeline saves history in _run_agent_pipeline.
            if not HAS_AGENT_PIPELINE:
                try:
                    conv_mgr = self.context.conversation_manager
                    curr_cid = await conv_mgr.get_curr_conversation_id(umo)
                    if curr_cid:
                        placeholder = self._get_cfg("proactive_settings", "proactive_user_placeholder") or "[用户本人未发送消息，本轮为 AI 主动对 Mando 发起对话]"
                        await self._add_message_pair_to_history(
                            umo,
                            curr_cid,
                            None,
                            placeholder,
                            response_text,
                        )
                except Exception as e:
                    logger.warning(f"[Spark] 保存主动回复历史失败: {e}")

            # Update state
            now_ts = now.timestamp()
            if umo not in self._states:
                self._states[umo] = SessionState()
            st = self._states[umo]
            st.last_ts = now_ts
            st.last_proactive_reply_ts = now_ts
            await self._debounced_save_session_data()

            return True

        except Exception as e:
            logger.error(f"[Spark] proactive error({umo}): {e}", exc_info=True)
            return False

    def _extract_history_text(self, content) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, str):
                    text = part
                elif isinstance(part, dict):
                    text = part.get("text") or part.get("content") or part.get("value") or ""
                else:
                    text = getattr(part, "text", "") or getattr(part, "content", "") or ""
                if text:
                    parts.append(str(text).strip())
            return " ".join(p for p in parts if p).strip()
        if isinstance(content, dict):
            text = content.get("text") or content.get("content") or content.get("value") or ""
            return str(text).strip() if text else ""
        text = getattr(content, "text", "") or getattr(content, "content", "") or ""
        return str(text).strip() if text else ""

    def _dedupe_contexts(self, contexts: list) -> list:
        # Iterate in reverse so the latest occurrence of duplicate content is kept,
        # not the oldest. This matters when proactive placeholders repeat identically.
        seen = set()
        result = []
        for msg in reversed(contexts):
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            content = self._extract_history_text(msg.get("content", ""))
            if role not in ("user", "assistant") or not content:
                continue
            if self._is_internal_history_noise(role, content):
                continue
            key = (role, re.sub(r"\s+", " ", content).strip())
            if key in seen:
                continue
            seen.add(key)
            result.append({"role": role, "content": content})
        result.reverse()
        return result

    def _format_context_tail_for_log(self, contexts: list, limit: int = 4) -> str:
        lines = []
        for msg in contexts[-limit:]:
            role = msg.get("role", "") if isinstance(msg, dict) else ""
            content = self._extract_history_text(msg.get("content", "")) if isinstance(msg, dict) else ""
            if not content:
                continue
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
        if "最近聊天：" in stripped and ("[用户本人未说话" in stripped or "[用户本人未发送消息" in stripped):
            return True
        if norm.startswith("最近聊天：") and "AI:" in norm and "Mando:" in norm:
            return True
        return False

    def _parse_conversation_history(self, conversation) -> list:
        if not conversation or not getattr(conversation, "history", None):
            return []
        try:
            parsed = json.loads(conversation.history) if isinstance(conversation.history, str) else conversation.history
            return parsed if isinstance(parsed, list) else []
        except Exception as e:
            logger.warning(f"[Spark] 解析对话历史失败: {e}")
            return []

    async def _remove_internal_history_tail(self, umo: str, conversation_id: str, before_len: int | None = None, assistant_response: str = "") -> int:
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
            if in_recent_tail and response_key and role == "assistant" and self._normalize_history_text(content) == response_key:
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
    ) -> None:
        if not conversation_id or not assistant_response:
            return
        placeholder = self._proactive_placeholder()
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
            {"role": "user", "content": placeholder},
            {"role": "assistant", "content": assistant_response},
        ]
        if history[-2:] != expected_tail:
            history.extend(expected_tail)
            await conv_mgr.update_conversation(umo, conversation_id, history=history)
        logger.debug(
            f"[Spark] 已写入标准主动历史: {conversation_id}, "
            f"baseline={baseline_len}, cleaned={removed}"
        )

    async def _run_agent_pipeline(self, umo: str, prompt: str, tz: Optional[str] = None,
                                   provider=None, persona: str = "") -> Optional[str]:
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

        if not provider:
            provider = self._get_gen_provider(umo)

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
        retrieval_query = self._build_proactive_retrieval_query(retrieval_contexts, prompt)
        logger.info(f"[Spark] Generation recent contexts for {umo}: {self._format_context_tail_for_log(retrieval_contexts)}")
        logger.info(f"[Spark] Generation retrieval query for {umo}: {retrieval_query}")
        generation_prompt = prompt
        req.prompt = retrieval_query
        cron_event.set_extra("spark_proactive_retrieval", True)

        result = await build_main_agent(
            event=cron_event,
            plugin_context=self.context,
            config=config,
            provider=provider,
            req=req,
            apply_reset=False,
        )

        if not result or not result.agent_runner:
            logger.warning(f"[Spark] build_main_agent 返回空结果: {umo}")
            return None

        runner = result.agent_runner

        cron_event.message_str = generation_prompt
        # hook 期间 req.prompt 保持 retrieval_query（真实聊天+模板指令），供 livingmemory/knowledge_base 做检索
        # 与正常 pipeline 顺序一致：OnLLMRequestEvent 触发后各插件注入记忆/知识库/节律等
        if await call_event_hook(cron_event, EventType.OnLLMRequestEvent, req):
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
                    conversation = await self.context.conversation_manager.get_conversation(
                        umo, curr_cid
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
            f"placeholder={any(self._is_proactive_placeholder(self._extract_history_text(msg.get('content', ''))) for msg in req.contexts if isinstance(msg, dict))}"
        )
        if result.reset_coro:
            await result.reset_coro

        async for _ in runner.step_until_done(30):
            pass

        llm_resp = runner.get_final_llm_resp()
        if not llm_resp or not llm_resp.completion_text:
            logger.debug(f"[Spark] Agent 无文本响应: {umo}")
            return None

        response_text = llm_resp.completion_text.strip()
        if not response_text:
            return None

        # Only suppress Spark's manual text send if the agent/tool already sent the
        # full response and returned no text (e.g. a tool that sends its own text reply).
        # Do NOT suppress when a tool only sent an image while the agent still returned text.
        self._last_cron_event_sent = (
            getattr(cron_event, '_has_send_oper', False) and not response_text
        )

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
                    response_text,
                    hook_history_len or 0,
                )
            else:
                logger.warning(f"[Spark] 保存主动回复历史失败: 未找到当前会话ID {umo}")
        except Exception as e:
            logger.warning(f"[Spark] 保存对话历史失败: {e}")

        return response_text

    async def _run_legacy_llm(self, umo: str, prompt: str,
                              provider=None, persona: str = "") -> Optional[str]:
        """Fallback: direct provider.text_chat() for older framework versions."""
        if not provider:
            provider = self._get_gen_provider(umo)
        if not provider:
            logger.warning(f"[Spark] provider missing for {umo}")
            return None

        if not persona:
            persona = await self._get_gen_persona(umo)

        self._last_cron_event_sent = False

        contexts = await self._get_conversation_contexts(umo, 10)
        logger.info(f"[Spark] Legacy generation recent contexts for {umo}: {self._format_context_tail_for_log(contexts)}")

        llm_resp = await provider.text_chat(
            prompt=None,
            contexts=[{"role": "user", "content": prompt}] + contexts,
            system_prompt=persona,
        )
        text = llm_resp.completion_text if hasattr(llm_resp, "completion_text") else ""
        return text.strip() if text else None
    
    async def _proactive_reminder_reply(self, umo: str, reminder_content: str) -> bool:
        """
        执行由 AI 生成的主动提醒回复
        
        v3 改造：复用 _run_agent_pipeline / _run_legacy_llm，走合规调用。
        """
        try:
            tz = self._get_cfg("basic_settings", "timezone") or None
            now = _now_tz(tz)
            time_fmt = self._get_cfg("basic_settings", "time_format") or "%Y-%m-%d %H:%M"
            now_str = now.strftime(time_fmt)

            st = self._states.get(umo)
            time_since_last_chat = "未知"
            if st:
                _last_chat_ts = max(st.last_user_reply_ts, st.last_proactive_reply_ts, st.last_ai_reply_ts)
                if _last_chat_ts > 0:
                    time_since_last_chat = _format_time_delta(now.timestamp() - _last_chat_ts)

            last_user, last_ai = await self._get_last_messages(umo)

            # 使用提醒 prompt 模板
            template = self._get_cfg("reminders_settings", "reminder_prompt_template") or "用户提醒：{reminder_content}"
            try:
                prompt = template.format(
                    reminder_content=reminder_content,
                    now=now_str,
                    umo=umo,
                    time_since_last_chat=time_since_last_chat,
                    last_user=last_user,
                    last_ai=last_ai
                )
            except KeyError as e:
                logger.warning(f"[Spark] 提醒模板格式化失败，未知占位符: {e}，使用默认模板")
                prompt = f"用户提醒：{reminder_content}"

            logger.info(f"[Spark] 触发 AI 提醒 for {umo}: {reminder_content}")

            gen_provider = self._get_gen_provider(umo)
            gen_persona = await self._get_gen_persona(umo)

            if HAS_AGENT_PIPELINE:
                response_text = await self._run_agent_pipeline(
                    umo, prompt, tz, provider=gen_provider, persona=gen_persona
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

            if not response_text:
                return False

            if not getattr(self, '_last_cron_event_sent', False):
                await self._send_text(umo, response_text)
            logger.info(f"[Spark] 已发送 AI 提醒给 {umo}: {response_text[:50]}...")

            # Save history only for legacy path; agent pipeline saves history in _run_agent_pipeline.
            if not HAS_AGENT_PIPELINE:
                try:
                    conv_mgr = self.context.conversation_manager
                    curr_cid = await conv_mgr.get_curr_conversation_id(umo)
                    if curr_cid:
                        placeholder = self._get_cfg("proactive_settings", "proactive_user_placeholder") or "[用户本人未发送消息，本轮为 AI 主动对 Mando 发起对话]"
                        await self._add_message_pair_to_history(
                            umo,
                            curr_cid,
                            None,
                            placeholder,
                            response_text,
                        )
                except Exception as e:
                    logger.warning(f"[Spark] 保存提醒历史失败: {e}")

            # 更新状态
            if umo not in self._states:
                self._states[umo] = SessionState()
            st = self._states[umo]
            st.last_proactive_reply_ts = _now_tz(tz).timestamp()
            await self._debounced_save_session_data()

            return True

        except Exception as e:
            logger.error(f"[Spark] proactive reminder error({umo}): {e}", exc_info=True)
            return False

    async def _add_message_pair_to_history(self, umo: str, conversation_id: str, conversation, user_prompt: str, assistant_response: str):
        """
        将消息对添加到对话历史（使用官方 API）
        
        注意：走 build_main_agent 的主动回复会在 _run_agent_pipeline 中保存历史，
        此方法仅用于降级路径或其他需要手动追加历史的场景。
        """
        try:
            if not conversation_id:
                logger.warning("[Spark] conversation_id 为空，无法更新历史")
                return

            conv_mgr = self.context.conversation_manager

            if HAS_NEW_MESSAGE_API:
                try:
                    user_msg = UserMessageSegment(content=[TextPart(text=user_prompt)])
                    assistant_msg = AssistantMessageSegment(content=[TextPart(text=assistant_response)])
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

    async def _get_last_messages(self, umo: str) -> Tuple[str, str]:
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

            history = json.loads(conversation.history) if isinstance(conversation.history, str) else conversation.history
            if not isinstance(history, list):
                return last_user, last_ai

            for msg in reversed(history):
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role", "")
                content = self._extract_history_text(msg.get("content", ""))[:200]
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
    ) -> list:
        """Fetch recent conversation history and Spark proactive projection as context dicts."""
        if rounds <= 0:
            return []

        msgs = []
        try:
            conv_mgr = self.context.conversation_manager
            curr_cid = await conv_mgr.get_curr_conversation_id(umo)
            if curr_cid:
                conversation = await conv_mgr.get_conversation(umo, curr_cid)
                if conversation and conversation.history:
                    history = json.loads(conversation.history) if isinstance(conversation.history, str) else conversation.history
                    if isinstance(history, list):
                        source_history = (
                            history
                            if preserve_round_boundaries
                            else history[-rounds * 4:]
                        )
                        for msg in source_history:
                            if not isinstance(msg, dict):
                                continue
                            role = msg.get("role", "")
                            if role not in ("user", "assistant"):
                                continue
                            content = self._extract_history_text(msg.get("content", ""))
                            if content:
                                msgs.append({"role": role, "content": content})
        except Exception as e:
            logger.warning(f"[Spark] Failed to get conversation contexts for {umo}: {e}")

        if preserve_round_boundaries:
            return msgs
        msgs = self._dedupe_contexts(msgs)
        return msgs[-rounds * 2:]

    def _apply_segmentation(self, text: str) -> list[str]:
        """应用分段回复逻辑（模拟 AstrBot 的分段正则处理）
        
        Returns:
            分段后的文本列表
        """
        try:
            # 获取分段配置
            seg_config = self.context.get_config().get("platform_settings", {}).get("segmented_reply", {})
            
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

    async def _send_text(self, umo: str, text: str):
        """发送主动回复消息到指定会话"""
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
            
        except Exception as e:
            logger.error(f"[Spark] ❌ 发送消息失败({umo}): {e}")
    
    async def _send_reminder_message(self, umo: str, text: str):
        """发送提醒消息到指定会话"""
        await self._send_text(umo, text)

    # 生命周期管理
    async def terminate(self):
        """插件销毁"""
        self._stopped = True  # 设置停止标志，让调度器循环退出
        
        if self._loop_task and not self._loop_task.done():
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass  # 预期的取消异常

        # 取消所有对话增强任务
        for umo, task in list(self._enhancement_tasks.items()):
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