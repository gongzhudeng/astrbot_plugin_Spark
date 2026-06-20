# 灵犀 · 主动对话 更新日志

## v1.3.0

### 优化

- **主动对话缓存命中率提升**：主动对话现在通过 `OnLLMRequestEvent` 钩子触发 busy_schedule 插件的注入，使主动对话的 system_prompt 与正常对话保持一致，共享 KV Cache，提高缓存命中率，降低 API 调用成本
- **清理提示词模板**：移除提示词模板中的 busy_schedule 插件占位符（`{today_schedule}`, `{outfit}`, `{current_activity}`, `{next_activity}`, `{custom_prompt}`），改为由 busy_schedule 插件通过钩子自动注入
- **人设注入方式统一**：主动对话现在通过 `_ensure_persona_and_skills` 方法注入人设，与正常对话保持一致，进一步提升缓存命中率

### 新增

- **灵犀AI忙碌时段管理插件集成说明**：在 README 中添加了对 astrbot_plugin_busy_schedule 插件的集成说明，安装即生效，无需配置

## v1.2.6

### 修复

- 修复沉寂问候在智能判断返回"否"后不标记已触发标记，导致调度器每 30 秒重复调用 LLM 判断直到碰巧返回"是"的问题。现在判断结果为"否"时也会标记该分钟已触发，避免无意义的重复调用。

## v1.2.1

### 修复

- 智能判断 LLM 调用失败时自动重试最多 3 次（指数退避），仅对临时性错误重试（502/503/504/超时/空响应）
- 判断失败后不再静默跳过，日志会记录重试过程和最终失败原因

## v1.0.0

基于 [astrbot_plugin_Conversa v3.0.0](https://github.com/Luna-channel/astrbot_plugin_Conversa) 二次开发，灵感参考：Luna-channel。

### 新增

- **两步式主动对话流程**：第一步 LLM 智能判断是否适合开口，第二步生成回复内容；两步可独立配置供应商和人格
- **无限每日定时问候**：支持添加任意数量的问候时段，每个时段独立控制 `ignore_dnd`（是否跳过免打扰和忙碌）
- **忙碌时段免打扰**：与 `astrbot_plugin_busy_schedule` 深度集成，忙碌期间自动静默
- **中文指令**：`/灵犀`、`/立即主动`、`/主动状态`、`/主动帮助` 及子命令中文别名
- **人格下拉选择**：判断用人格和生成用人格支持 `_special: select_persona` 下拉选择

### 改造

- 从 Conversa v3.0.0 fork，插件名重命名为 `astrbot_plugin_Spark`
- 旧的句号拦截逻辑替换为 LLM 判断步骤
- 固定 3 时段每日问候迁移为列表格式，支持无限添加
- 旧版配置自动迁移（`daily1/2/3` → 列表、`special.provider` → `advanced.fixed_provider` 等）
- 所有日志前缀 `[Conversa]` → `[Spark]`
- 命令系统保留 `/conversa` 兼容，新增 `/灵犀` 中文主命令
- 主动回复走 AstrBot 官方 Agent Pipeline（继承自 Conversa v3.0.0）

### 保留的上游功能

- 沉寂问候（基准时间 + 随机波动）
- 私聊对话增强（概率追回复）
- Agent 订阅模式（manual / auto / agent）
- 免打扰时段（全局 + 用户专属）
- 自动退订 & 重新激活
- 提醒事项迁移到 AstrBot 原生 cron 系统