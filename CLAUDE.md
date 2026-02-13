# Context Forge — 项目规范

## 项目概览

**名称**：Context Forge — 高性能动态上下文组装引擎
**定位**：LLM 应用的 ORM，把上下文组装从字符串拼接提升为声明式工程层
**双重身份**：书籍《LLM 工程化项目实战指南·2026 版》第 6 章配套项目 + 可 fork 直接用于生产的开源基础设施
**许可证**：Apache 2.0
**项目路径**：`D:\MyProjects\test\context-forge\`
**状态**：v0.1.0 — 功能代码全部完成，测试套件完整（903/903 通过，覆盖率 84.26%），P1 缺陷已全部修复

## 已确认的关键决策

### 1. API 风格：异步优先 + 同步包装（方案 B）
- 主 API：`await forge.build(...)`
- 同步便捷方法：`forge.build_sync(...)`（内部 `asyncio.run()`）
- Jupyter 场景自动检测并提示使用 `await` 或 `nest_asyncio`

### 2. LLM 调用分层
- **默认路径**：纯规则 + 截断，不调 LLM（零外部依赖）
- **高级路径**：可选接入 LLM Provider（摘要压缩、模型路由等）
- 每个需要 LLM 的模块都提供不需要 LLM 的默认实现：
  - 压缩：`TruncationCompressor`（默认） / `LLMSummaryCompressor`（可选）
  - Injection 检测：`HeuristicDetector`（默认） / `ClassifierDetector`（可选）
  - 路由：`RuleBasedRouter`（默认） / `LLMRouter`（可选）

### 3. 依赖策略：全部直接依赖
- 核心依赖：pydantic v2, pyyaml, tiktoken, typer, rich, fastapi, uvicorn, opentelemetry, redis, sentence-transformers, protobuf, anyio
- 框架集成（langchain/llamaindex/haystack）：optional-dependencies，内部用 lazy import

### 4. Tokenizer 策略
- 默认：tiktoken 近似计数（对所有模型都可用，误差 < 5%）
- 可插拔：用户可注入精确 Tokenizer（`TokenCounter` Protocol）
- Fallback：字符数 / 4 粗估（零依赖），中文自动检测用 chars/2
- 模型名到编码方案的前缀匹配映射在 `tokenizer/registry.py`

### 5. MVP 范围
- 完整实现：11 个功能模块 + CLI（6 子命令） + 6 大场景示例 + 7 个模块 Demo
- 仅标注不实现（`🏭 生产提示`）：
  - Context Distillation → LoRA/Soft Prompts
  - RadixAttention 深度对接（vLLM/SGLang）
  - 分布式锁/一致性（多 Agent 并发写入）

### 6. 工具链
- **包管理**：uv（pyproject.toml 已配置，当前用 pip 安装也可以）
- **CLI 框架**：Typer + Rich
- **Lint/Format**：ruff（target-version = "py310"）
- **类型检查**：mypy（strict）
- **测试**：pytest + pytest-asyncio（asyncio_mode = "auto"），覆盖率 > 85%
- **文档**：mkdocs + mkdocstrings
- **构建后端**：hatchling

### 7. 示例 Mock 模式
- 所有示例支持 `--mock` 模式（内置 mock LLM 响应）
- 无 API Key 自动降级到 mock，输出提示信息

## 技术规格

### Python 版本
- 最低：Python 3.10（pyproject.toml `requires-python = ">=3.10"`）
- 开发环境实际使用 Python 3.12.7
- 可使用：`X | Y` 联合类型、`list[X]` 泛型、`match/case`
- 不可使用：`StrEnum`（3.11）、`tomllib`（3.11）、`TaskGroup`（3.11）

### 语言规范
- **全部中文**：注释、docstring、错误信息、CLI 帮助文本、文档
- 代码标识符保持英文（Python 惯例）

### 代码风格
- 所有公共 API 必须有类型标注和 docstring（Google 风格）
- Pydantic 数据模型使用 `frozen=True`（不可变）
- 更新操作返回新对象（`with_xxx()` 模式），不原地修改
- 完整代码，不用 `...` 或 `# 其余部分类似` 省略
- 每个文件开头有模块级 docstring，说明用途和章节映射

### 教学标注（在代码注释中使用）
- `# [Design Decision]`：架构/技术设计决策及理由
- `# [DX Decision]`：开发者体验设计决策及理由
- `# → 6.x.x.x 标题`：章节映射，方便读者交叉引用
- `# 🏭 生产提示：...`：生产环境需要补充的细节
- `# ⚠️ 反模式：...`：反模式警告（映射 6.7）

### 错误处理
- 结构化异常体系：`ContextForgeError` 基类 + 子类（全在 `errors/exceptions.py`）
- 三段式错误信息：What（发生了什么）+ Why（为什么）+ How（怎么修）
- Warning 而非静默丢弃：自动降级时必须发 warning
- 异常子类：BudgetExceededError, SanitizationError, InjectionDetectedError, PipelineError, PipelineStageError, ConfigValidationError, PolicyLoadError, ModelNotFoundError, TokenizerError, CompressionError, CacheError, RoutingError, PluginError, SerializationError, AntiPatternWarning

## 功能模块与章节映射

| 功能模块 | 映射章节 | 核心代码位置 | 状态 |
|---------|---------|-------------|------|
| Context Package 数据模型 | 6.1.1 | `models/` | ✅ 完成 |
| Context Builder Pipeline | 6.1.2 | `pipeline/` | ✅ 完成（含 compress_stage） |
| 决策审计与可解释性 | 6.1.3 | `models/audit.py` | ✅ 完成 |
| Token Budget Manager | 6.2.2 | `budget/` + `pipeline/allocate.py` | ✅ 完成（独立模块 + 竞价机制） |
| 缓存与复用层 | 6.2.3 | `cache/` | ✅ 完成（Memory + Redis 后端） |
| 压缩策略引擎 | 6.2.4 + 6.3.3 | `compress/` | ✅ 完成（截断/去重/摘要/引擎） |
| Write/Select/Compress/Isolate 策略 | 6.3 | 分散在 pipeline/ 和专属模块 | ✅ 完成 |
| 零信任清洗管道 | 6.4 | `sanitize/` + `pipeline/sanitize_stage.py` | ✅ 完成（6 个清洗插件） |
| 可观测性套件 | 6.5 | `observability/` | ✅ 完成（Snapshot/Diff/Golden Set/Metrics/Tracing） |
| 意图路由与动态调度 | 6.6 | `routing/` | ✅ 完成（规则路由/LLM 路由/Context Bus） |
| 反模式检测器 | 6.7 | `antipattern/` | ✅ 完成（规则引擎 + 检测器） |

## 18 个交付物完成状态

| # | 交付物 | 轮次 | 状态 |
|---|--------|------|------|
| 1 | 架构概览（Mermaid + 数据流描述） | 第一轮 | ✅ |
| 2 | 完整目录树 | 第一轮 | ✅ |
| 3 | 核心数据模型（models/） | 第一轮 | ✅ |
| 4 | High-Level Facade API（facade.py） | 第一轮 | ✅ |
| 5 | Pipeline 完整实现 + 策略 YAML | 第二轮 | ✅ |
| 6 | Budget Manager 独立模块 | 第二轮 | ✅ |
| 7 | 清洗与安全层（sanitize/） | 第二轮 | ✅ |
| 8 | 压缩与缓存（compress/ + cache/） | 第三轮 | ✅ |
| 9 | 路由与调度（routing/ + context_bus） | 第三轮 | ✅ |
| 10 | 可观测性（observability/） | 第三轮 | ✅ |
| 11 | 反模式检测器 | 第四轮 | ✅ |
| 12 | CLI 工具完整实现 | 第四轮 | ✅ |
| 13 | 错误处理体系 | 第一轮 | ✅ |
| 14 | 六大场景集成示例 | 第五轮 | ✅ |
| 15 | 测试套件 | 第五轮 | ✅ 452/452 通过，覆盖率 74%（见审计报告） |
| 16 | 配置参考（6 个策略 YAML） | 第五轮 | ✅ |
| 17 | DevOps 文件 | 第六轮 | ✅ |
| 18 | README.md 完整版 | 第六轮 | ✅ |

## 完整文件清单

### 核心引擎 — `src/context_forge/`

```
__init__.py                          # 包导出，暴露 ContextForge + 所有核心模型
facade.py                            # ContextForge 顶层 Facade，build() / build_sync()
facade_observability.py              # ObservabilityMixin — diff/snapshot/golden_record 方法
py.typed                             # PEP 561 类型标记

models/
  __init__.py                        # 模型导出
  segment.py                         # Segment, SegmentType, Priority, DEFAULT_PRIORITY_MAP
  provenance.py                      # Provenance, SourceType
  control.py                         # ControlFlags, Visibility
  metadata.py                        # SegmentMetadata
  context_package.py                 # ContextPackage, TokenUsage
  budget.py                          # BudgetPolicy, BudgetAllocation, SpendType
  routing.py                         # ModelConfig, RoutingRule, RoutingDecision, ComplexityLevel
  audit.py                           # AuditEntry, DecisionType, ReasonCode

pipeline/
  __init__.py                        # Pipeline 导出
  base.py                            # PipelineStage Protocol, Pipeline 编排器, PipelineContext
  normalize.py                       # NormalizeStage — Unicode 归一化 + Token 计数填充
  sanitize_stage.py                  # SanitizeStage — 集成 sanitize/ 模块插件链
  rerank.py                          # RerankStage — 优先级排序 + TTL + 去重 + MMR 多样性
  allocate.py                        # AllocateStage — 刚性/弹性预算分配
  compress_stage.py                  # CompressStage — 集成 compress/ 引擎
  assemble.py                        # AssembleStage — 最终组装 + 顺序整理

budget/
  __init__.py                        # Budget 导出
  manager.py                         # BudgetManager — 完整预算分配逻辑
  strategies.py                      # 刚性/弹性/预留三种策略实现
  bidding.py                         # 弹性区间竞价算法（→ 6.2.2.2）

sanitize/
  __init__.py                        # Sanitize 导出
  base.py                            # Sanitizer Protocol
  unicode_normalizer.py              # Unicode 归一化插件
  html_stripper.py                   # HTML/Markdown 剥离插件
  pii_redactor.py                    # PII 脱敏插件（手机号/邮箱/身份证）
  injection_detector.py              # Injection 检测插件（启发式 + 模式匹配）
  length_guard.py                    # 长度攻击防御插件

compress/
  __init__.py                        # Compress 导出
  base.py                            # Compressor Protocol
  truncation.py                      # TruncationCompressor（默认，无 LLM）
  dedup.py                           # DedupCompressor — 去重压缩
  summary.py                         # LLMSummaryCompressor（可选，需 LLM）
  engine.py                          # CompressionEngine — 策略编排

cache/
  __init__.py                        # Cache 导出
  base.py                            # CacheBackend Protocol
  keys.py                            # 缓存键生成策略
  manager.py                         # CacheManager — 缓存编排
  memory.py                          # MemoryCacheBackend（内存后端）
  redis_backend.py                   # RedisCacheBackend（Redis 后端）

routing/
  __init__.py                        # Routing 导出
  base.py                            # Router Protocol
  complexity.py                      # ComplexityAnalyzer — 复杂度分析
  rule_based.py                      # RuleBasedRouter（默认，无 LLM）
  llm_router.py                      # LLMRouter（可选，需 LLM）
  context_bus.py                     # ContextBus — 多 Agent 上下文协调

observability/
  __init__.py                        # Observability 导出
  snapshot.py                        # Context Snapshot — 上下文快照
  diff.py                            # Prompt Diff — 版本差异比较
  golden_set.py                      # Golden Set — 回归测试基准
  metrics.py                         # 核心指标收集与导出
  tracing.py                         # OpenTelemetry 追踪集成

antipattern/
  __init__.py                        # Antipattern 导出
  base.py                            # AntiPatternRule Protocol
  detector.py                        # AntiPatternDetector — 检测引擎
  rules.py                           # 内置反模式规则集

cli/
  __init__.py                        # CLI 导出
  app.py                             # Typer 主应用（注册所有子命令）
  cmd_init.py                        # init 子命令 — 初始化项目配置
  cmd_build.py                       # build 子命令 — 构建上下文
  cmd_inspect.py                     # inspect 子命令 — 检查上下文包
  cmd_diff.py                        # diff 子命令 — 对比两个上下文快照
  cmd_validate.py                    # validate 子命令 — 校验策略 YAML
  cmd_serve.py                       # serve 子命令 — 启动 HTTP 服务
  server.py                          # FastAPI 服务实现
  utils.py                           # CLI 工具函数

errors/
  __init__.py                        # 异常导出
  exceptions.py                      # 全部 16 种异常类

tokenizer/
  __init__.py                        # Tokenizer 导出
  protocol.py                        # TokenCounter Protocol
  tiktoken_counter.py                # TiktokenCounter（精确计数）
  fallback.py                        # CharBasedCounter（粗估 fallback）
  registry.py                        # get_tokenizer() 自动选择 + 模型前缀映射

config/
  __init__.py                        # 配置导出
  defaults.py                        # MODEL_REGISTRY（15+ 模型）, MODEL_ALIASES, resolve_model()
  loader.py                          # load_policy(), YAML 加载/校验/合并
  schema.py                          # PolicyConfig 及子配置 Pydantic 模型

plugins/
  __init__.py                        # 插件注册表（预留扩展）

integrations/
  __init__.py                        # 框架适配器（预留扩展）
```

### 策略配置 — `configs/`

```
default_policy.yaml                  # 默认策略文件，含详细中文注释
rag_policy.yaml                      # RAG 场景策略
conversation_policy.yaml             # 多轮对话场景策略
security_policy.yaml                 # 安全合规场景策略
multi_agent_policy.yaml              # 多 Agent 协调场景策略
cost_optimization_policy.yaml        # 成本优化场景策略
```

### 示例 — `examples/`

```
_shared.py                           # 示例共享工具（mock LLM、输出格式化等）
README.md                            # 示例说明文档
quickstart.py                        # 快速上手（3 个基础场景）
budget_manager_demo.py               # Budget Manager 模块演示
sanitize_demo.py                     # 清洗模块演示
compress_demo.py                     # 压缩模块演示
cache_demo.py                        # 缓存模块演示
routing_demo.py                      # 路由模块演示
observability_demo.py                # 可观测性模块演示
scenario_rag_quality.py              # 场景 1：RAG 上下文质量治理
scenario_conversation_memory.py      # 场景 2：多轮对话记忆管理
scenario_multi_agent.py              # 场景 3：多 Agent 上下文协调
scenario_security_compliance.py      # 场景 4：安全合规清洗
scenario_versioning.py               # 场景 5：Prompt 版本管理与回归
scenario_routing_cost.py             # 场景 6：多模型适配与成本优化
```

### 测试 — `tests/`

```
conftest.py                          # 共享 fixtures
unit/
  test_smoke.py                      # 冒烟测试（基础功能验证）
  test_models.py                     # 数据模型测试
  test_pipeline.py                   # Pipeline 各阶段测试
  test_budget.py                     # Budget Manager 测试
  test_sanitize_module.py            # 清洗模块测试
  test_compress.py                   # 压缩模块测试
  test_cache.py                      # 缓存模块测试
  test_routing.py                    # 路由模块测试
  test_observability.py              # 可观测性模块测试
  test_antipattern.py                # 反模式检测器测试
  test_cli.py                        # CLI 子命令测试
  test_config.py                     # 配置加载测试
  test_tokenizer.py                  # Tokenizer 测试
  test_errors.py                     # 异常体系测试
integration/
  test_full_pipeline.py              # 全流水线集成测试
  test_compress_integration.py       # 压缩集成测试
  test_round3_complete.py            # 第三轮模块集成测试
  test_serve.py                      # HTTP 服务集成测试
```

### DevOps 与文档

```
.gitignore                           # Python + 项目特定忽略规则
.gitattributes                       # 行尾符规范化（LF）+ 二进制文件标记
.env.example                         # 环境变量模板（含中文注释）
.dockerignore                        # Docker 构建排除文件
Dockerfile                           # 多阶段构建（builder + runtime），CPU-only PyTorch
docker-compose.yml                   # Context Forge + Redis 服务编排
.github/workflows/ci.yml            # CI 流水线（lint/typecheck/test/coverage/PyPI publish）
.github/workflows/release.yml       # 发布流水线（version validation/Docker push/GitHub release）
Makefile                             # 常用命令快捷方式
scripts/setup_dev.sh                 # Linux/Mac 开发环境配置脚本
scripts/setup_dev.ps1                # Windows 开发环境配置脚本
pyproject.toml                       # 项目配置（hatchling 构建）
LICENSE                              # Apache 2.0
README.md                            # 生产级 README（900+ 行，15 章节，3 Mermaid 图）
CONTRIBUTING.md                      # 贡献指南（中文，完整开发流程）
CHANGELOG.md                         # 版本历史（Keep a Changelog 格式）
```

### 未创建的文件

- `docs/architecture.md` — 架构文档（因 pre-commit hook 限制未创建，可后续补充）
- `docs/api_reference.md` — API 参考文档（同上）
- `benchmarks/` — 性能基准目录（预留，未实现）

## 关键架构细节

### Facade 内部流程
1. `ContextForge.__init__()` 解析模型 → 加载策略 → 创建 Pipeline
2. `build()` 调用 `_prepare_segments()` 将 dict 输入转换为 Segment 列表
3. `_prepare_segments()` 按类型创建 Segment（System→FewShot→Tools→Messages→RAG→State→Extra）
4. Pipeline 按顺序执行阶段：Normalize→Sanitize→Rerank→Allocate→Compress→Assemble
5. 组装 ContextPackage 返回

### 数据模型关键设计
- **Segment** 是核心，使用 `frozen=True`，`model_post_init` 自动填充默认优先级
- **Provenance** 和 **ControlFlags** 通过 `Any` 类型引用避免循环导入（在 `model_post_init` 中延迟导入）
- **AuditEntry** 记录流水线每一步的决策，包含 segment_id + decision + reason_code + reason_detail
- **BudgetPolicy** 的 `elastic_ratios` 使用 `dict[SegmentType, float]` 按类型配比

### Pipeline 阶段协议
```python
class PipelineStage(Protocol):
    @property
    def name(self) -> str: ...
    async def process(self, segments: list[Segment], context: PipelineContext) -> list[Segment]: ...
```
- `PipelineContext` 是 dataclass，在各阶段间传递共享状态（audit_log, warnings, metadata）
- `Pipeline.execute()` 按顺序调度，异常时包装为 `PipelineStageError`

### 模型注册表
- `config/defaults.py` 包含 15+ 主流模型配置（OpenAI/Anthropic/Google/本地）
- `MODEL_ALIASES` 提供简写映射（如 "sonnet" → "claude-sonnet-4-5-20250514"）
- `resolve_model()` 支持精确匹配 → 别名 → 前缀匹配三级查找

## 已遇到并解决的技术问题

1. **Pydantic protected_namespaces 警告**：`ModelConfig` 中的 `model_id` 字段与 Pydantic 的 `model_` 保护命名空间冲突。解决：在 `model_config` 中设置 `"protected_namespaces": ()`。
2. **hatchling 构建需要 README.md**：`pyproject.toml` 中 `readme = "README.md"` 要求文件存在。已创建完整版 README.md。
3. **Windows 终端中文乱码**：`examples/quickstart.py` 在 Windows CMD 中输出中文乱码（编码问题），但逻辑完全正确。可通过 `chcp 65001` 或 Rich console 输出解决。

## 性能指标要求

- 单次组装延迟（不含 LLM）：< 50ms P99（10 Segment, 128K 窗口）
- 内存占用：RSS < 512MB（200K Token 上下文）
- 缓存命中时延迟降低 > 60%

## 六大生产场景

1. **RAG 上下文质量治理**：Pipeline + Sanitize + Budget + Select 策略
2. **多轮对话记忆管理**：Budget + Compress + Rolling Summary + Must-Keep
3. **多 Agent 上下文协调**：Isolate + Context Bus + Handoff + Namespace
4. **安全合规清洗**：Sanitize + Injection 检测 + PII Redaction
5. **Prompt 版本管理与回归**：Observability + Snapshot + Diff + Golden Set
6. **多模型适配与成本优化**：Routing + Budget + Cache

## 原始需求核心约束

- **渐进式 API**：3 层（Facade → Builder → Plugin），每层自洽
- **零配置启动**：`pip install context-forge` → `import` → `build` 无需任何配置文件
- **错误信息即文档**：每条异常都有 What / Why / How
- **CLI 工具**：init / build / inspect / diff / validate / serve 六个子命令
- **可观测性**：Context Snapshot + Prompt Diff + Golden Set 回归 + 核心指标
- **插件化**：Sanitizer / Compressor / Reranker / Router 均为可注册的 Protocol 接口
- **Policy-as-Code**：所有策略通过 YAML 定义，支持版本管理
- **六大场景示例**：每个示例单文件可直接运行（`python examples/xxx.py`）
- **章节映射**：代码中标注 `# → 6.x.x` 对应的章节编号
- **生产考量标注**：省略的生产级细节用 `🏭 生产提示：` 标注

## 项目结构

```
context-forge/
├── src/context_forge/         # 核心引擎
│   ├── models/                # Pydantic v2 数据模型（9 个文件）
│   ├── pipeline/              # 六阶段流水线（8 个文件）
│   ├── budget/                # Token Budget Manager（4 个文件）
│   ├── sanitize/              # 零信任清洗（7 个文件）
│   ├── compress/              # 压缩策略（6 个文件）
│   ├── cache/                 # 缓存层（6 个文件）
│   ├── routing/               # 意图路由（6 个文件）
│   ├── observability/         # 可观测性（6 个文件）
│   ├── antipattern/           # 反模式检测（4 个文件）
│   ├── cli/                   # CLI 工具（10 个文件）
│   ├── errors/                # 结构化异常（2 个文件，16 种异常）
│   ├── config/                # 策略配置（4 个文件）
│   ├── tokenizer/             # Token 计数（5 个文件）
│   ├── plugins/               # 插件注册表（预留扩展）
│   ├── integrations/          # 框架适配器（预留扩展）
│   └── facade.py              # 顶层入口
├── tests/                     # 测试套件（14 单元 + 4 集成）
├── examples/                  # 场景示例（7 Demo + 6 场景 + quickstart）
├── configs/                   # YAML 策略文件（6 个场景策略）
├── scripts/                   # 开发脚本（setup_dev.sh/ps1）
├── .github/workflows/         # CI/CD 流水线（ci.yml + release.yml）
├── pyproject.toml             # 项目配置（hatchling 构建）
├── Dockerfile                 # 多阶段构建
├── docker-compose.yml         # 服务编排
├── Makefile                   # 开发命令快捷方式
├── LICENSE                    # Apache 2.0
├── README.md                  # 生产级 README（900+ 行）
├── CONTRIBUTING.md            # 贡献指南
├── CHANGELOG.md               # 版本历史
└── CLAUDE.md                  # 本文件
```

## 质量审计报告

### 最新审计（2026-02-13）

审计日期：2026-02-13 | 审计工具：pytest 8.x + coverage.py + ruff + mypy

#### 总览评分

| 维度 | 评估 | 得分 |
|------|------|------|
| 功能实现完整度 | 全部 11 模块均为真实逻辑实现，非骨架 | 95/100 |
| 代码质量 | 不可变模型、三段式错误、Protocol 接口、教学标注齐全、facade 已拆分 | 92/100 |
| 测试通过率 | **1037/1037 (100%)**，零失败零错误 | **100/100** |
| 测试覆盖率 | **89.93%**，超过 85% 目标 | **100/100** |
| 测试-实现一致性 | 测试与实现 API 已完全对齐 | **98/100** |
| 类型安全 | mypy 38 errors / 82 files（77% 文件无错误） | 78/100 |
| 代码规范 | ruff 148 issues（排除中文字符误报后），27 可自动修复 | 85/100 |

**综合评分：93/100**

#### 代码规模

| 指标 | 数值 |
|------|------|
| 源码文件数 | 82（含新增 facade_observability.py） |
| 源码总行数 | ~19,900（facade 拆分后净减少约 30 行） |
| 测试文件数 | 29（新增 test_cache_coverage.py, test_compress_summary.py） |
| 测试总行数 | ~22,100 |
| 测试/源码比 | 1.11 |
| 平均行数/文件 | 243 |
| 类定义 | 75 |
| 同步方法/函数 | ~395 |
| 异步方法/函数 | ~82 |

#### 测试运行结果

```
1037 collected, 1037 passed, 0 failed, 0 errors, 23 warnings
代码覆盖率：89.93%（目标 85%，已达标 ✅）
语句总数：5,230 | 未覆盖：387
分支总数：1,386 | 未覆盖：207
```

#### 按模块覆盖率汇总

| 模块 | 语句数 | 未覆盖 | 覆盖率 | 评级 |
|------|--------|--------|--------|------|
| cache/ | 357 | 3 | **98.6%** | 优秀 |
| models/ | 492 | 9 | **97.5%** | 优秀 |
| tokenizer/ | 124 | 3 | **96.3%** | 优秀 |
| (root: facade.py 等) | 252 | 5 | **96.0%** | 优秀 |
| observability/ | 577 | 36 | **92.3%** | 优秀 |
| budget/ | 232 | 13 | **91.7%** | 优秀 |
| routing/ | 407 | 28 | **90.5%** | 优秀 |
| pipeline/ | 445 | 36 | **89.2%** | 良好 |
| sanitize/ | 370 | 28 | **89.1%** | 良好 |
| compress/ | 338 | 23 | **88.3%** | 良好 |
| antipattern/ | 379 | 31 | **87.9%** | 良好 |
| config/ | 186 | 23 | **84.7%** | 良好 |
| cli/ | 970 | 134 | **83.1%** | 良好 |
| errors/ | 101 | 15 | **80.0%** | 达标 |

#### 覆盖率 < 80% 的文件（8 个，从 12 个减少）

| 文件 | 覆盖率 | 未覆盖行 | 说明 |
|------|--------|----------|------|
| `routing/__init__.py` | 66.7% | 3 | 便捷导出函数 |
| `config/loader.py` | 68.5% | 18 | YAML 加载错误路径未充分测试 |
| `routing/base.py` | 74.4% | 6 | RoutingContext 部分属性 |
| `cli/cmd_diff.py` | 75.6% | 27 | diff 子命令复杂分支 |
| `cli/utils.py` | 77.9% | 24 | CLI 工具函数 |
| `cli/cmd_build.py` | 79.6% | 20 | build 子命令错误处理分支 |
| `errors/exceptions.py` | 79.6% | 15 | 部分异常类 `__str__` 未覆盖 |
| `pipeline/sanitize_stage.py` | 79.8% | 11 | 清洗插件动态加载分支 |

#### 文件规模合规

项目编码规范要求：200-400 行典型，800 行上限。

| 文件 | 行数 | 状态 |
|------|------|------|
| `facade.py` | 774 | 合规（从 860 行拆分，-86 行） |
| `antipattern/rules.py` | 747 | 接近上限 |
| `cli/server.py` | 696 | 合规 |
| `errors/exceptions.py` | 491 | 合规 |
| `observability/diff.py` | 481 | 合规 |
| `observability/golden_set.py` | 474 | 合规 |
| `pipeline/rerank.py` | 459 | 合规 |
| `routing/context_bus.py` | 408 | 合规 |
| `facade_observability.py` | 138 | 新文件（从 facade.py 拆出） |

#### Ruff 静态分析（排除中文字符误报 RUF001/002/003）

共 148 个 issue（从 189 个减少 22%），分布如下：

| 规则 | 数量 | 类别 | 可自动修复 |
|------|------|------|-----------|
| E501 行过长 | 46 | 格式 | 否 |
| TC001 仅类型导入 | 39 | 类型 | 否 |
| I001 导入未排序 | 12 | 格式 | 是 |
| RUF022 `__all__` 未排序 | 12 | 格式 | 是 |
| RUF010 显式 f-string 转换 | 8 | 格式 | 是 |
| B028 warn 缺 stacklevel | 6 | Bug | 否 |
| SIM102 可合并 if | 5 | 简化 | 否 |
| F841 未使用变量 | 3 | 清理 | 是 |
| 其他 (SIM/RUF/UP/TC/B) | 17 | 混合 | 部分 |

**已修复的规则（本轮清理）：**
- ~~B904（11 个）~~ → 0（全部添加 from e/from None）
- ~~F401（19 个）~~ → 0（全部清理）
- ~~F541（12 个）~~ → 0（全部修正）

#### mypy 类型检查

共 38 个错误（从 51 个减少 25%），分布在 19 个文件中：

| 错误类别 | 数量 | 严重性 |
|----------|------|--------|
| `valid-type`（`any` vs `Any`） | 5 | 低 — 类型别名拼写 |
| `arg-type` / `call-arg` | 4 | 中 — 参数类型偶尔不匹配 |
| `var-annotated` | 4 | 低 — 缺类型标注 |
| `assignment` | 4 | 中 — 类型不兼容赋值 |
| `no-any-return` | 2 | 低 — Any 返回值 |
| `import-untyped` | 2 | 低 — 缺 yaml stubs |
| 其他 | 17 | 混合 |

**已修复的高危 mypy 错误（本轮修复）：**
- ~~`facade.py:849`：`GoldenSetRunner` 传入不存在的 `snapshot_manager` 参数~~ ✅
- ~~`facade.py:853`：调用不存在的 `GoldenSetRunner.compare()` 方法~~ ✅
- ~~`facade.py:796`：返回 `Coroutine` 而非 `dict[str, Any]`（缺少 `await`）~~ ✅
- ~~`facade.py:450-451`：未检查 union 类型中的 str~~ ✅
- ~~`cli/server.py:304`：`build()` 传入不存在的 `few_shot` 参数~~ ✅
- ~~`cli/server.py:305,320,342,610`：4 个类型不匹配~~ ✅
- ~~`config/schema.py:287`：`to_budget_policy()` 返回 `Any`~~ ✅

### 已修复的问题

#### 测试修复（2026-02-13 第一轮）

原测试套件有 112 个失败 + 5 个错误，根因是测试根据规格说明批量生成后 API 有演化。
已通过 6 个并行修复任务全部解决：

| 修复任务 | 影响文件 | 修复数 | 主要变更 |
|----------|----------|--------|----------|
| test_models.py | tests/unit/test_models.py, tests/conftest.py | 37F+5E | 对齐 42 个字段名/枚举值/构造器签名，修复共享 fixtures |
| test_errors.py | tests/unit/test_errors.py | 10F | 适配中文三段式错误格式，移除不存在的 kwargs |
| test_tokenizer.py | tests/unit/test_tokenizer.py | 10F | `model=` → `encoding_name=`，补充 Protocol 合规测试 |
| test_budget.py | tests/unit/test_budget.py | 28F | 重写全部策略/竞价测试，覆盖率 0%→84-94% |
| test_config + test_pipeline | tests/unit/test_config.py, tests/unit/test_pipeline.py | 17F | 修复 schema 字段名、Pipeline 阶段行为断言 |
| cache+cli+serve+integration | 5 个测试文件 | 19F | 修复导入、mock 路径、服务端 API 适配 |

#### P1 代码缺陷修复（2026-02-13 第二轮）

审计中发现的 4 个代码缺陷（3 个已修复，1 个新发现）：

| 缺陷位置 | 问题描述 | 修复状态 |
|---------|---------|---------|
| `cli/cmd_build.py:141-158` | 反模式检测为占位代码，未集成 `antipattern/` 模块 | ✅ 已修复 |
| `cli/server.py` 多个端点 | API 调用与实际模块方法不匹配 | ✅ 已修复 |
| `facade.py` 路由集成 | 向 `RuleBasedRouter` 传入 `dict` 而非 `RoutingRule` 对象 | ✅ 已修复 |
| `observability/diff.py:418` | `_diff_budget` 引用不存在的 `rigid_budget` 字段 | ✅ 已修复（→ `rigid_used`） |

#### 覆盖率提升（2026-02-13 第三轮）

新增 `tests/unit/test_coverage_gaps.py`（65 个测试），覆盖率从 84.26% 提升至 87.76%：

| 模块 | 提升前 | 提升后 | 变化 |
|------|--------|--------|------|
| `pipeline/assemble.py` | 69% | 100% | +31% |
| `tokenizer/registry.py` | 79% | 100% | +21% |
| `routing/llm_router.py` | 14% | 99% | +85% |
| `observability/diff.py` | 67% | 98% | +31% |
| `routing/context_bus.py` | 63% | 97% | +34% |
| `observability/golden_set.py` | 67% | 96% | +29% |
| `tokenizer/fallback.py` | 55% | 96% | +41% |
| `facade.py` | 46% | 96% | +50% |
| `routing/rule_based.py` | 67% | 93% | +26% |
| `observability/tracing.py` | 33% | 93% | +60% |
| `cache/redis_backend.py` | 0% | 100% | +100% |
| `observability/snapshot.py` | 76% | 90% | +14% |

### 已知问题与技术债务

#### P1 + P2 全部已修复 ✅

| 编号 | 问题 | 修复方式 |
|------|------|----------|
| P1-1 | facade.py 超过 800 行上限 | 拆分为 facade.py (774行) + facade_observability.py (138行) |
| P1-2 | facade.py 4 个 mypy 高危错误 | ObservabilityMixin 使用 DiffEngine 正确 API |
| P1-3 | cli/server.py 5 个 mypy 错误 | few_shot→few_shot_examples、类型保护 |
| P1-4 | B904 raise-without-from（11 处） | 4 个文件 11 处全部添加 from e/from None |
| P2-5 | rule_based.py fallback 死代码 | 捕获 ModelNotFoundError 包装为 RoutingError |
| P2-6 | cache/ 覆盖率 74.9% | 新增 49 个测试，整体→98.6% |
| P2-7 | compress/summary.py 覆盖率 28.1% | 新增 20 个 mock LLM 测试，→100% |
| P2-8 | F401 未使用导入（19 处） | 15 个文件全部清理 |
| P2-9 | F541 空 f-string（12 处） | 5 个文件全部修正 |

#### P3（可选优化）

1. **TC001 仅类型导入（39 处）** — 生产导入应移入 `TYPE_CHECKING` 块
2. **E501 行过长（46 处）** — 超出行宽限制
3. **config/loader.py 覆盖率 68.5%** — YAML 错误处理分支未充分测试
4. **安装 types-PyYAML** — 消除 2 个 `import-untyped` mypy 错误
5. **cli/ 整体覆盖率 83.1%** — cmd_diff.py (75.6%) 和 utils.py (77.9%) 可提升
