"""
ContextForge — 顶层 Facade API。

这是 Context Forge 的主入口，面向 80% 的用户。
设计目标：**3 行代码完成核心场景。**

→ 6.1.2 Context Builder
→ 渐进式 API 设计的第一层：High-Level Facade

使用示例
--------

最简用法（3 行代码）::

    from context_forge import ContextForge

    forge = ContextForge(model="gpt-4o")
    context = await forge.build(
        system_prompt="你是一个有用的助手。",
        messages=[{"role": "user", "content": "你好"}],
    )
    # context.to_messages() → 直接传给 LLM API

带 RAG 片段::

    context = await forge.build(
        system_prompt="你是一个客服助手。",
        messages=conversation_history,
        rag_chunks=[
            {"content": "退货政策：7天内可退...", "score": 0.95},
            {"content": "退款流程：提交申请后...", "score": 0.87},
        ],
    )

同步用法::

    context = forge.build_sync(
        system_prompt="你是一个助手。",
        messages=[{"role": "user", "content": "你好"}],
    )

# [DX Decision] Facade 是整个引擎的"前门"。
# 它的设计遵循以下原则：
# 1. 最少参数完成最常见任务（system_prompt + messages 即可）
# 2. 所有高级功能通过可选参数暴露（渐进式暴露）
# 3. 合理默认值覆盖 80% 场景（零配置启动）
# 4. 错误信息告诉用户怎么修，而非只说什么坏了
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from context_forge.config.defaults import resolve_model
from context_forge.config.loader import load_policy
from context_forge.config.schema import PolicyConfig
from context_forge.facade_observability import ObservabilityMixin
from context_forge.models.budget import BudgetAllocation, BudgetPolicy
from context_forge.models.context_package import ContextPackage
from context_forge.models.control import ControlFlags
from context_forge.models.metadata import SegmentMetadata
from context_forge.models.provenance import Provenance, SourceType
from context_forge.models.routing import RoutingDecision
from context_forge.models.segment import Priority, Segment, SegmentType
from context_forge.pipeline.base import Pipeline, PipelineContext, create_default_pipeline
from context_forge.tokenizer.registry import get_tokenizer

logger = logging.getLogger(__name__)


class ContextForge(ObservabilityMixin):
    """
    Context Forge 顶层入口 — 3 行代码完成上下文组装。

    这是面向 80% 用户的 High-Level API。它封装了完整的流水线、
    预算管理、清洗和缓存逻辑，用户只需要关心输入和输出。

    渐进式 API 层级：
    - **第一层**（本类）：最少参数，最快上手
    - **第二层**（Pipeline + PipelineContext）：精细控制各阶段
    - **第三层**（Plugin Protocols）：自定义组件

    基本初始化::

        forge = ContextForge(model="gpt-4o")

    带策略文件::

        forge = ContextForge(
            model="claude-sonnet-4-5-20250514",
            policy_path="configs/production.yaml",
        )

    自定义配置::

        forge = ContextForge(
            model="gpt-4o",
            max_context_tokens=32768,
            output_reserved_tokens=2048,
        )

    参数:
        model: 目标模型名称或别名。支持简写如 "gpt-4o"、"sonnet"、"haiku"。
        policy_path: YAML 策略文件路径。None 时使用默认配置。
        max_context_tokens: 覆盖策略中的最大上下文 Token 数。
        output_reserved_tokens: 覆盖策略中的 Output 预留。
        thinking_reserved_tokens: 覆盖策略中的 Thinking Token 预留。
        debug: 是否启用调试模式（详细日志输出）。
        pipeline: 自定义 Pipeline 实例（高级用法）。
        cache_backend: 自定义缓存后端（高级用法）。
        router: 自定义路由器（高级用法）。
        metrics_collector: 自定义指标收集器（高级用法）。
        snapshot_manager: 自定义快照管理器（高级用法）。
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        policy_path: str | Path | None = None,
        max_context_tokens: int | None = None,
        output_reserved_tokens: int | None = None,
        thinking_reserved_tokens: int | None = None,
        debug: bool = False,
        pipeline: Pipeline | None = None,
        cache_backend: Any | None = None,
        router: Any | None = None,
        metrics_collector: Any | None = None,
        snapshot_manager: Any | None = None,
    ) -> None:
        # 解析模型配置
        # [DX Decision] 根据模型名自动确定窗口大小和 tokenizer，
        # 用户不需要查阅各厂商文档来获取这些信息。
        self._model_config = resolve_model(model)
        self._model = self._model_config.model_id
        self._debug = debug

        # 加载策略
        self._policy = load_policy(path=policy_path)

        # 应用运行时覆盖
        overrides: dict[str, Any] = {}
        if max_context_tokens is not None:
            overrides["max_context_tokens"] = max_context_tokens
        elif self._model_config:
            overrides["max_context_tokens"] = self._model_config.max_context_tokens

        if output_reserved_tokens is not None:
            overrides["output_reserved_tokens"] = output_reserved_tokens
        if thinking_reserved_tokens is not None:
            overrides["thinking_reserved_tokens"] = thinking_reserved_tokens
        elif self._model_config.supports_thinking:
            # 自动为 Reasoning Model 预留 Thinking Token
            overrides["thinking_reserved_tokens"] = 8192

        if overrides:
            budget_dict = self._policy.budget.model_dump()
            budget_dict.update(overrides)
            from context_forge.config.schema import BudgetConfig
            self._policy = self._policy.model_copy(update={
                "budget": BudgetConfig(**budget_dict)
            })

        # 创建预算策略
        self._budget_policy = self._policy.to_budget_policy()

        # 创建或使用自定义 Pipeline
        # [DX Decision] 传递 policy 配置给 pipeline，让各阶段根据配置自动调整
        self._pipeline = pipeline or create_default_pipeline(policy=self._policy)

        # Tokenizer
        self._tokenizer = get_tokenizer(self._model)

        # 第三轮：缓存、路由、可观测性
        # [DX Decision] 延迟初始化，仅在启用时才创建对象
        self._cache_manager: Any = None
        self._router: Any = None
        self._metrics_collector: Any = None
        self._snapshot_manager: Any = None

        # 根据 policy 配置初始化可选组件
        if self._policy.cache.enabled:
            if cache_backend is None:
                from context_forge.cache import CacheManager, MemoryCache

                l1_cache = MemoryCache(
                    max_size=self._policy.cache.max_entries,
                    default_ttl=self._policy.cache.ttl_seconds,
                )
                self._cache_manager = CacheManager(l1=l1_cache)
            else:
                self._cache_manager = cache_backend

        if self._policy.routing.enabled:
            if router is None:
                from context_forge.models.routing import RoutingRule
                from context_forge.routing import RuleBasedRouter

                # 将配置中的规则字典转换为 RoutingRule 对象
                rules = [
                    RoutingRule(**rule_dict) if isinstance(rule_dict, dict) else rule_dict
                    for rule_dict in (self._policy.routing.rules or [])
                ]

                self._router = RuleBasedRouter(
                    default_model=self._policy.routing.default_model,
                    rules=rules,
                )
            else:
                self._router = router

        if self._policy.observability.metrics_enabled:
            if metrics_collector is None:
                from context_forge.observability import MetricsCollector

                self._metrics_collector = MetricsCollector()
            else:
                self._metrics_collector = metrics_collector

        if self._policy.observability.snapshot_enabled:
            if snapshot_manager is None:
                from context_forge.observability import SnapshotManager

                self._snapshot_manager = SnapshotManager(
                    storage_dir=self._policy.observability.snapshot_dir
                )
            else:
                self._snapshot_manager = snapshot_manager

        if self._debug:
            logging.basicConfig(level=logging.DEBUG)
            logger.debug(
                "ContextForge 初始化完成：model=%s, "
                "max_tokens=%d, output_reserved=%d, thinking_reserved=%d, "
                "cache=%s, routing=%s, observability=%s",
                self._model,
                self._budget_policy.max_context_tokens,
                self._budget_policy.output_reserved_tokens,
                self._budget_policy.thinking_reserved_tokens,
                "enabled" if self._cache_manager else "disabled",
                "enabled" if self._router else "disabled",
                "enabled" if self._snapshot_manager else "disabled",
            )

    async def build(
        self,
        system_prompt: str = "",
        messages: list[dict[str, str]] | None = None,
        rag_chunks: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        few_shot_examples: list[dict[str, str]] | None = None,
        state: dict[str, Any] | None = None,
        extra_segments: list[Segment] | None = None,
        current_turn: int = 0,
        namespace: str = "default",
        check_antipatterns: bool = False,
    ) -> ContextPackage:
        """
        组装上下文 — 异步主 API。

        这是最常用的方法。传入原始输入，返回组装好的 ContextPackage。

        参数:
            system_prompt: 系统提示文本
            messages: 对话历史消息列表 [{"role": "user/assistant", "content": "..."}]
            rag_chunks: RAG 检索片段 [{"content": "...", "score": 0.9, ...}]
            tools: 工具定义列表 [{"name": "...", "description": "...", ...}]
            few_shot_examples: 少样本示例 [{"role": "user/assistant", "content": "..."}]
            state: 状态锚点 {"key": "value", ...}
            extra_segments: 预构建的 Segment 列表（高级用法）
            current_turn: 当前对话轮次
            namespace: 目标命名空间
            check_antipatterns: 是否在构建后自动检测反模式（→ 6.7）

        返回:
            ContextPackage — 组装结果，调用 .to_messages() 获取 LLM API 格式

        异常:
            BudgetExceededError: 内容超出预算且无法降级
            SanitizationError: 清洗过程中发现不可修复的问题
        """
        start_time = time.perf_counter()

        # 第一步：将各种输入转换为 Segment 列表
        # [DX Decision] 提前转换，用于路由决策和缓存计算
        segments = self._prepare_segments(
            system_prompt=system_prompt,
            messages=messages or [],
            rag_chunks=rag_chunks or [],
            tools=tools or [],
            few_shot_examples=few_shot_examples or [],
            state=state,
            extra_segments=extra_segments or [],
            current_turn=current_turn,
        )

        # 第二步：路由决策（如果启用）
        routing_decision: RoutingDecision | None = None
        target_model = self._model
        adjusted_budget_policy = self._budget_policy

        if self._router:
            # 构建路由上下文
            # → 6.6.1 意图驱动路由
            from context_forge.routing.base import RoutingContext

            # 构建查询文本（用于关键词匹配和复杂度分析）
            query_parts = []
            if system_prompt:
                query_parts.append(system_prompt)
            if messages:
                # 使用最后一条用户消息作为当前查询
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        query_parts.append(msg.get("content", ""))
                        break
            query = " ".join(query_parts)

            routing_context = RoutingContext(
                segments=segments,
                query=query,
                max_budget_tokens=self._budget_policy.max_context_tokens,
                current_turn=current_turn,
                metadata={"namespace": namespace},
            )

            # 执行路由决策（同步方法）
            routing_decision = self._router.route(routing_context)
            assert routing_decision is not None  # 类型守卫
            target_model = routing_decision.selected_model.model_id
            # [DX Decision] RoutingDecision 暂不支持 budget_adjustment，
            # 可在后续版本扩展以支持动态预算调整
            if hasattr(routing_decision, 'budget_adjustment') and routing_decision.budget_adjustment:
                adjusted_budget_policy = self._budget_policy.model_copy(
                    update=routing_decision.budget_adjustment
                )

        # 第三步：检查缓存（如果启用）
        cache_key = None
        if self._cache_manager:
            # 构建缓存键（基于输入和模型）
            import hashlib
            import json

            cache_input = {
                "model": target_model,
                "system_prompt": system_prompt,
                "messages": messages,
                "rag_chunks": rag_chunks,
                "tools": tools,
                "few_shot_examples": few_shot_examples,
                "state": state,
                "namespace": namespace,
            }
            cache_key = hashlib.sha256(
                json.dumps(cache_input, sort_keys=True).encode()
            ).hexdigest()

            cached_entry = await self._cache_manager.get(cache_key)
            if cached_entry:
                if self._debug:
                    logger.debug(f"缓存命中：{cache_key[:16]}...")
                # 记录缓存命中指标
                if self._metrics_collector:
                    self._metrics_collector.record("cache_hit", 1.0, tags={"model": target_model})
                # 反序列化 ContextPackage
                cached_dict = json.loads(cached_entry.value)
                # 简化：直接返回重新构建的 package
                # 🏭 生产提示：应该实现 ContextPackage.from_dict() 方法
                # 这里为了向后兼容，暂时跳过缓存（在测试中禁用）
                if self._debug:
                    logger.debug("缓存命中，但反序列化暂未实现，继续构建")

        # 第四步：创建 Pipeline 上下文
        pipeline_context = PipelineContext(
            model=target_model,
            budget_policy=adjusted_budget_policy,
            current_turn=current_turn,
            target_namespace=namespace,
            debug=self._debug,
        )

        # 传递预算信息给 CompressStage
        pipeline_context.metadata["available_tokens"] = adjusted_budget_policy.available_for_content
        pipeline_context.metadata["model_name"] = target_model

        # 第五步：执行流水线
        result_segments = await self._pipeline.execute(segments, pipeline_context)

        # 第六步：组装 ContextPackage
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        budget_allocation = pipeline_context.metadata.get("budget_allocation")
        if not isinstance(budget_allocation, BudgetAllocation):
            # 如果 Allocate 阶段没有生成分配记录，创建一个基础的
            total_tokens = sum(s.token_count or 0 for s in result_segments)
            budget_allocation = BudgetAllocation(
                total_budget=adjusted_budget_policy.max_context_tokens,
                content_budget=adjusted_budget_policy.available_for_content,
                total_used=total_tokens,
            )

        package = ContextPackage(
            segments=result_segments,
            audit_log=pipeline_context.audit_log,
            budget_allocation=budget_allocation,
            routing_decision=routing_decision,
            model=target_model,
            policy_version=self._policy.version,
            assembly_duration_ms=elapsed_ms,
            warnings=pipeline_context.warnings,
        )

        # 第七步：保存到缓存（如果启用）
        if self._cache_manager and cache_key:
            import json

            from context_forge.cache.base import CacheEntry

            # 序列化 ContextPackage
            package_dict = package.to_snapshot_dict()
            # 使用 default=str 处理日期时间等无法序列化的对象
            cache_entry = CacheEntry(
                value=json.dumps(package_dict, ensure_ascii=False, default=str)
            )
            await self._cache_manager.set(cache_key, cache_entry)
            if self._debug:
                logger.debug(f"缓存保存：{cache_key[:16]}...")

        # 第八步：保存快照（如果启用）
        if self._snapshot_manager:
            snapshot_id = await self._snapshot_manager.save(package)
            if self._debug:
                logger.debug(f"快照已保存：{snapshot_id}")

        # 第九步：记录指标（如果启用）
        if self._metrics_collector:
            self._metrics_collector.collect_from_package(package)

        # 第十步：反模式检测（如果启用）
        # → 6.7 反模式检测与诊断
        if check_antipatterns or self._policy.antipattern.check_on_build:
            antipattern_results = self.detect_antipatterns(package, format="raw")
            if antipattern_results and isinstance(antipattern_results, list):
                import warnings

                from context_forge.antipattern.base import AntiPatternSeverity

                # 统计各级别问题数量
                critical_count = len([r for r in antipattern_results if r.severity == AntiPatternSeverity.CRITICAL])
                warning_count = len([r for r in antipattern_results if r.severity == AntiPatternSeverity.WARNING])

                # 发出警告
                warnings.warn(
                    f"检测到 {len(antipattern_results)} 个反模式问题 "
                    f"(CRITICAL: {critical_count}, WARNING: {warning_count})。"
                    f"调用 detect_antipatterns() 查看详情。",
                    UserWarning,
                    stacklevel=2,
                )

                # 如果配置要求，在检测到 CRITICAL 时抛异常
                if self._policy.antipattern.fail_on_critical and critical_count > 0:
                    from context_forge.errors.exceptions import AntiPatternError
                    raise AntiPatternError(
                        f"检测到 {critical_count} 个 CRITICAL 级别的反模式问题。\n"
                        f"调用 detect_antipatterns() 查看详情。"
                    )

        if self._debug:
            logger.debug("组装完成：%s", package.summary())

        return package

    def build_sync(
        self,
        system_prompt: str = "",
        messages: list[dict[str, str]] | None = None,
        rag_chunks: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        few_shot_examples: list[dict[str, str]] | None = None,
        state: dict[str, Any] | None = None,
        extra_segments: list[Segment] | None = None,
        current_turn: int = 0,
        namespace: str = "default",
        check_antipatterns: bool = False,
    ) -> ContextPackage:
        """
        组装上下文 — 同步便捷方法。

        # [DX Decision] 为不使用 async 的用户提供同步包装。
        # 在 Jupyter Notebook 或简单脚本中特别有用。
        # 内部使用 asyncio.run()，如果已在 event loop 中运行
        # 会自动检测并给出友好提示。

        参数和返回值与 build() 相同。
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            # 已在 event loop 中（如 Jupyter），提供友好提示
            import warnings
            warnings.warn(
                "检测到已有运行中的 event loop（可能在 Jupyter 环境中）。"
                "build_sync() 无法在已有 event loop 中使用。"
                "请使用 'await forge.build(...)' 代替，"
                "或安装 nest_asyncio：pip install nest_asyncio",
                RuntimeWarning,
                stacklevel=2,
            )
            try:
                import nest_asyncio
                nest_asyncio.apply()
            except ImportError:
                raise RuntimeError(
                    "在已有 event loop 中调用 build_sync() 需要 nest_asyncio。\n"
                    "→ 修复方案 1：使用 'await forge.build(...)' 代替\n"
                    "→ 修复方案 2：pip install nest_asyncio"
                ) from None

        return asyncio.run(self.build(
            system_prompt=system_prompt,
            messages=messages,
            rag_chunks=rag_chunks,
            tools=tools,
            few_shot_examples=few_shot_examples,
            state=state,
            extra_segments=extra_segments,
            current_turn=current_turn,
            namespace=namespace,
            check_antipatterns=check_antipatterns,
        ))

    def _prepare_segments(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        rag_chunks: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        few_shot_examples: list[dict[str, str]],
        state: dict[str, Any] | None,
        extra_segments: list[Segment],
        current_turn: int,
    ) -> list[Segment]:
        """
        将各种输入统一转换为 Segment 列表。

        # [DX Decision] 用户可以传入简单的 dict 格式，
        # Facade 负责将它们包装为结构化的 Segment 对象。
        # 这样用户不需要了解 Segment 模型的细节就能使用引擎。
        """
        segments: list[Segment] = []

        # 1. System Prompt → CRITICAL 优先级，锁定位置
        if system_prompt:
            segments.append(Segment(
                type=SegmentType.SYSTEM,
                content=system_prompt,
                role="system",
                priority=Priority.CRITICAL,
                control=ControlFlags(
                    lock_position=True,
                    compressible=False,
                    must_keep=True,
                ),
                provenance=Provenance(
                    source_id="system_prompt",
                    source_type=SourceType.SYSTEM_CONFIG,
                ),
                metadata=SegmentMetadata(turn_number=0),
            ))

        # 2. Few-Shot 示例
        for i, example in enumerate(few_shot_examples):
            segments.append(Segment(
                type=SegmentType.FEW_SHOT,
                content=example.get("content", ""),
                role=example.get("role", "user"),
                priority=Priority.HIGH,
                control=ControlFlags(lock_position=True),
                provenance=Provenance(
                    source_id=f"few_shot_{i}",
                    source_type=SourceType.SYSTEM_CONFIG,
                ),
            ))

        # 3. 工具定义
        for i, tool in enumerate(tools):
            # 工具定义序列化为 JSON 字符串
            import json
            tool_content = json.dumps(tool, ensure_ascii=False, indent=2)
            segments.append(Segment(
                type=SegmentType.TOOL_DEFINITION,
                content=tool_content,
                role="system",
                priority=Priority.HIGH,
                control=ControlFlags(namespace="tools"),
                provenance=Provenance(
                    source_id=f"tool_{tool.get('name', i)}",
                    source_type=SourceType.SYSTEM_CONFIG,
                ),
            ))

        # 4. 对话历史
        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            msg_type = {
                "user": SegmentType.USER,
                "assistant": SegmentType.ASSISTANT,
                "system": SegmentType.SYSTEM,
            }.get(role, SegmentType.USER)

            segments.append(Segment(
                type=msg_type,
                content=content,
                role=role,
                provenance=Provenance(
                    source_id=f"message_{i}",
                    source_type=SourceType.USER_INPUT if role == "user" else SourceType.SYSTEM_CONFIG,
                ),
                metadata=SegmentMetadata(turn_number=i // 2),
            ))

        # 5. RAG 检索片段
        for i, chunk in enumerate(rag_chunks):
            content = chunk.get("content", "") if isinstance(chunk, dict) else str(chunk)
            score = chunk.get("score", 0.0) if isinstance(chunk, dict) else 0.0
            source_id = chunk.get("source_id", f"rag_{i}") if isinstance(chunk, dict) else f"rag_{i}"
            uri = chunk.get("uri") if isinstance(chunk, dict) else None

            segments.append(Segment(
                type=SegmentType.RAG,
                content=content,
                role="user",  # RAG 内容通常作为 user 角色注入
                priority=Priority.MEDIUM,
                provenance=Provenance(
                    source_id=source_id,
                    source_type=SourceType.RAG_RETRIEVAL,
                    uri=uri,
                    retrieval_score=score,
                ),
                metadata=SegmentMetadata(
                    retrieval_score=score,
                    turn_number=current_turn,
                ),
            ))

        # 6. 状态锚点（→ 6.3.1.2 State Anchoring）
        if state:
            import json
            state_content = "当前状态：\n" + json.dumps(state, ensure_ascii=False, indent=2)
            segments.append(Segment(
                type=SegmentType.STATE,
                content=state_content,
                role="system",
                priority=Priority.HIGH,
                control=ControlFlags(must_keep=True),
                provenance=Provenance(
                    source_id="state_anchor",
                    source_type=SourceType.SYSTEM_CONFIG,
                ),
            ))

        # 7. 用户预构建的 Segment（高级用法）
        segments.extend(extra_segments)

        return segments

    # --- 便捷属性 ---

    @property
    def model(self) -> str:
        """当前模型 ID。"""
        return self._model

    @property
    def policy(self) -> PolicyConfig:
        """当前策略配置。"""
        return self._policy

    @property
    def budget_policy(self) -> BudgetPolicy:
        """当前预算策略。"""
        return self._budget_policy

    @property
    def pipeline(self) -> Pipeline:
        """当前流水线实例（可用于高级定制）。"""
        return self._pipeline

    # --- 反模式检测便捷方法（第四轮新增）---

    def detect_antipatterns(
        self,
        package: ContextPackage,
        format: str = "text",
    ) -> list[Any] | str:
        """
        检测 ContextPackage 中的反模式。

        → 6.7 反模式检测与诊断

        此方法使用默认的反模式检测器检查上下文组装结果，
        发现潜在的配置问题、安全风险或性能瓶颈。

        参数:
            package: 要检测的 ContextPackage
            format: 输出格式（"text" / "json" / "rich" / "raw"）
                   - "raw" 返回 DetectionResult 列表
                   - 其他格式返回格式化的字符串报告

        返回:
            检测结果（格式由 format 参数决定）

        使用示例::

            # 检测并打印文本报告
            report = forge.detect_antipatterns(package, format="text")
            print(report)

            # 获取原始结果列表
            results = forge.detect_antipatterns(package, format="raw")
            for result in results:
                if result.severity == AntiPatternSeverity.CRITICAL:
                    print(f"严重问题: {result.title}")
        """
        from context_forge.antipattern import create_default_detector
        from context_forge.antipattern.base import DetectionContext

        # 创建检测器
        detector = create_default_detector()

        # 构建检测上下文（包含策略配置中的阈值）
        config = {
            "critical_ratio_threshold": self._policy.antipattern.critical_ratio_threshold,
            "rigid_budget_threshold": self._policy.antipattern.rigid_budget_threshold,
            "compression_ratio_threshold": self._policy.antipattern.compression_ratio_threshold,
            "ttl_days_threshold": self._policy.antipattern.ttl_days_threshold,
            "routing_effectiveness_threshold": self._policy.antipattern.routing_effectiveness_threshold,
        }

        context = DetectionContext(
            segments=package.segments,
            budget_policy=self._budget_policy,
            budget_allocation=package.budget_allocation,
            audit_log=package.audit_log,
            model=package.model,
            policy_version=package.policy_version,
            config=config,
        )

        # 执行检测
        results = detector.detect(context)

        # 根据格式返回
        if format == "raw":
            return results
        else:
            return detector.format_report(results, format=format)

    # 可观测性便捷方法 diff() / snapshot() / golden_record()
    # 通过 ObservabilityMixin 注入，参见 facade_observability.py

    def __repr__(self) -> str:
        return (
            f"ContextForge(model='{self._model}', "
            f"max_tokens={self._budget_policy.max_context_tokens:,}, "
            f"policy='{self._policy.name}')"
        )
