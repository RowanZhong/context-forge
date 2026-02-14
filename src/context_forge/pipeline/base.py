"""
Pipeline 基础结构 — 流水线阶段协议与编排器。

→ 6.1.2.1 Pipeline 模式：Normalize → Sanitize → Rerank → Allocate → Compress → Assemble

流水线模式是 Context Forge 的核心架构决策。每个阶段：
- 接收一组 Segment 和上下文信息
- 执行特定的处理逻辑
- 返回处理后的 Segment 列表和审计记录
- 可以被独立测试、替换或跳过

# [Design Decision] 使用 Protocol 而非 ABC 定义阶段接口，
# 因为 Protocol 支持结构化子类型——任何实现了 process() 方法的对象
# 都可以作为阶段使用，无需显式继承。这降低了自定义阶段的接入门槛。

⚠️ 反模式对照：不使用流水线的系统，所有处理逻辑混在一个函数里——
"先拼接 System Prompt，然后如果有 RAG 就加上，然后检查长度..."
这种代码无法复用、无法测试、出了问题无法定位到具体阶段。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from context_forge.models.budget import BudgetPolicy

if TYPE_CHECKING:
    from context_forge.models.audit import AuditEntry
    from context_forge.models.segment import Segment

logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    """
    流水线运行时上下文——在各阶段之间传递的共享状态。

    → 6.1.2.3 Build Context

    # [Design Decision] 使用 dataclass 而非 dict 传递上下文，
    # 因为 dataclass 提供类型安全和 IDE 自动补全，
    # 减少了各阶段之间因为键名拼错而导致的隐蔽 bug。
    """

    model: str = ""
    """目标模型 ID"""

    budget_policy: BudgetPolicy = field(default_factory=BudgetPolicy)
    """预算策略"""

    current_turn: int = 0
    """当前对话轮次"""

    target_namespace: str = "default"
    """目标命名空间"""

    audit_log: list[AuditEntry] = field(default_factory=list)
    """审计日志（各阶段追加）"""

    warnings: list[str] = field(default_factory=list)
    """警告信息"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """自由扩展字段"""

    debug: bool = False
    """是否启用调试模式"""


@runtime_checkable
class PipelineStage(Protocol):
    """
    流水线阶段协议。

    所有流水线阶段必须实现此协议。每个阶段接收 Segment 列表和上下文，
    返回处理后的 Segment 列表。

    最小实现示例::

        class MyCustomStage:
            @property
            def name(self) -> str:
                return "my_custom_stage"

            async def process(
                self,
                segments: list[Segment],
                context: PipelineContext,
            ) -> list[Segment]:
                # 自定义处理逻辑
                return [s for s in segments if len(s.content) > 10]
    """

    @property
    def name(self) -> str:
        """阶段名称，用于日志和审计。"""
        ...

    async def process(
        self,
        segments: list[Segment],
        context: PipelineContext,
    ) -> list[Segment]:
        """
        处理 Segment 列表。

        参数:
            segments: 输入的 Segment 列表
            context: 流水线上下文（含预算策略、审计日志等）

        返回:
            处理后的 Segment 列表
        """
        ...


class Pipeline:
    """
    流水线编排器 — 按顺序执行各阶段。

    → 6.1.2 Context Builder

    Pipeline 负责按顺序调度各阶段，记录每个阶段的执行时间，
    并在阶段失败时执行降级逻辑。

    基本用法::

        pipeline = Pipeline(stages=[
            NormalizeStage(),
            SanitizeStage(),
            RerankStage(),
            AllocateStage(),
            AssembleStage(),
        ])

        result_segments = await pipeline.execute(segments, context)

    跳过特定阶段::

        pipeline = Pipeline(
            stages=[...],
            skip_stages={"sanitize"},  # 跳过清洗阶段
        )
    """

    def __init__(
        self,
        stages: list[PipelineStage] | None = None,
        skip_stages: set[str] | None = None,
    ) -> None:
        """
        初始化流水线。

        参数:
            stages: 阶段列表，按执行顺序排列
            skip_stages: 需要跳过的阶段名称集合
        """
        self._stages = stages or []
        self._skip_stages = skip_stages or set()

    @property
    def stage_names(self) -> list[str]:
        """返回所有阶段的名称列表。"""
        return [s.name for s in self._stages]

    def add_stage(self, stage: PipelineStage, position: int | None = None) -> None:
        """
        添加阶段。

        参数:
            stage: 要添加的阶段
            position: 插入位置（None 表示追加到末尾）
        """
        if position is None:
            self._stages.append(stage)
        else:
            self._stages.insert(position, stage)

    def remove_stage(self, name: str) -> None:
        """按名称移除阶段。"""
        self._stages = [s for s in self._stages if s.name != name]

    def replace_stage(self, name: str, new_stage: PipelineStage) -> None:
        """按名称替换阶段。"""
        self._stages = [new_stage if s.name == name else s for s in self._stages]

    async def execute(
        self,
        segments: list[Segment],
        context: PipelineContext,
    ) -> list[Segment]:
        """
        执行完整的流水线。

        参数:
            segments: 输入的 Segment 列表
            context: 流水线上下文

        返回:
            处理后的 Segment 列表

        异常:
            PipelineStageError: 某个阶段执行失败且没有降级策略
        """
        from context_forge.errors import PipelineStageError

        current = segments

        for stage in self._stages:
            if stage.name in self._skip_stages:
                if context.debug:
                    logger.debug("跳过阶段: %s", stage.name)
                continue

            start_time = time.perf_counter()
            input_count = len(current)

            try:
                if context.debug:
                    logger.debug(
                        "执行阶段: %s（输入 %d 个 Segment）",
                        stage.name,
                        input_count,
                    )

                current = await stage.process(current, context)

                elapsed_ms = (time.perf_counter() - start_time) * 1000

                if context.debug:
                    logger.debug(
                        "阶段 %s 完成：%d → %d 个 Segment（%.1fms）",
                        stage.name,
                        input_count,
                        len(current),
                        elapsed_ms,
                    )

            except PipelineStageError:
                raise  # 已经是结构化异常，直接传播
            except Exception as e:
                # → 6.1.2.4 Failure Mode：构建失败时的降级路径
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                logger.error(
                    "阶段 %s 执行失败（%.1fms）：%s",
                    stage.name,
                    elapsed_ms,
                    e,
                )

                raise PipelineStageError(
                    what=f"流水线阶段 '{stage.name}' 执行失败。",
                    why=str(e),
                    how=f"检查 '{stage.name}' 阶段的配置和输入数据。"
                        f"如果该阶段依赖外部服务，请确认服务可用性。"
                        f"也可以通过 skip_stages={{'{stage.name}'}} 临时跳过该阶段。",
                    stage_name=stage.name,
                ) from e

        return current


def create_default_pipeline(policy: Any = None) -> Pipeline:
    """
    创建默认流水线（六个标准阶段）。

    → 6.1.2.1 Normalize → Sanitize → Rerank → Allocate → Compress → Assemble

    # [DX Decision] 提供一键创建默认流水线的工厂函数，
    # 让 Facade 层的代码保持简洁。

    第三轮增强：添加 CompressStage，根据 PolicyConfig 配置各阶段的参数。

    参数:
        policy: PolicyConfig 实例（可选）。如果提供，将根据配置创建增强版阶段。

    返回:
        配置好默认六个阶段的 Pipeline 实例
    """
    # 延迟导入避免循环依赖
    from context_forge.pipeline.allocate import AllocateStage
    from context_forge.pipeline.assemble import AssembleStage
    from context_forge.pipeline.normalize import NormalizeStage
    from context_forge.pipeline.rerank import RerankStage
    from context_forge.pipeline.sanitize_stage import SanitizeStage

    # 如果没有提供 policy，使用默认配置
    if policy is None:
        return Pipeline(stages=[
            NormalizeStage(),
            SanitizeStage(),
            RerankStage(),
            AllocateStage(),
            AssembleStage(),
        ])

    # 根据 policy 配置创建增强版阶段
    # → 6.4 Sanitize 策略
    sanitize_cfg = policy.sanitize
    sanitize_stage = SanitizeStage(
        max_segment_chars=sanitize_cfg.max_segment_chars,
        strip_html=sanitize_cfg.strip_html,
        detect_injection=sanitize_cfg.injection_detection,
        on_injection=sanitize_cfg.on_injection,
        injection_level=sanitize_cfg.injection_level,
        injection_confidence_threshold=sanitize_cfg.injection_confidence_threshold,
        pii_redaction=sanitize_cfg.pii_redaction,
        pii_patterns=sanitize_cfg.pii_patterns,
        max_repeat_chars=sanitize_cfg.max_repeat_chars,
    )

    # → 6.3.2 Rerank 策略
    rerank_cfg = policy.rerank
    rerank_stage = RerankStage(
        enable_mmr=rerank_cfg.enable_mmr,
        mmr_lambda=rerank_cfg.mmr_lambda,
        similarity_threshold=rerank_cfg.similarity_threshold,
        max_per_type=rerank_cfg.max_per_type,
        enable_temporal_weighting=rerank_cfg.enable_temporal_weighting,
        temporal_decay_rate=rerank_cfg.temporal_decay_rate,
        temporal_min_weight=rerank_cfg.temporal_min_weight,
    )

    # 构建基础阶段列表
    stages = [
        NormalizeStage(),
        sanitize_stage,
        rerank_stage,
        AllocateStage(),
    ]

    # → 6.2.4 Compress 策略（根据配置决定是否启用）
    if policy.compress.enabled:
        from context_forge.compress.engine import CompressEngine
        from context_forge.pipeline.compress_stage import CompressStage

        # 根据配置选择默认压缩器
        default_compressor = None
        compressor_name = getattr(policy.compress, "default_compressor", "truncation")
        if compressor_name == "summary":
            # 尝试使用 LLM 摘要压缩器，降级为截断
            try:
                from context_forge.compress.summary import LLMSummaryCompressor
                default_compressor = LLMSummaryCompressor()
                logger.info("使用 LLM 摘要压缩器（需要 LLM Provider）")
            except Exception:
                logger.warning(
                    "default_compressor='summary' 需要 LLM 支持，"
                    "当前降级为 truncation 模式。"
                )
        elif compressor_name == "dedup":
            from context_forge.compress.dedup import DedupCompressor
            default_compressor = DedupCompressor()

        compress_engine = CompressEngine(
            saturation_threshold=policy.compress.saturation_trigger,
            preserve_must_keep=policy.compress.preserve_must_keep,
            min_segment_tokens=policy.compress.min_segment_tokens,
            default_compressor=default_compressor,
        )
        stages.append(CompressStage(engine=compress_engine))

    # 最后添加 Assemble 阶段
    stages.append(AssembleStage())

    return Pipeline(stages=stages)
