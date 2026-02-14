"""
压缩阶段 — Pipeline 中的压缩步骤。

→ 6.1.2.1 Pipeline 架构：5 阶段流水线

压缩阶段在 Allocate 之后、Assemble 之前执行：
1. Normalize：填充 Token 计数
2. Sanitize：清洗和安全检查
3. Rerank：排序和去重
4. Allocate：预算分配
5. **Compress**：饱和度触发的压缩（本阶段）
6. Assemble：最终组装

# [Design Decision] 压缩阶段放在 Allocate 之后，
# 因为只有完成预算分配后才能计算饱和度，判断是否需要压缩。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from context_forge.compress.engine import CompressEngine
from context_forge.errors.exceptions import PipelineStageError
from context_forge.models.audit import AuditEntry, DecisionType, ReasonCode

if TYPE_CHECKING:
    from context_forge.models.segment import Segment
    from context_forge.pipeline.base import PipelineContext

logger = logging.getLogger(__name__)


class CompressStage:
    """
    压缩阶段 — 饱和度触发的自适应压缩。

    → 6.2.4.3 压缩阶段集成

    压缩阶段使用 CompressEngine 执行压缩：
    - 从 PipelineContext 读取预算信息（available_tokens）
    - 计算饱和度并触发压缩（如果需要）
    - 将压缩决策记录到审计日志

    基本用法（在 Pipeline 中自动调用）::

        stage = CompressStage(engine=CompressEngine())
        compressed = await stage.process(segments, context)

    自定义压缩引擎::

        from context_forge.compress.engine import CompressEngine
        from context_forge.compress.summary import LLMSummaryCompressor

        engine = CompressEngine(
            saturation_threshold=0.9,  # 更高的阈值
            default_compressor=LLMSummaryCompressor(provider),
        )
        stage = CompressStage(engine=engine)

    属性:
        engine: 压缩引擎（CompressEngine 实例）
    """

    def __init__(self, engine: CompressEngine | None = None):
        """
        初始化压缩阶段。

        参数:
            engine: 压缩引擎，None 时使用默认配置
        """
        self._engine = engine or CompressEngine()

    @property
    def name(self) -> str:
        """阶段名称。"""
        return "compress"

    async def process(
        self, segments: list[Segment], context: PipelineContext
    ) -> list[Segment]:
        """
        执行压缩阶段。

        → 6.2.4.3 压缩触发逻辑

        流程:
        1. 从 PipelineContext 读取预算信息
        2. 调用 CompressEngine 执行压缩
        3. 将压缩决策记录到审计日志
        4. 返回压缩后的 Segment 列表

        参数:
            segments: 输入 Segment 列表（已完成 Allocate）
            context: Pipeline 上下文（包含预算信息）

        返回:
            压缩后的 Segment 列表

        抛出:
            PipelineStageError: 压缩失败
        """
        if not segments:
            return []

        # 从 context 读取预算信息
        # 🏭 生产提示：PipelineContext 应该包含 budget_info 字段
        # 这里假设通过 metadata 传递
        available_tokens = context.metadata.get("available_tokens", 100_000)
        model_name = context.metadata.get("model_name")

        # 记录压缩前状态
        original_count = len(segments)
        original_tokens = sum(seg.token_count or 0 for seg in segments)

        logger.info(
            f"压缩阶段开始：{original_count} 个 Segment，{original_tokens} Token"
        )

        try:
            # 调用压缩引擎
            compressed_segments = await self._engine.compress(
                segments=segments,
                available_tokens=available_tokens,
                audit_log=context.audit_log,
                model_name=model_name,
            )
        except Exception as e:
            logger.error(f"压缩阶段失败：{e}")
            raise PipelineStageError(
                stage_name=self.name,
                what="压缩失败",
                why=str(e),
                how="检查压缩引擎配置或增加预算",
            ) from e

        # 记录压缩后状态
        compressed_count = len(compressed_segments)
        compressed_tokens = sum(seg.token_count or 0 for seg in compressed_segments)

        logger.info(
            f"压缩阶段完成：{compressed_count} 个 Segment，{compressed_tokens} Token "
            f"(节省 {original_tokens - compressed_tokens} Token)"
        )

        # 记录审计日志
        if original_tokens != compressed_tokens:
            # 为压缩操作创建单条审计记录（使用第一个 segment 的 ID 作为代表）
            # 🏭 生产提示：可以为每个被压缩的 Segment 创建单独的审计记录
            representative_id = segments[0].id if segments else "batch_compress"
            context.audit_log.append(
                AuditEntry(
                    segment_id=representative_id,
                    decision=DecisionType.COMPRESS,
                    reason_code=ReasonCode.COMPRESS_WINDOW_SATURATION,
                    reason_detail=(
                        f"饱和度超过阈值，压缩 {original_count - compressed_count} 个 Segment，"
                        f"节省 {original_tokens - compressed_tokens} Token"
                    ),
                    pipeline_stage=self.name,
                    token_impact=-(original_tokens - compressed_tokens),
                    metadata={
                        "original_count": original_count,
                        "compressed_count": compressed_count,
                        "original_tokens": original_tokens,
                        "compressed_tokens": compressed_tokens,
                        "compression_ratio": compressed_tokens / original_tokens
                        if original_tokens > 0
                        else 1.0,
                        "affected_segment_ids": [seg.id for seg in segments],
                    },
                )
            )

        return compressed_segments
