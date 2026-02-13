"""
压缩引擎 — 编排多个压缩器，按优先级和饱和度触发压缩。

→ 6.2.4.3 压缩引擎：饱和度触发 + 优先级保护

压缩引擎是压缩策略的总控制器：
- 监控 Token 饱和度，超过阈值时触发压缩
- 按优先级顺序压缩：LOW → MEDIUM → HIGH（保护 CRITICAL）
- 保护 must_keep 和 compressible=False 的 Segment
- 支持多轮压缩（去重 → 截断 → 摘要）

# [Design Decision] 压缩引擎不直接实现压缩算法，而是编排现有的压缩器。
# 这符合单一职责原则：引擎负责"何时压缩、压缩什么"，压缩器负责"如何压缩"。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from context_forge.compress.base import CompressContext, Compressor
from context_forge.compress.dedup import DedupCompressor
from context_forge.compress.truncation import TruncationCompressor, TruncationStrategy
from context_forge.errors.exceptions import CompressionError
from context_forge.models.segment import Priority, Segment

if TYPE_CHECKING:
    from context_forge.models.audit import AuditEntry

logger = logging.getLogger(__name__)


class CompressEngine:
    """
    压缩引擎 — 饱和度触发的多阶段压缩编排器。

    → 6.2.4.3 压缩引擎实现

    压缩引擎的核心逻辑：
    1. 计算当前饱和度（total_tokens / available_tokens）
    2. 如果饱和度 < 阈值，不压缩
    3. 如果饱和度 >= 阈值，启动压缩流程：
       - 先去重（删除重复片段）
       - 再压缩 LOW 优先级 Segment
       - 仍超标则压缩 MEDIUM 优先级
       - 最后压缩 HIGH 优先级（保护 CRITICAL）

    基本用法::

        engine = CompressEngine(saturation_threshold=0.85)
        result = await engine.compress(
            segments,
            available_tokens=100000,
            audit_log=[],
        )

    自定义压缩器::

        engine = CompressEngine(
            saturation_threshold=0.85,
            dedup_compressor=MyDedupCompressor(),
            default_compressor=MyTruncationCompressor(),
        )

    属性:
        saturation_threshold: 饱和度阈值，超过时触发压缩（默认 0.85）
        preserve_must_keep: 是否保护 must_keep 标记的 Segment（默认 True）
        min_segment_tokens: 压缩后 Segment 的最小 Token 数（默认 50）
    """

    def __init__(
        self,
        saturation_threshold: float = 0.85,
        preserve_must_keep: bool = True,
        min_segment_tokens: int = 50,
        dedup_compressor: Compressor | None = None,
        default_compressor: Compressor | None = None,
    ):
        """
        初始化压缩引擎。

        参数:
            saturation_threshold: 饱和度阈值（默认 0.85）
            preserve_must_keep: 是否保护 must_keep（默认 True）
            min_segment_tokens: 最小 Segment Token 数（默认 50）
            dedup_compressor: 去重压缩器（默认 DedupCompressor）
            default_compressor: 默认压缩器（默认 TruncationCompressor）
        """
        self._saturation_threshold = max(0.0, min(1.0, saturation_threshold))
        self._preserve_must_keep = preserve_must_keep
        self._min_segment_tokens = min_segment_tokens
        self._dedup_compressor = dedup_compressor or DedupCompressor()
        self._default_compressor = default_compressor or TruncationCompressor(
            strategy=TruncationStrategy.TAIL
        )

    async def compress(
        self,
        segments: list[Segment],
        available_tokens: int,
        audit_log: list[AuditEntry],
        model_name: str | None = None,
    ) -> list[Segment]:
        """
        执行压缩（如果需要）。

        → 6.2.4.3 饱和度触发机制

        流程:
        1. 计算总 Token 数和饱和度
        2. 如果饱和度 < 阈值，直接返回
        3. 否则按优先级顺序压缩：
           - 阶段 1：去重所有可压缩 Segment
           - 阶段 2：压缩 LOW 优先级
           - 阶段 3：压缩 MEDIUM 优先级
           - 阶段 4：压缩 HIGH 优先级（极端情况）
        4. 每阶段后重新检查饱和度，满足即停止

        参数:
            segments: 待压缩的 Segment 列表
            available_tokens: 可用 Token 总量
            audit_log: 审计日志（追加压缩决策）
            model_name: 目标模型名称（传递给压缩器）

        返回:
            压缩后的 Segment 列表

        抛出:
            CompressionError: 压缩失败（所有策略都无法满足预算）
        """
        if not segments:
            return []

        # 计算总 Token 数
        total_tokens = sum(seg.token_count or 0 for seg in segments)
        saturation = total_tokens / available_tokens if available_tokens > 0 else 0.0

        # 如果饱和度低于阈值，不压缩
        if saturation < self._saturation_threshold:
            logger.debug(
                f"饱和度 {saturation:.2%} 低于阈值 {self._saturation_threshold:.2%}，跳过压缩。"
            )
            return segments

        logger.info(
            f"饱和度 {saturation:.2%} 超过阈值 {self._saturation_threshold:.2%}，启动压缩引擎。"
            f"原始 Token 数：{total_tokens}，可用 Token：{available_tokens}"
        )

        current_segments = segments

        # 阶段 1：去重所有可压缩 Segment
        current_segments = await self._dedup_phase(
            current_segments, available_tokens, model_name, audit_log
        )

        # 重新计算饱和度
        total_tokens = sum(seg.token_count or 0 for seg in current_segments)
        if total_tokens <= available_tokens:
            logger.info(f"去重后满足预算，压缩完成。剩余 Token：{total_tokens}")
            return current_segments

        # 阶段 2：压缩 LOW 优先级
        current_segments = await self._compress_by_priority(
            current_segments,
            available_tokens,
            Priority.LOW,
            model_name,
            audit_log,
        )

        total_tokens = sum(seg.token_count or 0 for seg in current_segments)
        if total_tokens <= available_tokens:
            logger.info(f"压缩 LOW 后满足预算，剩余 Token：{total_tokens}")
            return current_segments

        # 阶段 3：压缩 MEDIUM 优先级
        current_segments = await self._compress_by_priority(
            current_segments,
            available_tokens,
            Priority.MEDIUM,
            model_name,
            audit_log,
        )

        total_tokens = sum(seg.token_count or 0 for seg in current_segments)
        if total_tokens <= available_tokens:
            logger.info(f"压缩 MEDIUM 后满足预算，剩余 Token：{total_tokens}")
            return current_segments

        # 阶段 4：压缩 HIGH 优先级（极端情况）
        logger.warning(
            "压缩 LOW 和 MEDIUM 后仍超标，开始压缩 HIGH 优先级 Segment。"
        )
        current_segments = await self._compress_by_priority(
            current_segments,
            available_tokens,
            Priority.HIGH,
            model_name,
            audit_log,
        )

        total_tokens = sum(seg.token_count or 0 for seg in current_segments)
        if total_tokens <= available_tokens:
            logger.info(f"压缩 HIGH 后满足预算，剩余 Token：{total_tokens}")
            return current_segments

        # 仍然超标，抛出异常
        raise CompressionError(
            what="压缩失败",
            why=f"所有压缩策略（去重 + 多级压缩）都无法将 Token 数降低到预算范围内。"
            f"当前 Token：{total_tokens}，可用 Token：{available_tokens}",
            how="检查是否有过多的 CRITICAL 或 must_keep Segment，或增加预算",
        )

    async def _dedup_phase(
        self,
        segments: list[Segment],
        available_tokens: int,
        model_name: str | None,
        audit_log: list[AuditEntry],
    ) -> list[Segment]:
        """
        去重阶段 — 删除重复 Segment。

        参数:
            segments: Segment 列表
            available_tokens: 可用 Token
            model_name: 模型名称
            audit_log: 审计日志

        返回:
            去重后的 Segment 列表
        """
        # 过滤出可压缩的 Segment
        compressible = [seg for seg in segments if self._is_compressible(seg)]
        protected = [seg for seg in segments if not self._is_compressible(seg)]

        if not compressible:
            return segments

        # 计算目标 Token 数（去重通常节省 10-30%）
        total_tokens = sum(seg.token_count or 0 for seg in compressible)
        target_tokens = min(total_tokens, available_tokens)

        context = CompressContext(
            available_tokens=available_tokens,
            target_token_count=target_tokens,
            saturation=total_tokens / available_tokens if available_tokens > 0 else 0.0,
            model_name=model_name,
        )

        # 去重
        result = await self._dedup_compressor.compress(compressible, context)

        logger.info(
            f"去重完成：{len(compressible)} → {len(result.compressed_segments)} 个 Segment，"
            f"Token 数：{result.original_token_count} → {result.compressed_token_count}"
        )

        # 合并保护的和压缩后的 Segment
        return protected + result.compressed_segments

    async def _compress_by_priority(
        self,
        segments: list[Segment],
        available_tokens: int,
        priority: Priority,
        model_name: str | None,
        audit_log: list[AuditEntry],
    ) -> list[Segment]:
        """
        按优先级压缩 Segment。

        参数:
            segments: Segment 列表
            available_tokens: 可用 Token
            priority: 要压缩的优先级
            model_name: 模型名称
            audit_log: 审计日志

        返回:
            压缩后的 Segment 列表
        """
        # 分离目标优先级的可压缩 Segment 和其他 Segment
        target_segments = [
            seg
            for seg in segments
            if seg.effective_priority == priority and self._is_compressible(seg)
        ]
        other_segments = [
            seg
            for seg in segments
            if seg.effective_priority != priority or not self._is_compressible(seg)
        ]

        if not target_segments:
            return segments

        # 计算其他 Segment 的 Token 数
        other_tokens = sum(seg.token_count or 0 for seg in other_segments)
        remaining_budget = max(0, available_tokens - other_tokens)

        # 压缩目标 Segment
        context = CompressContext(
            available_tokens=remaining_budget,
            target_token_count=remaining_budget,
            saturation=1.0,  # 已经超标，强制压缩
            model_name=model_name,
        )

        result = await self._default_compressor.compress(target_segments, context)

        logger.info(
            f"压缩 {priority.value.upper()} 优先级：{len(target_segments)} → "
            f"{len(result.compressed_segments)} 个 Segment，"
            f"Token 数：{result.original_token_count} → {result.compressed_token_count}"
        )

        # 合并
        return other_segments + result.compressed_segments

    def _is_compressible(self, segment: Segment) -> bool:
        """
        判断 Segment 是否可压缩。

        → 6.2.4.3 优先级保护

        不可压缩的情况:
        - CRITICAL 优先级（系统提示、Schema）
        - must_keep 标记
        - compressible=False 标记（ControlFlags）

        参数:
            segment: Segment

        返回:
            True 可压缩，False 不可压缩
        """
        # 保护 CRITICAL 优先级
        if segment.effective_priority == Priority.CRITICAL:
            return False

        # 检查 must_keep 标记
        if self._preserve_must_keep and segment.control:
            if segment.control.must_keep:
                return False

        # 检查 compressible 标记
        if segment.control and hasattr(segment.control, "compressible"):
            if not segment.control.compressible:
                return False

        # Token 数低于最小阈值不压缩（已经很小了）
        if segment.token_count and segment.token_count < self._min_segment_tokens:
            return False

        return True
