"""
LLM 摘要压缩器 — 使用 LLM 生成抽象摘要（含滚动摘要）。

→ 6.3.3 Write 策略：Rolling Summary 与 Context Distillation

提供两种摘要压缩器：
- LLMSummaryCompressor：无状态一次性摘要
- RollingSummaryCompressor：有状态增量滚动摘要（跨 build() 调用保留历史）

# [Design Decision] 异步优先设计，支持 LLM API 调用。
# 使用 LLMProvider Protocol 解耦具体的 LLM 客户端（OpenAI/Anthropic/本地）。
"""

from __future__ import annotations

import logging
from typing import Protocol

from context_forge.compress.base import CompressContext, CompressionResult
from context_forge.compress.truncation import TruncationCompressor, TruncationStrategy
from context_forge.errors.exceptions import CompressionError
from context_forge.models.provenance import Provenance, SourceType
from context_forge.models.segment import Segment, SegmentType

logger = logging.getLogger(__name__)


class LLMProvider(Protocol):
    """LLM 提供者协议 — 解耦具体的 LLM 客户端。"""

    async def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """生成文本。"""
        ...


class LLMSummaryCompressor:
    """
    LLM 摘要压缩器 — 使用 LLM 生成抽象摘要（无状态）。

    → 6.3.3 Rolling Summary 实现

    属性:
        provider: LLM 提供者（实现 LLMProvider Protocol）
        enable_fallback: 是否启用 fallback（默认 True）
        max_summary_tokens: 摘要最大 Token 数（默认 500）
    """

    def __init__(
        self,
        provider: LLMProvider | None = None,
        enable_fallback: bool = True,
        max_summary_tokens: int = 500,
    ):
        """初始化 LLM 摘要压缩器。"""
        self._provider = provider
        self._enable_fallback = enable_fallback
        self._max_summary_tokens = max_summary_tokens
        self._fallback_compressor = TruncationCompressor(
            strategy=TruncationStrategy.TAIL
        )

    @property
    def name(self) -> str:
        """压缩器名称。"""
        return "llm_summary"

    async def compress(
        self, segments: list[Segment], context: CompressContext
    ) -> CompressionResult:
        """使用 LLM 生成抽象摘要。"""
        if not segments:
            return CompressionResult(
                compressed_segments=[], original_token_count=0,
                compressed_token_count=0, method=self.name, parent_segment_ids=[],
            )

        original_tokens = sum(seg.token_count or 0 for seg in segments)
        parent_ids = [seg.id for seg in segments]

        if self._provider is None:
            if self._enable_fallback:
                logger.warning(
                    "LLM 提供者未配置，降级到截断压缩。"
                    "提示：传入 LLMProvider 实例以启用摘要压缩。"
                )
                return await self._fallback_compress(segments, context)
            else:
                raise CompressionError(
                    what="LLM 摘要压缩失败",
                    why="未配置 LLM 提供者且未启用 fallback",
                    how="请传入 LLMProvider 实例或设置 enable_fallback=True",
                )

        try:
            summary_text = await self._generate_summary(segments, context)
        except Exception as e:
            if self._enable_fallback:
                logger.warning(
                    f"LLM 摘要生成失败：{e}，降级到截断压缩。"
                )
                return await self._fallback_compress(segments, context)
            else:
                raise CompressionError(
                    what="LLM 摘要生成失败", why=str(e),
                    how="检查 LLM API 配置或启用 fallback",
                ) from e

        summary_segment = Segment(
            type=SegmentType.SUMMARY, content=summary_text, role="assistant",
            provenance=Provenance(
                source_id=f"summary_{parent_ids[0] if parent_ids else 'empty'}",
                source_type=SourceType.COMPRESSION,
                parent_segment_ids=parent_ids, compression_method=self.name,
            ),
            token_count=None,
        )

        # 🏭 生产提示：调用 Tokenizer 获取精确 Token 数
        estimated_tokens = len(summary_text) // 4

        return CompressionResult(
            compressed_segments=[summary_segment],
            original_token_count=original_tokens,
            compressed_token_count=estimated_tokens,
            method=self.name, parent_segment_ids=parent_ids,
        )

    async def _generate_summary(
        self, segments: list[Segment], context: CompressContext
    ) -> str:
        """调用 LLM 生成摘要。"""
        combined_content = "\n\n".join(
            f"[{seg.type.value.upper()}] {seg.content}" for seg in segments
        )
        prompt = f"""请总结以下内容的关键要点，保留核心信息。
输出格式：简洁的要点列表（2-5 条）。

内容：
{combined_content}

总结："""

        if self._provider is None:
            raise ValueError("LLM 提供者未配置")

        summary = await self._provider.generate(
            prompt, max_tokens=self._max_summary_tokens
        )
        return summary.strip()

    async def _fallback_compress(
        self, segments: list[Segment], context: CompressContext
    ) -> CompressionResult:
        """降级到截断压缩。"""
        return await self._fallback_compressor.compress(segments, context)


class RollingSummaryCompressor:
    """
    滚动摘要压缩器 — 跨 build() 调用保留增量摘要状态。

    → 6.3.3 Rolling Summary 真实实现

    与 LLMSummaryCompressor 的关键区别：
    - **有状态**：``_previous_summary`` 在多次 ``compress()`` 调用间保留历史摘要
    - **增量更新**："上轮摘要 + 新消息 → 更新摘要"
    - **轮次感知**：最近 N 轮保持原文，仅对更早轮次执行摘要

    基本用法::

        compressor = RollingSummaryCompressor(provider=my_llm, keep_recent_turns=3)
        result1 = await compressor.compress(segments_round1, context)  # 初始摘要
        result2 = await compressor.compress(segments_round2, context)  # 增量更新
        compressor.reset()  # 重置状态
    """

    def __init__(
        self,
        provider: LLMProvider | None = None,
        keep_recent_turns: int = 2,
        enable_fallback: bool = True,
        max_summary_tokens: int = 500,
    ):
        """
        初始化滚动摘要压缩器。

        参数:
            provider: LLM 提供者，None 时强制使用 fallback
            keep_recent_turns: 保留最近 N 轮原文（默认 2）
            enable_fallback: 是否启用 fallback（默认 True）
            max_summary_tokens: 摘要最大 Token 数（默认 500）
        """
        self._provider = provider
        self._keep_recent_turns = max(0, keep_recent_turns)
        self._enable_fallback = enable_fallback
        self._max_summary_tokens = max_summary_tokens
        self._previous_summary: str | None = None
        self._fallback_compressor = TruncationCompressor(
            strategy=TruncationStrategy.TAIL
        )

    @property
    def name(self) -> str:
        """压缩器名称。"""
        return "rolling_summary"

    @property
    def has_state(self) -> bool:
        """是否存在累积的滚动摘要状态。"""
        return self._previous_summary is not None

    @property
    def previous_summary(self) -> str | None:
        """获取上一次的滚动摘要（只读）。"""
        return self._previous_summary

    def reset(self) -> None:
        """清除滚动摘要状态，恢复到初始状态。"""
        self._previous_summary = None

    def _get_turn_number(self, seg: Segment) -> int | None:
        """安全获取 Segment 的轮次号。"""
        if seg.metadata and hasattr(seg.metadata, "turn_number"):
            return seg.metadata.turn_number
        return None

    def _split_by_turns(
        self, segments: list[Segment]
    ) -> tuple[list[Segment], list[Segment]]:
        """
        按轮次将 Segment 分为"可摘要"和"保留原文"两组。

        使用 metadata.turn_number 判断轮次。若无 turn_number，
        则按列表位置推断：末尾的 keep_recent_turns*2 条视为最近轮次。
        """
        if self._keep_recent_turns <= 0:
            return segments, []

        # 尝试按 turn_number 分组
        has_turn_info = any(self._get_turn_number(seg) is not None for seg in segments)

        if has_turn_info:
            turn_numbers = sorted({
                self._get_turn_number(seg)
                for seg in segments
                if self._get_turn_number(seg) is not None
            })
            if len(turn_numbers) <= self._keep_recent_turns:
                return [], segments

            cutoff_turns = set(turn_numbers[-self._keep_recent_turns:])
            older = [seg for seg in segments if self._get_turn_number(seg) not in cutoff_turns]
            recent = [seg for seg in segments if self._get_turn_number(seg) in cutoff_turns]
            return older, recent

        # 无 turn_number 时按位置推断（每轮 user + assistant = 2 条）
        recent_count = self._keep_recent_turns * 2
        if len(segments) <= recent_count:
            return [], segments
        return segments[:-recent_count], segments[-recent_count:]

    async def compress(
        self, segments: list[Segment], context: CompressContext
    ) -> CompressionResult:
        """
        执行滚动摘要压缩。

        → 6.3.3 Rolling Summary 增量算法

        流程：按轮次拆分 → 旧轮次生成/更新摘要 → 最近轮次保持原文 → 更新状态
        """
        if not segments:
            return CompressionResult(
                compressed_segments=[], original_token_count=0,
                compressed_token_count=0, method=self.name, parent_segment_ids=[],
            )

        original_tokens = sum(seg.token_count or 0 for seg in segments)
        parent_ids = [seg.id for seg in segments]
        older_segments, recent_segments = self._split_by_turns(segments)

        # 所有轮次都在保留范围内，无需摘要
        if not older_segments:
            return CompressionResult(
                compressed_segments=segments,
                original_token_count=original_tokens,
                compressed_token_count=original_tokens,
                method=self.name, parent_segment_ids=parent_ids,
                metadata={"rolling_state": "no_older_turns"},
            )

        # 没有 provider → fallback 或报错
        if self._provider is None:
            if self._enable_fallback:
                logger.warning("LLM 提供者未配置，滚动摘要降级到截断压缩。")
                return await self._fallback_compress(segments, context)
            raise CompressionError(
                what="滚动摘要压缩失败",
                why="未配置 LLM 提供者且未启用 fallback",
                how="请传入 LLMProvider 实例或设置 enable_fallback=True",
            )

        # 调用 LLM 生成/更新摘要
        was_incremental = self._previous_summary is not None
        try:
            summary_text = await self._generate_rolling_summary(older_segments)
        except Exception as e:
            if self._enable_fallback:
                logger.warning(f"滚动摘要生成失败：{e}，降级到截断压缩。")
                return await self._fallback_compress(segments, context)
            raise CompressionError(
                what="滚动摘要生成失败", why=str(e),
                how="检查 LLM API 配置或启用 fallback",
            ) from e

        # 更新滚动状态
        self._previous_summary = summary_text

        summary_segment = Segment(
            type=SegmentType.SUMMARY, content=summary_text, role="assistant",
            provenance=Provenance(
                source_id=f"rolling_summary_{parent_ids[0] if parent_ids else 'empty'}",
                source_type=SourceType.COMPRESSION,
                parent_segment_ids=[seg.id for seg in older_segments],
                compression_method=self.name,
            ),
            token_count=None,
        )

        result_segments = [summary_segment] + list(recent_segments)
        estimated_tokens = len(summary_text) // 4 + sum(
            seg.token_count or 0 for seg in recent_segments
        )

        return CompressionResult(
            compressed_segments=result_segments,
            original_token_count=original_tokens,
            compressed_token_count=estimated_tokens,
            method=self.name, parent_segment_ids=parent_ids,
            metadata={
                "rolling_state": "incremental" if was_incremental else "initial",
                "older_count": len(older_segments),
                "recent_count": len(recent_segments),
            },
        )

    async def _generate_rolling_summary(self, segments: list[Segment]) -> str:
        """生成或增量更新滚动摘要（有 _previous_summary 时构造增量 Prompt）。"""
        new_content = "\n\n".join(
            f"[{seg.type.value.upper()}] {seg.content}" for seg in segments
        )

        if self._previous_summary:
            prompt = (
                f"你是一个对话摘要助手。请根据以下信息更新摘要。\n\n"
                f"上一轮摘要：\n{self._previous_summary}\n\n"
                f"新消息：\n{new_content}\n\n"
                f"请生成更新后的摘要，保留所有关键信息（2-5 条要点）：\n"
            )
        else:
            prompt = (
                f"请总结以下对话内容的关键要点，保留核心信息。\n"
                f"输出格式：简洁的要点列表（2-5 条）。\n\n"
                f"内容：\n{new_content}\n\n总结：\n"
            )

        if self._provider is None:
            raise ValueError("LLM 提供者未配置")

        summary = await self._provider.generate(
            prompt, max_tokens=self._max_summary_tokens
        )
        return summary.strip()

    async def _fallback_compress(
        self, segments: list[Segment], context: CompressContext
    ) -> CompressionResult:
        """降级到截断压缩。"""
        return await self._fallback_compressor.compress(segments, context)
