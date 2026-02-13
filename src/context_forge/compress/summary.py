"""
LLM 摘要压缩器 — 使用 LLM 生成抽象摘要。

→ 6.3.3 Write 策略：Rolling Summary 与 Context Distillation

抽象摘要是最高质量的压缩方式，可以将长对话压缩为精炼的要点。
但它依赖 LLM 调用，有成本和延迟。因此本压缩器提供 fallback 机制：
LLM 调用失败时自动降级到截断压缩。

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
    """
    LLM 提供者协议 — 解耦具体的 LLM 客户端。

    # [Design Decision] 使用 Protocol 而非具体类，
    # 允许用户注入任何符合接口的 LLM 客户端（OpenAI/Anthropic/本地模型）。
    """

    async def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """
        生成文本。

        参数:
            prompt: 输入提示
            max_tokens: 最大生成 Token 数

        返回:
            生成的文本

        抛出:
            任何 LLM 调用异常
        """
        ...


class LLMSummaryCompressor:
    """
    LLM 摘要压缩器 — 使用 LLM 生成抽象摘要。

    → 6.3.3 Rolling Summary 实现

    摘要压缩器将多条 Segment 合并为一条 SUMMARY 类型的 Segment。
    摘要保留关键信息，大幅减少 Token 消耗（通常压缩比 0.1-0.3）。

    基本用法::

        from my_llm import MyLLMClient

        provider = MyLLMClient(api_key="...")
        compressor = LLMSummaryCompressor(provider)
        result = await compressor.compress(segments, context)

    启用 fallback（推荐）::

        compressor = LLMSummaryCompressor(
            provider,
            enable_fallback=True,  # LLM 失败时降级到截断
        )

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
        """
        初始化 LLM 摘要压缩器。

        参数:
            provider: LLM 提供者，None 时强制使用 fallback
            enable_fallback: 是否启用 fallback（默认 True）
            max_summary_tokens: 摘要最大 Token 数（默认 500）
        """
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
        """
        使用 LLM 生成抽象摘要。

        → 6.3.3 Rolling Summary 算法

        流程:
        1. 合并所有 Segment 的内容
        2. 构造摘要提示（Prompt）
        3. 调用 LLM 生成摘要
        4. 创建 SUMMARY 类型的 Segment
        5. 失败时降级到 fallback（如果启用）

        参数:
            segments: 待摘要的 Segment 列表
            context: 压缩上下文

        返回:
            CompressionResult，包含单条摘要 Segment

        抛出:
            CompressionError: LLM 调用失败且未启用 fallback
        """
        if not segments:
            return CompressionResult(
                compressed_segments=[],
                original_token_count=0,
                compressed_token_count=0,
                method=self.name,
                parent_segment_ids=[],
            )

        # 计算原始总 Token 数
        original_tokens = sum(seg.token_count or 0 for seg in segments)
        parent_ids = [seg.id for seg in segments]

        # 如果没有 provider，直接使用 fallback
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

        # 尝试使用 LLM 生成摘要
        try:
            summary_text = await self._generate_summary(segments, context)
        except Exception as e:
            # LLM 调用失败
            if self._enable_fallback:
                logger.warning(
                    f"LLM 摘要生成失败：{e}，降级到截断压缩。"
                )
                return await self._fallback_compress(segments, context)
            else:
                raise CompressionError(
                    what="LLM 摘要生成失败",
                    why=str(e),
                    how="检查 LLM API 配置或启用 fallback",
                ) from e

        # 创建摘要 Segment
        summary_segment = Segment(
            type=SegmentType.SUMMARY,
            content=summary_text,
            role="assistant",  # 摘要通常作为助手回复
            provenance=Provenance(
                source_id=f"summary_{parent_ids[0] if parent_ids else 'empty'}",
                source_type=SourceType.COMPRESSION,
                parent_segment_ids=parent_ids,
                compression_method=self.name,
            ),
            token_count=None,  # 由后续流水线重新计数
        )

        # 粗略估算摘要 Token 数（实际应由 Tokenizer 计数）
        # 🏭 生产提示：调用 Tokenizer 获取精确 Token 数
        estimated_tokens = len(summary_text) // 4

        return CompressionResult(
            compressed_segments=[summary_segment],
            original_token_count=original_tokens,
            compressed_token_count=estimated_tokens,
            method=self.name,
            parent_segment_ids=parent_ids,
        )

    async def _generate_summary(
        self, segments: list[Segment], context: CompressContext
    ) -> str:
        """
        调用 LLM 生成摘要。

        → 6.3.3.1 摘要 Prompt 设计

        Prompt 设计原则:
        - 明确任务：总结对话要点
        - 指定格式：简洁的要点列表
        - 控制长度：限制输出 Token 数

        参数:
            segments: 待摘要的 Segment 列表
            context: 压缩上下文

        返回:
            摘要文本
        """
        # 合并所有 Segment 的内容
        combined_content = "\n\n".join(
            f"[{seg.type.value.upper()}] {seg.content}" for seg in segments
        )

        # 构造摘要提示
        prompt = f"""请总结以下内容的关键要点，保留核心信息。
输出格式：简洁的要点列表（2-5 条）。

内容：
{combined_content}

总结："""

        # 调用 LLM 生成
        if self._provider is None:
            raise ValueError("LLM 提供者未配置")

        summary = await self._provider.generate(
            prompt, max_tokens=self._max_summary_tokens
        )

        return summary.strip()

    async def _fallback_compress(
        self, segments: list[Segment], context: CompressContext
    ) -> CompressionResult:
        """
        降级到截断压缩。

        参数:
            segments: 待压缩的 Segment 列表
            context: 压缩上下文

        返回:
            截断压缩结果
        """
        return await self._fallback_compressor.compress(segments, context)
