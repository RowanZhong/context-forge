"""
压缩器基础协议与数据结构。

→ 6.2.4 压缩策略引擎 + 6.3.3 Write 策略：Rolling Summary 与 Context Distillation

压缩引擎的核心抽象：将一组 Segment 压缩为更少的 Token，
同时保留关键信息。不同于简单截断，压缩器可以使用摘要、去重、提取等策略。

# [Design Decision] 使用 Protocol 而非抽象基类，
# 符合 Python 的"鸭子类型"风格，降低实现者的耦合度。
# 用户可以轻松注册自定义压缩器而无需继承。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from context_forge.models.segment import Segment


@dataclass(frozen=True)
class CompressionResult:
    """
    压缩结果，包含压缩后的 Segment 列表和统计信息。

    → 6.2.4.2 压缩策略选择

    不仅返回压缩后的内容,还携带诊断信息：
    - 原始 Token 数 vs 压缩后 Token 数
    - 压缩方法标识（用于溯源和审计）
    - 被压缩的原始 Segment ID 列表（用于 Provenance 回溯）

    基本用法::

        result = CompressionResult(
            compressed_segments=[summary_segment],
            original_token_count=5000,
            compressed_token_count=500,
            method="abstractive_summary",
            parent_segment_ids=["seg_a", "seg_b", "seg_c"],
        )

    属性:
        compressed_segments: 压缩后的 Segment 列表（可能是单条摘要或多条去重结果）
        original_token_count: 压缩前的总 Token 数
        compressed_token_count: 压缩后的总 Token 数
        method: 压缩方法标识（truncation / summary / dedup 等）
        parent_segment_ids: 参与压缩的原始 Segment ID 列表
        metadata: 扩展元数据（压缩器特定的调试信息）
    """

    compressed_segments: list[Segment]
    original_token_count: int
    compressed_token_count: int
    method: str
    parent_segment_ids: list[str]
    metadata: dict[str, object] | None = None

    @property
    def compression_ratio(self) -> float:
        """
        计算压缩比例（压缩后/压缩前）。

        返回:
            压缩比例，范围 [0, 1]，值越小压缩效果越好
        """
        if self.original_token_count == 0:
            return 1.0
        return self.compressed_token_count / self.original_token_count

    @property
    def tokens_saved(self) -> int:
        """计算节省的 Token 数。"""
        return max(0, self.original_token_count - self.compressed_token_count)


@dataclass(frozen=True)
class CompressContext:
    """
    压缩上下文，向压缩器传递环境信息。

    → 6.2.4.3 饱和度触发机制

    压缩器需要知道当前的预算压力和目标，才能做出合理的决策：
    - 在 80% 饱和度时启用去重，90% 时启用摘要，95% 时强制截断
    - 根据 target_token_count 动态调整压缩强度

    基本用法::

        context = CompressContext(
            available_tokens=50000,
            target_token_count=40000,
            saturation=0.92,
        )

    属性:
        available_tokens: 可用 Token 总量（预算上限）
        target_token_count: 目标 Token 数（期望压缩后的总量）
        saturation: 当前饱和度 [0, 1]，超过阈值触发压缩
        model_name: 目标模型名称（用于选择合适的 Tokenizer）
    """

    available_tokens: int
    target_token_count: int
    saturation: float
    model_name: str | None = None


class Compressor(Protocol):
    """
    压缩器协议 — 所有压缩器必须实现的接口。

    → 6.2.4.1 压缩器接口设计

    压缩器是可插拔的策略组件，内置实现包括：
    - TruncationCompressor：简单截断（三种策略：head/tail/middle）
    - DedupCompressor：语义去重（基于 n-gram Jaccard 相似度）
    - LLMSummaryCompressor：抽象摘要（可选 LLM 调用）

    # [DX Decision] 异步接口支持 LLM 调用等 I/O 操作，
    # 但所有内置压缩器都提供同步快速路径（零外部依赖）。
    """

    @property
    def name(self) -> str:
        """压缩器名称（用于审计和配置引用）。"""
        ...

    async def compress(
        self, segments: list[Segment], context: CompressContext
    ) -> CompressionResult:
        """
        压缩 Segment 列表。

        参数:
            segments: 待压缩的 Segment 列表
            context: 压缩上下文（预算、饱和度等环境信息）

        返回:
            CompressionResult 对象，包含压缩后的 Segment 和统计信息

        抛出:
            CompressionError: 压缩失败（例如 LLM 调用失败且无 fallback）
        """
        ...
