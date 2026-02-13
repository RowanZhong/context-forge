"""
压缩模块 — 自适应压缩策略引擎。

→ 6.2.4 压缩策略引擎 + 6.3.3 Write 策略

压缩模块提供多种压缩策略，帮助在 Token 预算有限时保留关键信息：
- 截断（Truncation）：最简单，零依赖
- 去重（Deduplication）：删除重复片段，基于 n-gram Jaccard
- 摘要（Summary）：使用 LLM 生成抽象摘要（可选）

压缩引擎按饱和度触发，按优先级保护，确保关键 Segment 不被丢弃。

# [DX Decision] 暴露三个层次的 API：
# 1. 高级 API：CompressEngine（自动编排）
# 2. 中级 API：各种 Compressor（按需使用）
# 3. 低级 API：CompressionResult / CompressContext（自定义实现）
"""

from context_forge.compress.base import (
    CompressContext,
    CompressionResult,
    Compressor,
)
from context_forge.compress.dedup import DedupCompressor
from context_forge.compress.engine import CompressEngine
from context_forge.compress.summary import LLMProvider, LLMSummaryCompressor
from context_forge.compress.truncation import TruncationCompressor, TruncationStrategy

__all__ = [
    # 基础协议与数据结构
    "Compressor",
    "CompressionResult",
    "CompressContext",
    # 压缩器实现
    "TruncationCompressor",
    "TruncationStrategy",
    "DedupCompressor",
    "LLMSummaryCompressor",
    "LLMProvider",
    # 压缩引擎
    "CompressEngine",
]
