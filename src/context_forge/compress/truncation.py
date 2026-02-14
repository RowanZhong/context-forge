"""
截断压缩器 — 最简单的压缩策略。

→ 6.2.4.2 基于规则的压缩策略：截断、去重、模板抽取

截断是零依赖、零延迟的压缩方法，适用于大多数场景。
支持三种截断策略：
- tail：保留头部（最常用，保留系统提示和上下文开头）
- head：保留尾部（适用于需要保留最新消息的场景）
- middle：保留头尾（适用于需要保留开头和结尾的长文档）

# [Design Decision] 默认策略为 tail，因为在 LLM 上下文中，
# 系统提示（头部）比历史消息（尾部）更重要。
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from context_forge.compress.base import CompressContext, CompressionResult
from context_forge.models.provenance import Provenance, SourceType

if TYPE_CHECKING:
    from context_forge.models.segment import Segment


class TruncationStrategy(str, Enum):
    """
    截断策略。

    # [Design Decision] 使用枚举而非字符串，提供类型安全和编辑器补全。
    """

    TAIL = "tail"
    """保留头部，丢弃尾部 — 适用于保留系统提示和上下文开头"""

    HEAD = "head"
    """保留尾部，丢弃头部 — 适用于保留最新消息"""

    MIDDLE = "middle"
    """保留头尾，丢弃中间 — 适用于长文档的首尾摘要"""


class TruncationCompressor:
    """
    截断压缩器 — 最简单高效的压缩方式。

    → 6.2.4.2 截断策略

    截断是所有压缩器的兜底策略：
    - 零外部依赖（无需 LLM 或语义模型）
    - 零延迟（纯 CPU 计算）
    - 确定性（相同输入始终产生相同输出）

    基本用法::

        compressor = TruncationCompressor(strategy=TruncationStrategy.TAIL)
        result = await compressor.compress(segments, context)

    指定头尾比例（仅 middle 策略有效）::

        compressor = TruncationCompressor(
            strategy=TruncationStrategy.MIDDLE,
            head_ratio=0.3,  # 头部占 30%
        )

    属性:
        strategy: 截断策略（tail / head / middle）
        head_ratio: 头部比例（仅 middle 策略有效），取值范围 [0, 1]
    """

    def __init__(
        self,
        strategy: TruncationStrategy = TruncationStrategy.TAIL,
        head_ratio: float = 0.5,
    ):
        """
        初始化截断压缩器。

        参数:
            strategy: 截断策略，默认为 tail（保留头部）
            head_ratio: 头部比例（仅 middle 策略），默认 0.5（头尾各 50%）
        """
        self._strategy = strategy
        self._head_ratio = max(0.0, min(1.0, head_ratio))  # 钳制到 [0, 1]

    @property
    def name(self) -> str:
        """压缩器名称。"""
        return f"truncation_{self._strategy.value}"

    async def compress(
        self, segments: list[Segment], context: CompressContext
    ) -> CompressionResult:
        """
        截断压缩 Segment 列表。

        → 6.2.4.2 截断策略实现

        流程:
        1. 计算原始总 Token 数
        2. 根据 target_token_count 和策略选择保留的 Segment
        3. 更新 Provenance（标记为 COMPRESSION 来源）
        4. 返回压缩结果

        参数:
            segments: 待压缩的 Segment 列表
            context: 压缩上下文

        返回:
            CompressionResult，包含截断后的 Segment

        注意:
            截断不会修改 Segment 内容，只会选择保留哪些 Segment。
            如果需要截断单个 Segment 的内容，使用 _truncate_segment 方法。
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
        target_tokens = context.target_token_count

        # 如果已经满足目标，直接返回
        if original_tokens <= target_tokens:
            return CompressionResult(
                compressed_segments=segments,
                original_token_count=original_tokens,
                compressed_token_count=original_tokens,
                method=self.name,
                parent_segment_ids=[seg.id for seg in segments],
            )

        # 根据策略选择保留的 Segment
        if self._strategy == TruncationStrategy.TAIL:
            kept_segments = self._keep_head(segments, target_tokens)
        elif self._strategy == TruncationStrategy.HEAD:
            kept_segments = self._keep_tail(segments, target_tokens)
        else:  # MIDDLE
            kept_segments = self._keep_middle(segments, target_tokens)

        # 更新 Provenance
        parent_ids = [seg.id for seg in segments]
        updated_segments = []
        for seg in kept_segments:
            new_provenance = Provenance(
                source_id=f"compressed_{seg.id}",
                source_type=SourceType.COMPRESSION,
                parent_segment_ids=parent_ids,
                compression_method=self.name,
            )
            updated_segments.append(
                seg.model_copy(update={"provenance": new_provenance})
            )

        compressed_tokens = sum(seg.token_count or 0 for seg in updated_segments)

        return CompressionResult(
            compressed_segments=updated_segments,
            original_token_count=original_tokens,
            compressed_token_count=compressed_tokens,
            method=self.name,
            parent_segment_ids=parent_ids,
        )

    def _keep_head(self, segments: list[Segment], target: int) -> list[Segment]:
        """保留头部 — 从开始累加直到达到目标 Token 数。"""
        kept = []
        total = 0
        for seg in segments:
            seg_tokens = seg.token_count or 0
            if total + seg_tokens > target:
                # 如果加上当前 Segment 会超标，尝试截断它
                remaining = target - total
                if remaining > 0:
                    truncated = self._truncate_segment(seg, remaining)
                    if truncated:
                        kept.append(truncated)
                break
            kept.append(seg)
            total += seg_tokens
        return kept

    def _keep_tail(self, segments: list[Segment], target: int) -> list[Segment]:
        """保留尾部 — 从末尾反向累加直到达到目标 Token 数。"""
        kept = []
        total = 0
        for seg in reversed(segments):
            seg_tokens = seg.token_count or 0
            if total + seg_tokens > target:
                # 如果加上当前 Segment 会超标，尝试截断它
                remaining = target - total
                if remaining > 0:
                    truncated = self._truncate_segment(seg, remaining, from_tail=True)
                    if truncated:
                        kept.insert(0, truncated)
                break
            kept.insert(0, seg)
            total += seg_tokens
        return kept

    def _keep_middle(self, segments: list[Segment], target: int) -> list[Segment]:
        """保留头尾 — 按比例分配头部和尾部的 Token 预算。"""
        head_target = int(target * self._head_ratio)
        tail_target = target - head_target

        # 先取头部
        head_kept = []
        total = 0
        for seg in segments:
            seg_tokens = seg.token_count or 0
            if total + seg_tokens > head_target:
                remaining = head_target - total
                if remaining > 0:
                    truncated = self._truncate_segment(seg, remaining)
                    if truncated:
                        head_kept.append(truncated)
                break
            head_kept.append(seg)
            total += seg_tokens

        # 再取尾部
        tail_kept = []
        total = 0
        for seg in reversed(segments):
            seg_tokens = seg.token_count or 0
            if total + seg_tokens > tail_target:
                remaining = tail_target - total
                if remaining > 0:
                    truncated = self._truncate_segment(seg, remaining, from_tail=True)
                    if truncated:
                        tail_kept.insert(0, truncated)
                break
            tail_kept.insert(0, seg)
            total += seg_tokens

        # 合并头尾，去除重复（如果头尾有重叠）
        # [Design Decision] 简单合并，不处理重叠情况（极少发生）
        return head_kept + tail_kept

    def _truncate_segment(
        self, segment: Segment, target_tokens: int, from_tail: bool = False
    ) -> Segment | None:
        """
        截断单个 Segment 的内容。

        参数:
            segment: 待截断的 Segment
            target_tokens: 目标 Token 数
            from_tail: 是否从尾部保留（True）还是从头部保留（False）

        返回:
            截断后的 Segment，如果目标 Token 过小则返回 None
        """
        if target_tokens <= 0:
            return None

        # 粗略估算字符比例（假设平均每个 token 约 4 个字符）
        # 🏭 生产提示：应该使用精确的 Tokenizer 计数，而非简单字符截断
        seg_tokens = segment.token_count or 0
        if seg_tokens == 0:
            return None

        char_ratio = target_tokens / seg_tokens
        target_chars = int(len(segment.content) * char_ratio)

        if from_tail:
            # 保留尾部
            truncated_content = segment.content[-target_chars:] if target_chars > 0 else ""
        else:
            # 保留头部
            truncated_content = segment.content[:target_chars] if target_chars > 0 else ""

        if not truncated_content:
            return None

        # 返回新的 Segment（token_count 设为 None，由后续流水线重新计数）
        return segment.with_content(truncated_content).with_token_count(target_tokens)
