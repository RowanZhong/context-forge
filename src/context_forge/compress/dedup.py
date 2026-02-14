"""
去重压缩器 — 基于语义相似度的去重。

→ 6.2.4.2 基于规则的压缩策略：去重

在 RAG 场景中，检索结果常有重复或高度相似的片段。
去重压缩器使用 n-gram Jaccard 相似度检测重复，保留高优先级或高分数的版本。

# [Design Decision] 使用 n-gram Jaccard 而非嵌入向量，
# 因为零依赖、零延迟，且对文本重复检测效果足够好（F1 > 0.9）。
# 如需更精确的语义去重，可以继承本类并重写 _compute_similarity 方法。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from context_forge.compress.base import CompressContext, CompressionResult
from context_forge.models.provenance import Provenance, SourceType

if TYPE_CHECKING:
    from context_forge.models.segment import Segment


class DedupCompressor:
    """
    去重压缩器 — 删除重复或高度相似的 Segment。

    → 6.2.4.2 去重策略

    去重流程:
    1. 计算所有 Segment 两两之间的 Jaccard 相似度
    2. 相似度超过阈值的 Segment 对视为重复
    3. 保留优先级更高或检索分数更高的版本
    4. 删除重复项

    基本用法::

        compressor = DedupCompressor(similarity_threshold=0.85)
        result = await compressor.compress(segments, context)

    调整 n-gram 大小::

        compressor = DedupCompressor(
            similarity_threshold=0.85,
            ngram_size=3,  # 使用 3-gram（默认 2-gram）
        )

    属性:
        similarity_threshold: 相似度阈值，超过此值视为重复，取值范围 [0, 1]
        ngram_size: n-gram 大小，默认 2（bigram）
    """

    def __init__(self, similarity_threshold: float = 0.85, ngram_size: int = 2):
        """
        初始化去重压缩器。

        参数:
            similarity_threshold: 相似度阈值（默认 0.85）
            ngram_size: n-gram 大小（默认 2）
        """
        self._threshold = max(0.0, min(1.0, similarity_threshold))
        self._ngram_size = max(1, ngram_size)

    @property
    def name(self) -> str:
        """压缩器名称。"""
        return f"dedup_jaccard_{self._ngram_size}gram"

    async def compress(
        self, segments: list[Segment], context: CompressContext
    ) -> CompressionResult:
        """
        去重压缩 Segment 列表。

        → 6.2.4.2 去重算法实现

        流程:
        1. 计算所有 Segment 的 n-gram 集合
        2. 遍历 Segment 对，计算 Jaccard 相似度
        3. 标记相似度超过阈值的 Segment 为重复
        4. 根据优先级和检索分数选择保留版本
        5. 更新 Provenance

        参数:
            segments: 待去重的 Segment 列表
            context: 压缩上下文

        返回:
            CompressionResult，包含去重后的 Segment
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

        # 计算所有 Segment 的 n-gram 集合
        ngram_sets = [self._compute_ngrams(seg.content) for seg in segments]

        # 标记要保留的 Segment（初始全部保留）
        to_keep = [True] * len(segments)

        # 遍历所有 Segment 对，标记重复项
        for i in range(len(segments)):
            if not to_keep[i]:
                continue  # 已被标记为删除

            for j in range(i + 1, len(segments)):
                if not to_keep[j]:
                    continue

                # 计算 Jaccard 相似度
                similarity = self._jaccard_similarity(ngram_sets[i], ngram_sets[j])

                if similarity >= self._threshold:
                    # 重复！选择保留哪个
                    # 优先级：CRITICAL > HIGH > MEDIUM > LOW
                    # 相同优先级时比较检索分数（RAG 场景）
                    if self._should_keep_first(segments[i], segments[j]):
                        to_keep[j] = False
                    else:
                        to_keep[i] = False
                        break  # 当前 Segment 已被删除，跳出内层循环

        # 收集保留的 Segment
        kept_segments = [seg for idx, seg in enumerate(segments) if to_keep[idx]]

        # 更新 Provenance（标记为去重来源）
        parent_ids = [seg.id for seg in segments]
        updated_segments = []
        for seg in kept_segments:
            new_provenance = Provenance(
                source_id=f"dedup_{seg.id}",
                source_type=SourceType.COMPRESSION,
                parent_segment_ids=parent_ids,
                compression_method=self.name,
            )
            updated_segments.append(
                seg.model_copy(update={"provenance": new_provenance})
            )

        compressed_tokens = sum(seg.token_count or 0 for seg in updated_segments)

        # 🏭 生产提示：记录被删除的 Segment ID 用于调试
        removed_count = len(segments) - len(kept_segments)
        metadata = {"removed_count": removed_count, "threshold": self._threshold}

        return CompressionResult(
            compressed_segments=updated_segments,
            original_token_count=original_tokens,
            compressed_token_count=compressed_tokens,
            method=self.name,
            parent_segment_ids=parent_ids,
            metadata=metadata,
        )

    def _compute_ngrams(self, text: str) -> set[str]:
        """
        计算文本的 n-gram 集合。

        参数:
            text: 输入文本

        返回:
            n-gram 字符串集合
        """
        # 简单分词（按空白符）
        # 🏭 生产提示：对中文应该使用字符级 n-gram 或专用分词器
        tokens = text.split()
        ngrams = set()

        for i in range(len(tokens) - self._ngram_size + 1):
            ngram = " ".join(tokens[i : i + self._ngram_size])
            ngrams.add(ngram)

        return ngrams

    def _jaccard_similarity(self, set1: set[str], set2: set[str]) -> float:
        """
        计算两个集合的 Jaccard 相似度。

        Jaccard(A, B) = |A ∩ B| / |A ∪ B|

        参数:
            set1: 集合 1
            set2: 集合 2

        返回:
            相似度，取值范围 [0, 1]
        """
        if not set1 and not set2:
            return 1.0  # 两个空集完全相同

        if not set1 or not set2:
            return 0.0  # 一个空一个非空，完全不同

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def _should_keep_first(self, seg1: Segment, seg2: Segment) -> bool:
        """
        判断是否应该保留第一个 Segment（删除第二个）。

        决策优先级:
        1. 优先级高的保留
        2. 优先级相同时，检索分数高的保留（RAG 场景）
        3. 都相同时，保留第一个（稳定排序）

        参数:
            seg1: Segment 1
            seg2: Segment 2

        返回:
            True 保留 seg1，False 保留 seg2
        """
        # 比较优先级
        # Priority 枚举顺序：CRITICAL > HIGH > MEDIUM > LOW
        priority_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        p1 = priority_order.get(seg1.effective_priority.value, 0)
        p2 = priority_order.get(seg2.effective_priority.value, 0)

        if p1 != p2:
            return p1 > p2

        # 优先级相同，比较检索分数
        score1 = (
            seg1.provenance.retrieval_score if seg1.provenance else 0.0
        ) or 0.0
        score2 = (
            seg2.provenance.retrieval_score if seg2.provenance else 0.0
        ) or 0.0

        if score1 != score2:
            return score1 > score2

        # 都相同，保留第一个（稳定排序）
        return True
