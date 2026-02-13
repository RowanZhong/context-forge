"""
Rerank 阶段 — 选择与重排。

→ 6.1.2.1 Pipeline 第三阶段
→ 6.3.2 Select：选择性注入与架构决策

决定哪些 Segment 应该进入最终上下文，以及它们的排列顺序。

本阶段执行：
1. 按优先级和相关性分数排序
2. 过滤过期的 Segment（TTL 检查）
3. 过滤不可见的 Segment（命名空间和可见性检查）
4. 去除重复内容（基于内容哈希）
5. MMR 多样性过滤（可选，→ 6.3.2.3）
6. 时效性加权（可选，→ 6.3.2.4）
"""

from __future__ import annotations

import hashlib
import logging
import math

from context_forge.models.audit import AuditEntry, DecisionType, ReasonCode
from context_forge.models.segment import Priority, Segment, SegmentType
from context_forge.pipeline.base import PipelineContext

logger = logging.getLogger(__name__)

# 优先级到数值的映射（数值越高，排名越靠前）
_PRIORITY_SCORE: dict[Priority, int] = {
    Priority.CRITICAL: 1000,
    Priority.HIGH: 100,
    Priority.MEDIUM: 10,
    Priority.LOW: 1,
}


def _content_hash(content: str) -> str:
    """计算内容的短哈希（用于去重）。"""
    return hashlib.md5(content.encode()).hexdigest()[:16]


def _ngram_jaccard(text1: str, text2: str, n: int = 3) -> float:
    """
    计算两个文本的 n-gram Jaccard 相似度。

    → 6.3.2.3 MMR 多样性过滤

    使用字符级 n-gram（默认 3-gram）计算相似度，
    适用于中英文混合文本，无需分词。

    参数:
        text1: 第一个文本
        text2: 第二个文本
        n: n-gram 大小（默认 3）

    返回:
        Jaccard 相似度（0.0 ~ 1.0）

    # [Design Decision] 使用字符 n-gram 而非词级 n-gram，因为：
    # - 中文分词成本高且不稳定
    # - 字符级对中英文都有效
    # - 3-gram 已足够捕捉局部相似性
    """
    if not text1 or not text2:
        return 0.0

    # 生成 n-gram 集合
    def ngrams(text: str) -> set[str]:
        if len(text) < n:
            return {text}
        return {text[i:i+n] for i in range(len(text) - n + 1)}

    ngrams1 = ngrams(text1)
    ngrams2 = ngrams(text2)

    if not ngrams1 or not ngrams2:
        return 0.0

    # Jaccard 相似度 = 交集 / 并集
    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)

    return intersection / union if union > 0 else 0.0


class RerankStage:
    """
    选择与重排阶段。

    → 6.3.2 Select

    按优先级和相关性排序 Segment，过滤过期和重复内容。
    CRITICAL 优先级的 Segment 总是排在最前面。

    第二轮增强：
    - MMR 多样性过滤（避免语义重复的 RAG 片段）
    - 时效性加权（优先最近的对话）
    - 每类型数量限制
    """

    def __init__(
        self,
        enable_mmr: bool = False,
        mmr_lambda: float = 0.7,
        similarity_threshold: float = 0.85,
        max_per_type: int = 0,
        enable_temporal_weighting: bool = False,
        temporal_decay_rate: float = 0.1,
        temporal_min_weight: float = 0.3,
    ) -> None:
        """
        初始化 Rerank 阶段。

        参数:
            enable_mmr: 是否启用 MMR 多样性过滤
            mmr_lambda: MMR 权衡参数（0=仅多样性，1=仅相关性）
            similarity_threshold: 相似度阈值（超过此值视为重复）
            max_per_type: 每种类型最大 Segment 数（0=无限制）
            enable_temporal_weighting: 是否启用时效性加权
            temporal_decay_rate: 时效性衰减率（越大衰减越快）
            temporal_min_weight: 时效性最小权重
        """
        self.enable_mmr = enable_mmr
        self.mmr_lambda = mmr_lambda
        self.similarity_threshold = similarity_threshold
        self.max_per_type = max_per_type
        self.enable_temporal_weighting = enable_temporal_weighting
        self.temporal_decay_rate = temporal_decay_rate
        self.temporal_min_weight = temporal_min_weight

    @property
    def name(self) -> str:
        return "rerank"

    async def process(
        self,
        segments: list[Segment],
        context: PipelineContext,
    ) -> list[Segment]:
        """选择并重排 Segment。"""
        result: list[Segment] = []
        seen_hashes: set[str] = set()

        for seg in segments:
            # 1. TTL 过期检查（→ 6.1.1.3 Control Flags）
            if seg.control.ttl is not None:
                if seg.control.is_expired(context.current_turn, seg.metadata.turn_number or 0):
                    context.audit_log.append(AuditEntry(
                        segment_id=seg.id,
                        decision=DecisionType.DROP,
                        reason_code=ReasonCode.SELECT_EXPIRED,
                        reason_detail=(
                            f"TTL 过期：设定 {seg.control.ttl} 轮，"
                            f"当前已过 {context.current_turn - (seg.metadata.turn_number or 0)} 轮。"
                        ),
                        pipeline_stage=self.name,
                    ))
                    continue

            # 2. 可见性检查（→ 6.3.4 Isolate 策略）
            if not seg.control.is_visible_to(context.target_namespace):
                context.audit_log.append(AuditEntry(
                    segment_id=seg.id,
                    decision=DecisionType.DROP,
                    reason_code=ReasonCode.SELECT_LOW_RELEVANCE,
                    reason_detail=(
                        f"命名空间不匹配：Segment 属于 '{seg.control.namespace}'，"
                        f"目标命名空间为 '{context.target_namespace}'。"
                    ),
                    pipeline_stage=self.name,
                ))
                continue

            # 3. 内容去重（基于哈希）
            content_h = _content_hash(seg.content)
            if content_h in seen_hashes:
                context.audit_log.append(AuditEntry(
                    segment_id=seg.id,
                    decision=DecisionType.DROP,
                    reason_code=ReasonCode.SELECT_DUPLICATE,
                    reason_detail="与已有 Segment 内容完全重复，已去重。",
                    pipeline_stage=self.name,
                ))
                continue
            seen_hashes.add(content_h)

            result.append(seg)

        # 4. 时效性加权（→ 6.3.2.4）
        if self.enable_temporal_weighting:
            result = self._apply_temporal_weighting(result, context)

        # 5. MMR 多样性过滤（→ 6.3.2.3）
        if self.enable_mmr and len(result) <= 500:  # 性能保护
            result = self._apply_mmr_filter(result, context)
        elif self.enable_mmr and len(result) > 500:
            logger.warning(
                "[rerank] Segment 数量 %d 超过 500，跳过 MMR 过滤（性能保护）。",
                len(result),
            )

        # 6. 每类型数量限制
        if self.max_per_type > 0:
            result = self._limit_per_type(result, context)

        # 7. 排序：CRITICAL 在前，锁定位置的 Segment 保持原位
        locked = [s for s in result if s.control.lock_position]
        unlocked = [s for s in result if not s.control.lock_position]

        # 对未锁定的 Segment 按优先级 + rerank 分数排序
        unlocked.sort(key=lambda s: (
            _PRIORITY_SCORE.get(s.effective_priority, 0),
            s.metadata.rerank_score or s.metadata.retrieval_score or 0.0,
        ), reverse=True)

        # 合并：锁定位置的保持原位，未锁定的按排序结果重新排列
        # [Design Decision] 简化策略：锁定的 Segment 放在最前面（通常是 System Prompt），
        # 未锁定的按分数排在后面。
        final = locked + unlocked

        if context.debug:
            logger.debug(
                "[rerank] %d → %d Segment（去重 %d，过期/不可见 %d）",
                len(segments),
                len(final),
                len(segments) - len(result),
                len(result) - len(final),
            )

        return final

    def _apply_temporal_weighting(
        self,
        segments: list[Segment],
        context: PipelineContext,
    ) -> list[Segment]:
        """
        应用时效性加权。

        → 6.3.2.4 时效性加权

        越新的 Segment 得分越高。使用指数衰减公式：
        weight = max(min_weight, exp(-decay_rate × age))

        其中 age = current_turn - segment.turn_number

        参数:
            segments: Segment 列表
            context: 流水线上下文

        返回:
            更新了 rerank_score 的 Segment 列表
        """
        current_turn = context.current_turn
        updated_segments = []

        for seg in segments:
            turn_number = seg.metadata.turn_number or 0
            age = current_turn - turn_number

            # 指数衰减权重
            temporal_weight = max(
                self.temporal_min_weight,
                math.exp(-self.temporal_decay_rate * age),
            )

            # 原始相关性分数
            base_score = seg.metadata.rerank_score or seg.metadata.retrieval_score or 0.5

            # 加权后的分数
            weighted_score = base_score * temporal_weight

            # 更新 Segment 的 rerank_score
            updated_meta = seg.metadata.model_copy(update={
                "rerank_score": weighted_score,
            })
            updated_seg = seg.model_copy(update={"metadata": updated_meta})
            updated_segments.append(updated_seg)

            if context.debug and abs(weighted_score - base_score) > 0.01:
                logger.debug(
                    "[rerank] Segment %s 时效性加权：%.3f → %.3f（age=%d）",
                    seg.id,
                    base_score,
                    weighted_score,
                    age,
                )

        return updated_segments

    def _apply_mmr_filter(
        self,
        segments: list[Segment],
        context: PipelineContext,
    ) -> list[Segment]:
        """
        应用 MMR（最大边际相关性）多样性过滤。

        → 6.3.2.3 MMR 多样性过滤

        目标：在保持相关性的同时增加多样性，避免重复的 RAG 片段。

        MMR 公式：
        MMR = λ × Relevance(S) - (1-λ) × max(Similarity(S, R))

        其中：
        - λ（lambda）：相关性 vs 多样性的权衡参数（0~1）
        - Relevance(S)：Segment 的原始相关性分数
        - Similarity(S, R)：Segment 与已选集合 R 中最相似片段的相似度

        参数:
            segments: Segment 列表
            context: 流水线上下文

        返回:
            多样性过滤后的 Segment 列表
        """
        # 分离锁定和未锁定的 Segment
        locked = [s for s in segments if s.control.lock_position]
        unlocked = [s for s in segments if not s.control.lock_position]

        if not unlocked:
            return locked

        # MMR 贪心选择
        selected: list[Segment] = []
        candidates = unlocked.copy()

        # 第一轮：选择相关性最高的 Segment
        candidates.sort(
            key=lambda s: s.metadata.rerank_score or s.metadata.retrieval_score or 0.0,
            reverse=True,
        )
        if candidates:
            selected.append(candidates.pop(0))

        # 后续轮次：按 MMR 分数贪心选择
        dropped_count = 0
        while candidates:
            best_seg = None
            best_mmr_score = -float('inf')

            for seg in candidates:
                # 计算相关性分数
                relevance = seg.metadata.rerank_score or seg.metadata.retrieval_score or 0.0

                # 计算与已选集合的最大相似度
                max_similarity = 0.0
                for selected_seg in selected:
                    similarity = _ngram_jaccard(seg.content, selected_seg.content)
                    max_similarity = max(max_similarity, similarity)

                # MMR 分数
                mmr_score = (
                    self.mmr_lambda * relevance
                    - (1 - self.mmr_lambda) * max_similarity
                )

                # 如果相似度超过阈值，直接跳过（视为重复）
                if max_similarity > self.similarity_threshold:
                    context.audit_log.append(AuditEntry(
                        segment_id=seg.id,
                        decision=DecisionType.DROP,
                        reason_code=ReasonCode.SELECT_DUPLICATE,
                        reason_detail=(
                            f"MMR 过滤：与已选 Segment 相似度 {max_similarity:.2f} "
                            f"超过阈值 {self.similarity_threshold:.2f}。"
                        ),
                        pipeline_stage=self.name,
                    ))
                    dropped_count += 1
                    continue

                # 记录最佳候选
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_seg = seg

            # 如果没有找到合适的候选，结束
            if best_seg is None:
                break

            # 选择 MMR 分数最高的 Segment
            selected.append(best_seg)
            candidates.remove(best_seg)

        if context.debug:
            logger.debug(
                "[rerank] MMR 过滤：%d → %d Segment（去重 %d）",
                len(unlocked),
                len(selected),
                dropped_count,
            )

        return locked + selected

    def _limit_per_type(
        self,
        segments: list[Segment],
        context: PipelineContext,
    ) -> list[Segment]:
        """
        限制每种类型的 Segment 数量。

        → 6.3.2 Select 策略

        防止某一类型的 Segment 占用过多空间（如 RAG 片段过多）。

        参数:
            segments: Segment 列表
            context: 流水线上下文

        返回:
            限制后的 Segment 列表
        """
        type_counts: dict[SegmentType, int] = {}
        result: list[Segment] = []

        for seg in segments:
            seg_type = seg.type
            current_count = type_counts.get(seg_type, 0)

            # 锁定位置的 Segment 不受限制
            if seg.control.lock_position:
                result.append(seg)
                type_counts[seg_type] = current_count + 1
                continue

            # 检查是否超过限制
            if current_count >= self.max_per_type:
                context.audit_log.append(AuditEntry(
                    segment_id=seg.id,
                    decision=DecisionType.DROP,
                    reason_code=ReasonCode.SELECT_LOW_RELEVANCE,
                    reason_detail=(
                        f"类型 {seg_type.value} 的 Segment 数量已达上限 "
                        f"{self.max_per_type}。"
                    ),
                    pipeline_stage=self.name,
                ))
                continue

            # 保留该 Segment
            result.append(seg)
            type_counts[seg_type] = current_count + 1

        if context.debug:
            dropped = len(segments) - len(result)
            if dropped > 0:
                logger.debug(
                    "[rerank] 类型限制：%d → %d Segment（丢弃 %d）",
                    len(segments),
                    len(result),
                    dropped,
                )

        return result
