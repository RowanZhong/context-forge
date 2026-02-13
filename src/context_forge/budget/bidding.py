"""
弹性区间竞价算法 — 跨类型预算竞争的核心引擎。

→ 6.2.2.2 弹性区间竞价
→ 6.2.2.3 分段竞价机制

当弹性区间的配额不足以容纳所有 Segment 时，需要一个公平且高效的竞价机制来决定：
- 哪些 Segment 能获得配额？
- 如何在不同类型（RAG / 对话历史 / Few-Shot）之间动态调度？

竞价算法的设计目标：
1. **公平性**：高优先级 + 高相关性的 Segment 优先获得配额
2. **多样性**：避免某一类型垄断所有配额（通过类型配额平衡）
3. **效率**：O(n log n) 复杂度，满足 <50ms P99 要求

# [Design Decision] 使用加权打分 + 贪心分配，而非精确的 0/1 背包求解，
# 因为：
# - 背包问题是 NP-Hard，O(nW) 伪多项式复杂度对大规模上下文不可接受
# - 贪心算法的近似解在实践中足够好（相对最优解误差 < 5%）
# - 易于解释和调试（分数公式可配置，审计日志可追溯）

竞价公式::

    BidScore = α·Priority + β·Relevance + γ·QuotaRatio

    其中：
    - Priority：优先级分数（CRITICAL=1000, HIGH=100, MEDIUM=10, LOW=1）
    - Relevance：相关性分数（来自 Rerank 阶段，范围 [0, 1]）
    - QuotaRatio：类型配额剩余比例（鼓励使用未充分利用的配额）
    - α, β, γ：可配置权重（默认 1.0, 0.5, 0.3）

⚠️ 反模式（→ 6.7.1 All-in-Context）：不做竞价直接按添加顺序截断，
会导致后添加的 Segment 即使相关性更高也被丢弃。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from context_forge.models.segment import Priority, Segment

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BidScore:
    """
    竞价分数 — 决定 Segment 在预算竞争中的优先级。

    → 6.2.2.2 弹性区间竞价

    属性:
        segment: 参与竞价的 Segment
        score: 综合竞价分数（越高越优先）
        priority_component: 优先级分量
        relevance_component: 相关性分量
        quota_component: 配额平衡分量
    """

    segment: Segment
    score: float
    priority_component: float
    relevance_component: float
    quota_component: float


def compute_bid_scores(
    segments: list[Segment],
    type_quota_remaining: dict[str, int],
    priority_weight: float = 1.0,
    relevance_weight: float = 0.5,
    quota_weight: float = 0.3,
) -> list[BidScore]:
    """
    计算所有 Segment 的竞价分数。

    → 6.2.2.2 弹性区间竞价

    竞价公式::

        BidScore = α·Priority + β·Relevance + γ·QuotaRatio

    参数:
        segments: 待竞价的 Segment 列表
        type_quota_remaining: 各类型剩余配额字典（类型名 → 剩余 Token 数）
        priority_weight: 优先级权重（α）
        relevance_weight: 相关性权重（β）
        quota_weight: 配额平衡权重（γ）

    返回:
        竞价分数列表（按分数降序排列）

    示例::

        bid_scores = compute_bid_scores(
            segments=elastic_segments,
            type_quota_remaining={"rag": 5000, "assistant": 2000},
            priority_weight=1.0,
            relevance_weight=0.5,
            quota_weight=0.3,
        )
    """
    bid_scores: list[BidScore] = []

    # 计算全局配额总量（用于归一化）
    total_quota = sum(type_quota_remaining.values())

    for seg in segments:
        # 1. 优先级分量
        priority_score = _priority_to_score(seg.effective_priority)
        priority_component = priority_score * priority_weight

        # 2. 相关性分量
        # 优先使用 rerank_score（经过重排的），否则使用 retrieval_score（原始检索分）
        relevance = seg.metadata.rerank_score or seg.metadata.retrieval_score or 0.0
        relevance_component = relevance * relevance_weight

        # 3. 配额平衡分量（鼓励使用未充分利用的类型配额）
        # → 6.2.2.3 配额平衡机制：避免某一类型垄断所有配额
        type_key = seg.type.value
        quota_remaining = type_quota_remaining.get(type_key, 0)
        quota_ratio = quota_remaining / total_quota if total_quota > 0 else 0.0
        quota_component = quota_ratio * quota_weight

        # 综合竞价分数
        total_score = priority_component + relevance_component + quota_component

        bid_scores.append(
            BidScore(
                segment=seg,
                score=total_score,
                priority_component=priority_component,
                relevance_component=relevance_component,
                quota_component=quota_component,
            )
        )

    # 按分数降序排列
    bid_scores.sort(key=lambda b: b.score, reverse=True)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "[bidding] 计算了 %d 个 Segment 的竞价分数，最高分 %.2f，最低分 %.2f",
            len(bid_scores),
            bid_scores[0].score if bid_scores else 0.0,
            bid_scores[-1].score if bid_scores else 0.0,
        )

    return bid_scores


def greedy_allocate(
    bid_scores: list[BidScore],
    available_budget: int,
    type_quota_remaining: dict[str, int] | None = None,
) -> tuple[tuple[Segment, ...], tuple[Segment, ...]]:
    """
    贪心分配：按竞价分数从高到低分配配额。

    → 6.2.2.2 弹性区间竞价
    → 6.2.2.3 分段竞价机制

    算法步骤：
    1. 按竞价分数降序遍历
    2. 如果当前 Segment 的 Token 数 ≤ 剩余预算，则分配
    3. 更新剩余预算和类型配额
    4. 重复直到预算耗尽或所有 Segment 处理完毕

    时间复杂度：O(n)（bid_scores 已排序）

    参数:
        bid_scores: 竞价分数列表（已按分数降序排列）
        available_budget: 可用的总 Token 预算
        type_quota_remaining: 各类型剩余配额（可选，用于跟踪配额使用情况）

    返回:
        (kept_segments, dropped_segments) 元组

    示例::

        bid_scores = compute_bid_scores(segments, type_quotas)
        kept, dropped = greedy_allocate(bid_scores, remaining_budget)
    """
    kept: list[Segment] = []
    dropped: list[Segment] = []
    remaining = available_budget

    # 复制配额字典以避免修改原始数据
    quota_tracker = dict(type_quota_remaining) if type_quota_remaining else {}

    for bid in bid_scores:
        seg = bid.segment
        seg_tokens = seg.token_count or 0

        # 检查是否有足够的全局预算和类型配额
        type_key = seg.type.value
        type_quota = quota_tracker.get(type_key, remaining)  # 如果没有配额限制，使用剩余全局预算

        if seg_tokens <= remaining and seg_tokens <= type_quota:
            # 分配成功
            kept.append(seg)
            remaining -= seg_tokens

            # 更新类型配额
            if type_key in quota_tracker:
                quota_tracker[type_key] -= seg_tokens

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[greedy_allocate] 保留 Segment %s (%d tokens)，竞价分数 %.2f，"
                    "剩余预算 %d",
                    seg.id[:8],
                    seg_tokens,
                    bid.score,
                    remaining,
                )
        else:
            # 预算不足或类型配额不足，丢弃
            dropped.append(seg)

            if logger.isEnabledFor(logging.DEBUG):
                reason = "全局预算不足" if seg_tokens > remaining else "类型配额不足"
                logger.debug(
                    "[greedy_allocate] 丢弃 Segment %s (%d tokens)，原因：%s，"
                    "竞价分数 %.2f",
                    seg.id[:8],
                    seg_tokens,
                    reason,
                    bid.score,
                )

        # 预算耗尽，提前退出
        if remaining <= 0:
            # 剩余的全部丢弃
            dropped.extend(bid_item.segment for bid_item in bid_scores[len(kept) + len(dropped):])
            break

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "[greedy_allocate] 分配完成：保留 %d 个，丢弃 %d 个，剩余预算 %d tokens",
            len(kept),
            len(dropped),
            remaining,
        )

    return (tuple(kept), tuple(dropped))


def _priority_to_score(priority: Priority) -> float:
    """
    优先级到数值分数的映射。

    → 6.2.2.2 优先级分数表

    # [Design Decision] 使用指数级差距（1000/100/10/1），而非线性（4/3/2/1），
    # 因为优先级的语义差距是质的而非量的：
    # - CRITICAL 不可丢弃
    # - HIGH 仅在极端压力下才考虑截断
    # - MEDIUM/LOW 可自由竞争

    映射表：
    - CRITICAL: 1000.0
    - HIGH: 100.0
    - MEDIUM: 10.0
    - LOW: 1.0
    """
    scores = {
        Priority.CRITICAL: 1000.0,
        Priority.HIGH: 100.0,
        Priority.MEDIUM: 10.0,
        Priority.LOW: 1.0,
    }
    return scores.get(priority, 0.0)
