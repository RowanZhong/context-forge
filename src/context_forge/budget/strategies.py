"""
预算分配策略 — 三种分配模式的独立实现。

→ 6.2.2 预算分配策略（Budgeting）

Budget Manager 的核心是三层预算分配模型，每一层对应一种分配策略：

1. **刚性支出锁定**（RigidStrategy → 6.2.2.1）：
   - 类比"公务员工资"，不可压缩，必须全额保障
   - 适用于 System Prompt、Schema 定义、必要的工具声明
   - 优先级：CRITICAL 或显式标记为 rigid 的类型

2. **弹性区间竞价**（ElasticStrategy → 6.2.2.2）：
   - 类比"项目拨款"，按需竞争配额
   - 适用于 RAG 片段、对话历史、Few-Shot 示例
   - 两阶段分配：类型配额 → 优先级排序 → 配额回收 → 溢出竞争

3. **Output 预留**（ReserveStrategy → 6.2.2.4）：
   - 类比"应急储备金"，为模型输出和推理留空间
   - CoT 推理、Tool Call 生成、结构化输出

# [Design Decision] 分离三种策略的实现，而非写在一个大函数里，
# 因为它们的分配逻辑完全不同：
# - Rigid 是全收，不分配
# - Elastic 是竞价，需要复杂的排序和配额管理
# - Reserve 是预留，不参与分配

⚠️ 反模式（→ 6.7.1 All-in-Context）：不做预算分配，直接把所有检索结果塞入上下文。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from context_forge.models.budget import BudgetPolicy
from context_forge.models.segment import Priority, Segment, SegmentType

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AllocationResult:
    """
    单阶段的预算分配结果。

    → 6.2.2 预算分配策略

    # [Design Decision] 使用 tuple 而非 list 存储结果，确保不可变性。
    # 分配结果一旦生成就不应该再被修改，这避免了隐式的状态耦合。

    属性:
        kept_segments: 保留的 Segment（已分配配额）
        dropped_segments: 被丢弃的 Segment（配额不足）
        tokens_used: 本阶段实际使用的 Token 数
        tokens_allocated: 本阶段可用的 Token 配额
        overflow_count: 溢出次数（被丢弃或截断的 Segment 数量）
    """

    kept_segments: tuple[Segment, ...]
    dropped_segments: tuple[Segment, ...]
    tokens_used: int
    tokens_allocated: int
    overflow_count: int


class RigidStrategy:
    """
    刚性支出锁定策略。

    → 6.2.2.1 刚性支出

    刚性 Segment 具有以下特征：
    - 优先级为 CRITICAL
    - 或者类型在 policy.rigid_segment_types 中（通常是 SYSTEM / SCHEMA）
    - 在预算分配中全额保障，不参与竞价

    用法::

        strategy = RigidStrategy()
        result = strategy.allocate(
            segments=all_segments,
            available_tokens=100000,
            policy=budget_policy,
        )
    """

    def allocate(
        self,
        segments: list[Segment],
        available_tokens: int,
        policy: BudgetPolicy,
    ) -> AllocationResult:
        """
        执行刚性支出分配。

        刚性 Segment 全部保留，不考虑预算限制（但会记录溢出）。

        参数:
            segments: 待分配的 Segment 列表
            available_tokens: 可用的 Token 总量
            policy: 预算策略配置

        返回:
            分配结果（刚性 Segment 全部保留在 kept_segments 中）
        """
        # [Design Decision] 刚性支出不做预算检查，全部保留。
        # 如果刚性支出已超出预算，这是配置问题，应在后续阶段发出警告。
        rigid_segments = [
            seg
            for seg in segments
            if self._is_rigid(seg, policy)
        ]

        tokens_used = sum(seg.token_count or 0 for seg in rigid_segments)
        overflow = 1 if tokens_used > available_tokens else 0

        if overflow and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[RigidStrategy] 刚性支出 %d tokens 超出可用预算 %d tokens",
                tokens_used,
                available_tokens,
            )

        return AllocationResult(
            kept_segments=tuple(rigid_segments),
            dropped_segments=(),
            tokens_used=tokens_used,
            tokens_allocated=available_tokens,
            overflow_count=overflow,
        )

    def _is_rigid(self, segment: Segment, policy: BudgetPolicy) -> bool:
        """判断 Segment 是否为刚性支出。"""
        return (
            segment.effective_priority == Priority.CRITICAL
            or segment.type in policy.rigid_segment_types
        )


class ElasticStrategy:
    """
    弹性区间竞价策略。

    → 6.2.2.2 弹性区间竞价

    弹性分配是最复杂的策略，包含四个步骤：

    1. **类型配额计算**：根据 policy.elastic_ratios 为每种类型分配基础配额
    2. **优先级排序**：在类型内按优先级 + 相关性分数排序
    3. **配额回收**：未用完的配额返还到公共池
    4. **溢出竞争**：未获得配额的 Segment 参与全局竞价（使用 bidding.py）

    # [Design Decision] 两阶段分配而非一次性竞价，因为：
    # - 第一阶段保障各类型的最小配额（防止某一类型被完全挤压）
    # - 第二阶段全局竞价，实现跨类型的动态调度

    用法::

        strategy = ElasticStrategy()
        result = strategy.allocate(
            segments=elastic_segments,
            available_tokens=50000,
            policy=budget_policy,
        )
    """

    def allocate(
        self,
        segments: list[Segment],
        available_tokens: int,
        policy: BudgetPolicy,
    ) -> AllocationResult:
        """
        执行弹性区间竞价分配。

        参数:
            segments: 待分配的弹性 Segment 列表
            available_tokens: 弹性区间的可用 Token 总量
            policy: 预算策略配置

        返回:
            分配结果（保留的和被丢弃的 Segment）
        """
        if not segments:
            return AllocationResult(
                kept_segments=(),
                dropped_segments=(),
                tokens_used=0,
                tokens_allocated=available_tokens,
                overflow_count=0,
            )

        # 第一阶段：按类型分配基础配额
        # → 6.2.2.2 类型配额分配
        type_groups = self._group_by_type(segments)
        type_quotas = self._compute_type_quotas(type_groups, available_tokens, policy)

        kept: list[Segment] = []
        dropped: list[Segment] = []
        tokens_used = 0
        overflow_count = 0

        # 第二阶段：在每个类型内按优先级分配
        for seg_type, group in type_groups.items():
            quota = type_quotas.get(seg_type, 0)

            # 按优先级 + 相关性分数排序（高优先级优先）
            sorted_group = sorted(
                group,
                key=lambda s: (
                    self._priority_score(s.effective_priority),
                    s.metadata.rerank_score or s.metadata.retrieval_score or 0.0,
                ),
                reverse=True,
            )

            type_used = 0
            for seg in sorted_group:
                seg_tokens = seg.token_count or 0

                if type_used + seg_tokens <= quota:
                    # 在配额内，保留
                    kept.append(seg)
                    type_used += seg_tokens
                else:
                    # 超出类型配额
                    remaining = quota - type_used

                    # 尝试截断以适应剩余配额
                    if remaining >= policy.min_elastic_tokens and seg.control.compressible:
                        # 🏭 生产提示：这里应调用真实的截断函数（如 TiktokenCounter.truncate_to_tokens）
                        # MVP 中简化为按比例截断
                        truncated_content = self._truncate_simple(seg.content, remaining, seg_tokens)
                        truncated_seg = seg.with_content(
                            truncated_content + "...[已截断]"
                        ).with_token_count(remaining)

                        kept.append(truncated_seg)
                        type_used += remaining
                        overflow_count += 1
                    else:
                        # 无法截断，丢弃
                        dropped.append(seg)
                        overflow_count += 1

            tokens_used += type_used

        # 第三阶段：配额回收（未使用的配额返还公共池）
        # → 6.2.2.2 配额回收机制
        remaining_budget = available_tokens - tokens_used

        if remaining_budget > 0 and dropped:
            # 第四阶段：溢出竞争（跨类型全局竞价）
            # 🏭 生产提示：这里应调用 bidding.py 的 greedy_allocate 进行精确竞价
            # MVP 中简化为按优先级顺序分配剩余配额
            dropped_sorted = sorted(
                dropped,
                key=lambda s: (
                    self._priority_score(s.effective_priority),
                    s.metadata.rerank_score or s.metadata.retrieval_score or 0.0,
                ),
                reverse=True,
            )

            rescued: list[Segment] = []
            rescued_dropped: list[Segment] = []

            for seg in dropped_sorted:
                seg_tokens = seg.token_count or 0
                if seg_tokens <= remaining_budget:
                    rescued.append(seg)
                    remaining_budget -= seg_tokens
                    tokens_used += seg_tokens
                    overflow_count -= 1  # 救回来了，不算溢出
                else:
                    rescued_dropped.append(seg)

            kept.extend(rescued)
            dropped = rescued_dropped

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[ElasticStrategy] 分配完成：保留 %d 个 Segment (%d tokens)，"
                "丢弃 %d 个，溢出 %d 次",
                len(kept),
                tokens_used,
                len(dropped),
                overflow_count,
            )

        return AllocationResult(
            kept_segments=tuple(kept),
            dropped_segments=tuple(dropped),
            tokens_used=tokens_used,
            tokens_allocated=available_tokens,
            overflow_count=overflow_count,
        )

    def _group_by_type(self, segments: list[Segment]) -> dict[SegmentType, list[Segment]]:
        """按类型分组 Segment。"""
        groups: dict[SegmentType, list[Segment]] = {}
        for seg in segments:
            groups.setdefault(seg.type, []).append(seg)
        return groups

    def _compute_type_quotas(
        self,
        type_groups: dict[SegmentType, list[Segment]],
        available: int,
        policy: BudgetPolicy,
    ) -> dict[SegmentType, int]:
        """
        计算每种类型的 Token 配额。

        → 6.2.2.2 类型配额计算

        根据 policy.elastic_ratios 分配基础配额，确保每种类型至少有 min_elastic_tokens。
        """
        quotas: dict[SegmentType, int] = {}

        for seg_type in type_groups:
            ratio = policy.elastic_ratios.get(seg_type, 0.0)
            quota = int(available * ratio)
            # 保障最小配额
            quota = max(quota, policy.min_elastic_tokens if ratio > 0 else 0)
            quotas[seg_type] = quota

        return quotas

    def _priority_score(self, priority: Priority) -> int:
        """优先级到数值分数的映射。"""
        scores = {
            Priority.CRITICAL: 1000,
            Priority.HIGH: 100,
            Priority.MEDIUM: 10,
            Priority.LOW: 1,
        }
        return scores.get(priority, 0)

    def _truncate_simple(self, content: str, target_tokens: int, original_tokens: int) -> str:
        """
        简化版截断：按比例裁剪字符。

        # 🏭 生产提示：应使用 tokenizer.truncate_to_tokens() 进行精确截断。
        """
        if original_tokens <= 0:
            return content
        ratio = target_tokens / original_tokens
        target_chars = int(len(content) * ratio)
        return content[:target_chars]


class ReserveStrategy:
    """
    预留缓冲策略。

    → 6.2.2.4 Output 预留
    → 6.2.2.5 Thinking Token 管理

    预留策略不参与分配，而是从总预算中扣除固定的 Token 数：
    - output_reserved_tokens：为模型输出预留（Tool Call / 结构化输出）
    - thinking_reserved_tokens：为 Reasoning Models 的隐式推理预留（o1/o3/R1）

    # [Design Decision] 单独预留 Thinking Token，因为 Reasoning Models
    # 的隐式推理消耗大量 Token 但不可见于输出。如果不预留，
    # 模型可能因为 Token 不足而被迫提前结束推理，导致回答质量下降。

    用法::

        strategy = ReserveStrategy()
        content_available = strategy.calculate_available(policy)
    """

    def calculate_available(self, policy: BudgetPolicy) -> int:
        """
        计算扣除预留后可用于内容的 Token 总量。

        参数:
            policy: 预算策略配置

        返回:
            可用于内容的 Token 数（总预算 - Output 预留 - Thinking 预留）
        """
        available = (
            policy.max_context_tokens
            - policy.output_reserved_tokens
            - policy.thinking_reserved_tokens
        )
        return max(0, available)

    def get_reserved_tokens(self, policy: BudgetPolicy) -> tuple[int, int]:
        """
        获取预留的 Token 数量。

        返回:
            (output_reserved, thinking_reserved) 元组
        """
        return (policy.output_reserved_tokens, policy.thinking_reserved_tokens)
