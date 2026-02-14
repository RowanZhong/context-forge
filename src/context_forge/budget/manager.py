"""
BudgetManager — Token 预算管理的统一编排器。

→ 6.2.2 预算分配策略（Budgeting）

BudgetManager 是 Context Forge 的核心决策引擎，负责在有限的 Token 窗口中
做最优分配决策。它编排三种分配策略（Rigid / Elastic / Reserve），
并产出完整的预算分配报告。

核心流程：

1. **预留计算**（ReserveStrategy）：
   - 从总预算中扣除 Output 预留和 Thinking 预留
   - 得到可用于内容的 Token 总量

2. **刚性锁定**（RigidStrategy）：
   - 识别 CRITICAL 优先级和显式标记为 rigid 的 Segment
   - 全额保障，不参与竞价

3. **弹性竞价**（ElasticStrategy + Bidding）：
   - 剩余 Segment 参与弹性区间竞价
   - 两阶段分配：类型配额 → 优先级排序 → 配额回收 → 溢出竞争

4. **结果组装**（BudgetResult）：
   - 合并 kept_segments（刚性 + 弹性）
   - 记录 BudgetAllocation（用于可观测性）
   - 生成警告（窗口饱和度、刚性溢出等）

# [Design Decision] 编排器模式而非单一大函数，因为：
# - 三种策略的逻辑独立，分离后易于测试和扩展
# - 未来可支持自定义策略（如"重要性采样"、"滑动窗口"）
# - 审计日志需要区分不同阶段的决策

⚠️ 反模式（→ 6.7.1 All-in-Context）：不做预算管理，直接塞入所有内容，
会导致窗口饱和度过高，模型注意力被无关信息稀释。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from context_forge.budget.strategies import ElasticStrategy, ReserveStrategy, RigidStrategy
from context_forge.errors.exceptions import BudgetExceededError
from context_forge.models.audit import AuditEntry, DecisionType, ReasonCode
from context_forge.models.budget import BudgetAllocation, BudgetPolicy

if TYPE_CHECKING:
    from context_forge.models.segment import Segment

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BudgetResult:
    """
    完整的预算分配结果。

    → 6.2.2 预算分配策略

    这是 BudgetManager.allocate() 的返回值，包含：
    - 最终保留的 Segment 列表（刚性 + 弹性）
    - 预算分配记录（BudgetAllocation）
    - 审计日志（每个 Segment 的分配决策）
    - 警告信息（窗口饱和度、刚性溢出等）

    属性:
        kept_segments: 最终保留的 Segment（tuple 确保不可变）
        allocation: 预算分配记录
        audit_entries: 审计日志（每个 Segment 的决策记录）
        warnings: 警告信息列表
    """

    kept_segments: tuple[Segment, ...]
    allocation: BudgetAllocation
    audit_entries: tuple[AuditEntry, ...]
    warnings: tuple[str, ...]


class BudgetManager:
    """
    Token 预算管理器 — 三层预算模型的编排器。

    → 6.2.2 预算分配策略

    用法::

        manager = BudgetManager(policy=budget_policy)
        result = manager.allocate(segments=all_segments)

        # 检查饱和度
        if result.allocation.saturation_rate > 0.9:
            print("警告：窗口饱和度过高，建议启用压缩策略")

        # 获取最终 Segment 列表
        final_segments = result.kept_segments

    高级用法（自定义竞价权重）::

        manager = BudgetManager(
            policy=budget_policy,
            priority_weight=1.5,  # 更重视优先级
            relevance_weight=0.3,  # 降低相关性权重
        )
    """

    def __init__(
        self,
        policy: BudgetPolicy,
        priority_weight: float = 1.0,
        relevance_weight: float = 0.5,
        quota_weight: float = 0.3,
    ) -> None:
        """
        初始化 BudgetManager。

        参数:
            policy: 预算策略配置
            priority_weight: 竞价公式中的优先级权重（α）
            relevance_weight: 竞价公式中的相关性权重（β）
            quota_weight: 竞价公式中的配额平衡权重（γ）
        """
        self.policy = policy
        self.priority_weight = priority_weight
        self.relevance_weight = relevance_weight
        self.quota_weight = quota_weight

        # 初始化三种策略
        self.reserve_strategy = ReserveStrategy()
        self.rigid_strategy = RigidStrategy()
        self.elastic_strategy = ElasticStrategy()

    def allocate(self, segments: list[Segment]) -> BudgetResult:
        """
        执行完整的预算分配流程。

        → 6.2.2 预算分配策略

        流程：
        1. 计算可用预算（总预算 - 预留）
        2. 刚性支出锁定
        3. 弹性区间竞价
        4. 组装结果 + 生成审计日志

        参数:
            segments: 待分配的所有 Segment（必须已填充 token_count）

        返回:
            完整的预算分配结果（BudgetResult）

        异常:
            BudgetExceededError: 当刚性支出已超出总预算且 overflow_strategy="error" 时
        """
        audit_entries: list[AuditEntry] = []
        warnings: list[str] = []

        # 第一步：计算可用预算
        # → 6.2.2.4 Output 预留 + 6.2.2.5 Thinking Token 管理
        content_available = self.reserve_strategy.calculate_available(self.policy)
        output_reserved, thinking_reserved = self.reserve_strategy.get_reserved_tokens(self.policy)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[BudgetManager] 总预算 %d tokens，Output 预留 %d，Thinking 预留 %d，"
                "内容可用 %d",
                self.policy.max_context_tokens,
                output_reserved,
                thinking_reserved,
                content_available,
            )

        # 第二步：刚性支出锁定
        # → 6.2.2.1 刚性支出
        rigid_result = self.rigid_strategy.allocate(
            segments=segments,
            available_tokens=content_available,
            policy=self.policy,
        )

        rigid_segments = list(rigid_result.kept_segments)
        rigid_total = rigid_result.tokens_used

        # 记录刚性支出的审计日志
        for seg in rigid_segments:
            audit_entries.append(
                AuditEntry(
                    segment_id=seg.id,
                    decision=DecisionType.KEEP,
                    reason_code=ReasonCode.RIGID_GUARANTEED,
                    reason_detail=f"刚性支出，全额保障 {seg.token_count or 0} tokens。",
                    pipeline_stage="budget_allocate",
                    token_impact=seg.token_count or 0,
                )
            )

        # 检查刚性溢出
        if rigid_total > content_available:
            msg = (
                f"刚性支出（{rigid_total:,} tokens）已超出内容预算"
                f"（{content_available:,} tokens）。"
                f"建议精简 System Prompt 或切换到更大窗口的模型。"
            )
            warnings.append(msg)

            # 根据溢出策略决定是否抛异常
            if self.policy.overflow_strategy == "error":
                raise BudgetExceededError(
                    what=msg,
                    why=f"刚性 Segment 共 {len(rigid_segments)} 个，"
                        f"包括 SYSTEM / SCHEMA / CRITICAL 优先级的内容。",
                    how="尝试：精简 System Prompt 的长度，或将部分内容移至 User Message；"
                        "或切换到窗口更大的模型（如 claude-opus-4-20250115 的 200K 窗口）。",
                    required_tokens=rigid_total,
                    budget_tokens=content_available,
                )

        # 第三步：弹性区间竞价
        # → 6.2.2.2 弹性区间竞价
        elastic_available = max(0, content_available - rigid_total)
        elastic_candidates = [
            seg
            for seg in segments
            if seg not in rigid_segments
        ]

        elastic_result = self.elastic_strategy.allocate(
            segments=elastic_candidates,
            available_tokens=elastic_available,
            policy=self.policy,
        )

        elastic_segments = list(elastic_result.kept_segments)
        elastic_dropped = list(elastic_result.dropped_segments)
        elastic_total = elastic_result.tokens_used

        # 记录弹性分配的审计日志
        for seg in elastic_segments:
            audit_entries.append(
                AuditEntry(
                    segment_id=seg.id,
                    decision=DecisionType.KEEP,
                    reason_code=ReasonCode.ELASTIC_ALLOCATED,
                    reason_detail=f"弹性竞价成功，分配 {seg.token_count or 0} tokens。",
                    pipeline_stage="budget_allocate",
                    token_impact=seg.token_count or 0,
                )
            )

        for seg in elastic_dropped:
            audit_entries.append(
                AuditEntry(
                    segment_id=seg.id,
                    decision=DecisionType.DROP,
                    reason_code=ReasonCode.BUDGET_EXCEEDED,
                    reason_detail=f"弹性竞价失败，预算不足以容纳 {seg.token_count or 0} tokens。",
                    pipeline_stage="budget_allocate",
                    token_impact=-(seg.token_count or 0),
                )
            )

        # 第四步：组装最终结果
        final_segments = rigid_segments + elastic_segments
        total_used = rigid_total + elastic_total

        # 统计弹性支出（按类型）
        elastic_by_type: dict[str, int] = {}
        for seg in elastic_segments:
            type_key = seg.type.value
            elastic_by_type[type_key] = elastic_by_type.get(type_key, 0) + (seg.token_count or 0)

        # 创建 BudgetAllocation 记录
        allocation = BudgetAllocation(
            total_budget=self.policy.max_context_tokens,
            content_budget=content_available,
            rigid_used=rigid_total,
            elastic_used=elastic_by_type,
            output_reserved=output_reserved,
            thinking_reserved=thinking_reserved,
            total_used=total_used,
            overflow_count=elastic_result.overflow_count + rigid_result.overflow_count,
        )

        # 窗口饱和度警告
        saturation = allocation.saturation_rate
        if saturation > self.policy.saturation_threshold:
            warnings.append(
                f"窗口饱和度 {saturation:.1%} 超过阈值 {self.policy.saturation_threshold:.1%}。"
                f"建议启用压缩策略或减少输入内容。"
            )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[BudgetManager] 分配完成：刚性 %d + 弹性 %d = %d / %d tokens "
                "（饱和度 %.1f%%，溢出 %d 次）",
                rigid_total,
                elastic_total,
                total_used,
                content_available,
                saturation * 100,
                allocation.overflow_count,
            )

        return BudgetResult(
            kept_segments=tuple(final_segments),
            allocation=allocation,
            audit_entries=tuple(audit_entries),
            warnings=tuple(warnings),
        )

    def validate_segments(self, segments: list[Segment]) -> None:
        """
        验证 Segment 列表是否满足预算分配的前提条件。

        检查项：
        - 所有 Segment 都有 token_count（非 None）
        - token_count 都是非负整数

        参数:
            segments: 待验证的 Segment 列表

        异常:
            ValueError: 当 Segment 不满足条件时
        """
        for seg in segments:
            if seg.token_count is None:
                raise ValueError(
                    f"Segment {seg.id} 缺少 token_count。"
                    f"请在调用 BudgetManager.allocate() 之前通过 Normalize 阶段填充 token_count。"
                )
            if seg.token_count < 0:
                raise ValueError(
                    f"Segment {seg.id} 的 token_count ({seg.token_count}) 为负数，这是非法的。"
                )
