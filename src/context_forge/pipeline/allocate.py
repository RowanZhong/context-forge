"""
Allocate 阶段 — 预算分配。

→ 6.1.2.1 Pipeline 第四阶段
→ 6.2.2 预算分配策略（Budgeting）

这是流水线中最关键的决策阶段。它解决的核心问题是：
给定一个有限的 Token 窗口，如何在不同优先级的 Segment 之间做最优分配？

第二轮增强：委托给独立的 BudgetManager 模块，支持三层预算模型：
1. **刚性支出锁定**：CRITICAL 优先级的 Segment 全额保障
2. **弹性竞价分配**：剩余空间按比例和分数竞价
3. **溢出处理**：预算不足时按优先级从低到高截断或丢弃

⚠️ 反模式（→ 6.7.1 All-in-Context）：不做预算分配就把所有内容塞入上下文，
是最常见的工程反模式。窗口虽大（128K），但无关信息会稀释注意力密度，
导致关键信息在"大海捞针"中被遗漏。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from context_forge.models.segment import Segment
from context_forge.pipeline.base import PipelineContext

if TYPE_CHECKING:
    from context_forge.budget.manager import BudgetManager

logger = logging.getLogger(__name__)


class AllocateStage:
    """
    预算分配阶段。

    → 6.2.2 预算分配策略

    第二轮增强版：集成 BudgetManager，委托给独立的预算管理模块。
    同时保持 PipelineStage Protocol 接口不变，确保向后兼容。

    高级用法（自定义 BudgetManager）::

        from context_forge.budget.manager import BudgetManager

        manager = BudgetManager(
            policy=budget_policy,
            priority_weight=1.5,  # 更重视优先级
            relevance_weight=0.3,  # 降低相关性权重
        )
        stage = AllocateStage(budget_manager=manager)

    默认用法（自动创建 BudgetManager）::

        # 不传 budget_manager，内部自动根据 context.budget_policy 创建
        stage = AllocateStage()
    """

    def __init__(
        self,
        budget_manager: BudgetManager | None = None,
    ) -> None:
        """
        初始化 Allocate 阶段。

        参数:
            budget_manager: 自定义 BudgetManager 实例（高级用法）
        """
        self._budget_manager = budget_manager

    @property
    def name(self) -> str:
        return "allocate"

    async def process(
        self,
        segments: list[Segment],
        context: PipelineContext,
    ) -> list[Segment]:
        """
        执行预算分配，丢弃或截断超出预算的 Segment。

        → 6.2.2 预算分配策略

        流程：
        1. 确保所有 Segment 都有 token_count
        2. 创建或使用 BudgetManager
        3. 调用 BudgetManager.allocate() 执行三层预算分配
        4. 合并审计日志和警告
        5. 返回最终保留的 Segment 列表
        """
        from context_forge.budget.manager import BudgetManager
        from context_forge.tokenizer.registry import get_tokenizer

        policy = context.budget_policy
        counter = get_tokenizer(context.model)

        # 第一步：确保所有 Segment 都有 token_count
        counted_segments: list[Segment] = []
        for seg in segments:
            if seg.token_count is None:
                count = counter.count(seg.content)
                seg = seg.with_token_count(count)
            counted_segments.append(seg)

        # 第二步：创建或使用 BudgetManager
        if self._budget_manager is not None:
            manager = self._budget_manager
        else:
            # 使用默认配置创建 BudgetManager
            manager = BudgetManager(policy=policy)

        # 第三步：执行预算分配
        # [DX Decision] 委托给 BudgetManager，保持 Pipeline 阶段简洁
        result = manager.allocate(segments=counted_segments)

        # 第四步：合并审计日志和警告
        context.audit_log.extend(result.audit_entries)
        context.warnings.extend(result.warnings)

        # 第五步：记录预算分配结果到 context.metadata
        context.metadata["budget_allocation"] = result.allocation

        # 调试日志
        if context.debug:
            logger.debug(
                "[allocate] 预算分配完成：刚性 %d + 弹性 %d = %d / %d tokens "
                "（饱和度 %.1f%%，溢出 %d 次）",
                result.allocation.rigid_used,
                sum(result.allocation.elastic_used.values()),
                result.allocation.total_used,
                result.allocation.content_budget,
                result.allocation.saturation_rate * 100,
                result.allocation.overflow_count,
            )

        # 返回最终保留的 Segment 列表
        return list(result.kept_segments)
