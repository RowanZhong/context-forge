"""
Token Budget Manager 模块 — 三层预算模型的完整实现。

→ 6.2.2 预算分配策略（Budgeting）

Budget Manager 是 Context Forge 的核心决策引擎，负责在有限的 Token 窗口中
做最优分配决策。它实现了三层预算模型，灵感来自国家财政预算体系：

1. **刚性支出**（RigidStrategy → 6.2.2.1）：
   类比"公务员工资"，不可压缩，必须全额保障。
   适用于 System Prompt、Schema 定义、必要的工具声明。

2. **弹性区间**（ElasticStrategy → 6.2.2.2）：
   类比"项目拨款"，按需竞争配额。
   适用于 RAG 片段、对话历史、Few-Shot 示例。

3. **Output 预留**（ReserveStrategy → 6.2.2.4）：
   类比"应急储备金"，为模型输出和推理留空间。
   CoT 推理、Tool Call 生成、结构化输出。

模块结构：

- **manager.py**: BudgetManager 编排器（对外唯一入口）
- **strategies.py**: 三种分配策略的独立实现
- **bidding.py**: 弹性区间竞价算法（跨类型预算竞争）

基本用法::

    from context_forge.budget import BudgetManager
    from context_forge.models.budget import BudgetPolicy

    # 创建预算策略
    policy = BudgetPolicy(
        max_context_tokens=128_000,
        output_reserved_tokens=4_096,
        thinking_reserved_tokens=8_192,
    )

    # 创建 BudgetManager
    manager = BudgetManager(policy=policy)

    # 执行预算分配
    result = manager.allocate(segments=all_segments)

    # 检查结果
    print(f"保留 {len(result.kept_segments)} 个 Segment")
    print(f"窗口饱和度: {result.allocation.saturation_rate:.1%}")
    print(result.allocation.summary())

高级用法（自定义竞价权重）::

    manager = BudgetManager(
        policy=policy,
        priority_weight=1.5,   # 更重视优先级
        relevance_weight=0.3,  # 降低相关性权重
        quota_weight=0.5,      # 提高配额平衡权重
    )

# [Design Decision] Budget Manager 作为独立模块而非 Pipeline 的一部分，因为：
# - 它不仅用于 Pipeline 的 Allocate 阶段，也可单独用于预算规划和容量评估
# - 独立模块便于单元测试和性能基准测试
# - 未来可支持离线分析（如"如果切换到 32K 窗口，能容纳多少 RAG 片段？"）
"""

from context_forge.budget.bidding import BidScore, compute_bid_scores, greedy_allocate
from context_forge.budget.manager import BudgetManager, BudgetResult
from context_forge.budget.strategies import (
    AllocationResult,
    ElasticStrategy,
    ReserveStrategy,
    RigidStrategy,
)

__all__ = [
    # 主入口
    "BudgetManager",
    "BudgetResult",
    # 三种策略
    "RigidStrategy",
    "ElasticStrategy",
    "ReserveStrategy",
    "AllocationResult",
    # 竞价算法
    "BidScore",
    "compute_bid_scores",
    "greedy_allocate",
]
