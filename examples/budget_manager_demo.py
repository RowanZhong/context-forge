"""
Budget Manager 使用示例。

演示如何使用 BudgetManager 进行 Token 预算分配。

→ 6.2.2 预算分配策略（Budgeting）
"""

import asyncio
from context_forge.models.segment import Segment, SegmentType, Priority
from context_forge.models.budget import BudgetPolicy
from context_forge.budget import BudgetManager


def create_sample_segments():
    """创建示例 Segment 列表。"""
    return [
        # 刚性 Segment（CRITICAL 优先级）
        Segment(
            type=SegmentType.SYSTEM,
            content="你是一个专业的 Python 编程助手，精通异步编程和性能优化。",
            role="system",
            priority=Priority.CRITICAL,
            token_count=50,
        ),
        Segment(
            type=SegmentType.SCHEMA,
            content='{"type": "object", "properties": {"code": {"type": "string"}}}',
            role="system",
            priority=Priority.CRITICAL,
            token_count=30,
        ),
        # 高优先级 Segment
        Segment(
            type=SegmentType.USER,
            content="如何在 Python 中实现高性能的异步文件 I/O？",
            role="user",
            priority=Priority.HIGH,
            token_count=100,
        ),
        Segment(
            type=SegmentType.FEW_SHOT,
            content="示例：使用 aiofiles 库进行异步文件读写...",
            role="user",
            priority=Priority.HIGH,
            token_count=200,
        ),
        # 中优先级 Segment（RAG 检索结果）
        Segment(
            type=SegmentType.RAG,
            content="Python asyncio 文档摘录：asyncio 提供了一组高层级 API...",
            role="user",
            priority=Priority.MEDIUM,
            token_count=500,
        ),
        Segment(
            type=SegmentType.RAG,
            content="aiofiles 库使用指南：aiofiles 是一个用于异步文件操作的 Apache2 许可库...",
            role="user",
            priority=Priority.MEDIUM,
            token_count=400,
        ),
        Segment(
            type=SegmentType.RAG,
            content="性能优化最佳实践：在处理大量文件时，使用 asyncio.gather() 并发执行...",
            role="user",
            priority=Priority.MEDIUM,
            token_count=300,
        ),
        # 低优先级 Segment（历史对话）
        Segment(
            type=SegmentType.ASSISTANT,
            content="之前的回答：Python 中的异步编程主要使用 async/await 语法...",
            role="assistant",
            priority=Priority.LOW,
            token_count=600,
        ),
        Segment(
            type=SegmentType.ASSISTANT,
            content="更早的回答：关于性能优化，首先需要进行 profiling...",
            role="assistant",
            priority=Priority.LOW,
            token_count=400,
        ),
    ]


def demo_basic_allocation():
    """演示基本的预算分配。"""
    print("=" * 70)
    print("示例 1: 基本预算分配")
    print("=" * 70)

    # 创建预算策略（较小的窗口，会有溢出）
    policy = BudgetPolicy(
        max_context_tokens=2000,
        output_reserved_tokens=500,
        thinking_reserved_tokens=0,
        rigid_segment_types=[SegmentType.SYSTEM, SegmentType.SCHEMA],
        elastic_ratios={
            SegmentType.RAG: 0.4,
            SegmentType.USER: 0.2,
            SegmentType.ASSISTANT: 0.2,
            SegmentType.FEW_SHOT: 0.2,
        },
        min_elastic_tokens=100,
    )

    # 创建 BudgetManager
    manager = BudgetManager(policy=policy)

    # 执行预算分配
    segments = create_sample_segments()
    result = manager.allocate(segments)

    # 输出结果
    print(f"\n输入 Segment 数量: {len(segments)}")
    print(f"保留 Segment 数量: {len(result.kept_segments)}")
    print(f"总预算: {result.allocation.total_budget:,} tokens")
    print(f"内容预算: {result.allocation.content_budget:,} tokens")
    print(f"刚性支出: {result.allocation.rigid_used:,} tokens")
    print(f"弹性支出: {sum(result.allocation.elastic_used.values()):,} tokens")
    print(f"总使用: {result.allocation.total_used:,} tokens")
    print(f"饱和度: {result.allocation.saturation_rate:.1%}")
    print(f"溢出次数: {result.allocation.overflow_count}")

    print(f"\n弹性支出明细（按类型）:")
    for seg_type, used in sorted(result.allocation.elastic_used.items()):
        ratio = policy.elastic_ratios.get(SegmentType(seg_type), 0.0)
        print(f"  {seg_type:15s}: {used:5,} tokens (配额比例 {ratio:.0%})")

    print(f"\n警告信息:")
    for warning in result.warnings:
        print(f"  - {warning}")

    print(f"\n审计日志摘要:")
    keep_count = sum(1 for e in result.audit_entries if e.decision.value == "keep")
    drop_count = sum(1 for e in result.audit_entries if e.decision.value == "drop")
    print(f"  保留: {keep_count} 个 Segment")
    print(f"  丢弃: {drop_count} 个 Segment")


def demo_custom_weights():
    """演示自定义竞价权重。"""
    print("\n" + "=" * 70)
    print("示例 2: 自定义竞价权重（更重视优先级）")
    print("=" * 70)

    policy = BudgetPolicy(
        max_context_tokens=2000,
        output_reserved_tokens=500,
        elastic_ratios={
            SegmentType.RAG: 0.4,
            SegmentType.USER: 0.2,
            SegmentType.ASSISTANT: 0.2,
            SegmentType.FEW_SHOT: 0.2,
        },
    )

    # 创建 BudgetManager，提高优先级权重，降低相关性权重
    manager = BudgetManager(
        policy=policy,
        priority_weight=2.0,   # 更重视优先级（默认 1.0）
        relevance_weight=0.2,  # 降低相关性权重（默认 0.5）
        quota_weight=0.3,      # 保持配额平衡权重
    )

    segments = create_sample_segments()
    result = manager.allocate(segments)

    print(f"\n保留 Segment 数量: {len(result.kept_segments)}")
    print(f"饱和度: {result.allocation.saturation_rate:.1%}")

    print(f"\n保留的 Segment（按优先级排序）:")
    kept_by_priority = sorted(
        result.kept_segments,
        key=lambda s: (s.effective_priority.value, s.type.value),
    )
    for seg in kept_by_priority:
        print(
            f"  [{seg.effective_priority.value:8s}] {seg.type.value:15s} "
            f"({seg.token_count:3,} tokens)"
        )


def demo_large_window():
    """演示大窗口模型（无溢出）。"""
    print("\n" + "=" * 70)
    print("示例 3: 大窗口模型（128K，无溢出）")
    print("=" * 70)

    policy = BudgetPolicy(
        max_context_tokens=128_000,
        output_reserved_tokens=4_096,
        thinking_reserved_tokens=8_192,
        elastic_ratios={
            SegmentType.RAG: 0.4,
            SegmentType.USER: 0.2,
            SegmentType.ASSISTANT: 0.2,
            SegmentType.FEW_SHOT: 0.2,
        },
    )

    manager = BudgetManager(policy=policy)
    segments = create_sample_segments()
    result = manager.allocate(segments)

    print(f"\n总预算: {result.allocation.total_budget:,} tokens")
    print(f"内容预算: {result.allocation.content_budget:,} tokens")
    print(f"总使用: {result.allocation.total_used:,} tokens")
    print(f"剩余: {result.allocation.remaining:,} tokens")
    print(f"饱和度: {result.allocation.saturation_rate:.1%}")
    print(f"溢出次数: {result.allocation.overflow_count}")

    print(f"\n所有 {len(segments)} 个 Segment 都被保留（窗口足够大）")


def demo_reasoning_model():
    """演示 Reasoning Model（需要 Thinking Token 预留）。"""
    print("\n" + "=" * 70)
    print("示例 4: Reasoning Model（预留 Thinking Token）")
    print("=" * 70)

    # o1/o3/R1 等推理模型需要预留大量 Thinking Token
    policy = BudgetPolicy(
        max_context_tokens=128_000,
        output_reserved_tokens=4_096,
        thinking_reserved_tokens=32_000,  # 预留 32K 用于隐式推理
        elastic_ratios={
            SegmentType.RAG: 0.4,
            SegmentType.USER: 0.2,
            SegmentType.ASSISTANT: 0.2,
            SegmentType.FEW_SHOT: 0.2,
        },
    )

    manager = BudgetManager(policy=policy)
    segments = create_sample_segments()
    result = manager.allocate(segments)

    output_reserved, thinking_reserved = manager.reserve_strategy.get_reserved_tokens(policy)

    print(f"\n总预算: {result.allocation.total_budget:,} tokens")
    print(f"Output 预留: {output_reserved:,} tokens")
    print(f"Thinking 预留: {thinking_reserved:,} tokens")
    print(f"内容可用: {result.allocation.content_budget:,} tokens")
    print(f"内容使用: {result.allocation.total_used:,} tokens")
    print(f"饱和度: {result.allocation.saturation_rate:.1%}")

    print(f"\n预留了 {thinking_reserved:,} tokens 用于 Reasoning Model 的隐式推理")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Budget Manager 使用示例")
    print("=" * 70)

    demo_basic_allocation()
    demo_custom_weights()
    demo_large_window()
    demo_reasoning_model()

    print("\n" + "=" * 70)
    print("示例运行完成")
    print("=" * 70)
