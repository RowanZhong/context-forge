"""
路由模块示例——展示复杂度估计、规则路由和多 Agent 协调。

→ 6.6 上下文路由与动态调度

运行方式:
    python examples/routing_demo.py
"""

from __future__ import annotations

from context_forge.models.segment import Segment, SegmentType
from context_forge.routing import (
    AgentContext,
    ComplexityEstimator,
    ContextBus,
    HandoffRequest,
    RoutingContext,
    create_default_router,
)


def demo_complexity_estimation() -> None:
    """演示 1: 复杂度估计。"""
    print("=" * 60)
    print("演示 1: 复杂度估计")
    print("=" * 60)

    estimator = ComplexityEstimator()

    queries = [
        ("退货地址是哪？", "简单 FAQ"),
        ("请比较 Python 和 Go 的并发模型", "对比分析"),
        ("请设计一个高可用的分布式缓存系统，要求支持数据分片和自动故障转移", "复杂设计任务"),
        ("证明费马大定理在 n=3 时的情况", "专家级数学证明"),
    ]

    for query, desc in queries:
        signals = estimator.estimate_with_signals(query)
        print(f"\n查询: {query}")
        print(f"  描述: {desc}")
        print(f"  复杂度: {signals.estimated_level.value}")
        print(f"  置信度: {signals.confidence:.2f}")
        print(f"  关键特征:")
        if signals.has_comparison_words:
            print(f"    - 包含对比词")
        if signals.has_reasoning_words:
            print(f"    - 包含推理词")
        if signals.has_complex_task_words:
            print(f"    - 包含复杂任务词")
        if signals.code_block_count > 0:
            print(f"    - {signals.code_block_count} 个代码块")
        print(f"  查询长度: {signals.query_length} 字符")


def demo_rule_based_routing() -> None:
    """演示 2: 规则路由器。"""
    print("\n" + "=" * 60)
    print("演示 2: 规则路由器")
    print("=" * 60)

    # 创建默认路由器
    router = create_default_router(router_type="rule")

    queries = [
        "退货地址是哪？",
        "请分析 Python 的 GIL 机制",
        "请设计一个分布式系统架构",
    ]

    for query in queries:
        # 创建路由上下文
        segment = Segment(
            type=SegmentType.USER,
            content=query,
            role="user",
            token_count=len(query),
        )

        context = RoutingContext(
            segments=[segment],
            query=query,
            max_budget_tokens=128_000,
        )

        # 执行路由
        decision = router.route(context)

        print(f"\n查询: {query}")
        print(f"  选中模型: {decision.selected_model.model_id}")
        print(f"  复杂度: {decision.complexity.value}")
        print(f"  匹配规则: {decision.matched_rule}")
        print(f"  置信度: {decision.confidence:.2f}")
        print(f"  成本: ${decision.selected_model.cost_per_million_input:.2f}/M 输入")


def demo_context_bus() -> None:
    """演示 3: 多 Agent 上下文协调。"""
    print("\n" + "=" * 60)
    print("演示 3: 多 Agent 上下文协调")
    print("=" * 60)

    # 创建上下文总线
    bus = ContextBus()

    # 创建三个 Agent
    planner = AgentContext(
        agent_id="planner",
        namespace="planning",
        role="planner",
    )
    executor = AgentContext(
        agent_id="executor",
        namespace="execution",
        role="executor",
    )
    reviewer = AgentContext(
        agent_id="reviewer",
        namespace="review",
        role="reviewer",
    )

    # 注册 Agent
    bus.register_agent(planner)
    bus.register_agent(executor)
    bus.register_agent(reviewer)

    print("\n步骤 1: Planner 创建计划")
    plan_segment = Segment(
        type=SegmentType.STATE,
        content="任务计划：1) 数据收集 2) 分析处理 3) 生成报告",
        role="assistant",
    )
    bus.publish_segment(planner, plan_segment)

    stats = bus.get_namespace_stats("planning")
    print(f"  Planning namespace 统计: {stats['segment_count']} 个 Segment")

    print("\n步骤 2: Planner 移交上下文给 Executor")
    handoff = HandoffRequest(
        from_agent_id="planner",
        to_agent_id="executor",
        reason="规划完成，开始执行",
    )
    bus.handoff(handoff)

    executor_segments = bus.get_visible_segments(executor)
    print(f"  Executor 收到 {len(executor_segments)} 个 Segment")

    print("\n步骤 3: Executor 执行任务并记录结果")
    result_segment = Segment(
        type=SegmentType.ASSISTANT,
        content="执行结果：数据收集完成，分析中...",
        role="assistant",
    )
    bus.publish_segment(executor, result_segment)

    stats = bus.get_namespace_stats("execution")
    print(f"  Execution namespace 统计: {stats['segment_count']} 个 Segment")

    print("\n步骤 4: 查看各 Agent 的可见 Segment")
    print(f"  Planner 可见: {len(bus.get_visible_segments(planner))} 个")
    print(f"  Executor 可见: {len(bus.get_visible_segments(executor))} 个")
    print(f"  Reviewer 可见: {len(bus.get_visible_segments(reviewer))} 个")

    print("\n步骤 5: 查看最近事件")
    events = bus.get_recent_events(limit=5)
    for event in events:
        print(f"  事件: {event.event_type} | 发布者: {event.publisher_id}")


def demo_custom_routing_rules() -> None:
    """演示 4: 自定义路由规则。"""
    print("\n" + "=" * 60)
    print("演示 4: 自定义路由规则")
    print("=" * 60)

    from context_forge.models.routing import RoutingRule
    from context_forge.routing import RuleBasedRouter

    # 创建自定义规则
    custom_rules = [
        # 规则 1: 包含"代码"关键词的查询路由到 Claude Sonnet
        RoutingRule(
            name="code_to_sonnet",
            condition_type="keyword",
            condition_value="代码|code|编程|programming",
            target_model="claude-sonnet-4-5-20250514",
            priority=50,
        ),
        # 规则 2: Token 数量超过 1000 的查询路由到大窗口模型
        RoutingRule(
            name="large_context_to_opus",
            condition_type="token_count",
            condition_value=">1000",
            target_model="claude-opus-4-20250115",
            priority=40,
        ),
        # 规则 3: 包含 RAG 片段的查询路由到 GPT-4o
        RoutingRule(
            name="rag_to_gpt4o",
            condition_type="segment_type_present",
            condition_value="rag",
            target_model="gpt-4o",
            priority=30,
        ),
    ]

    router = RuleBasedRouter(
        rules=custom_rules,
        default_model="gpt-4o-mini",
    )

    # 测试场景
    scenarios = [
        ("请帮我写一段 Python 代码", []),
        ("这是一个普通问题", []),
        (
            "请根据以下文档回答",
            [
                Segment(
                    type=SegmentType.RAG,
                    content="文档片段",
                    role="user",
                    token_count=50,
                )
            ],
        ),
    ]

    for query, segments in scenarios:
        user_segment = Segment(
            type=SegmentType.USER,
            content=query,
            role="user",
            token_count=len(query),
        )
        all_segments = [user_segment] + segments

        context = RoutingContext(
            segments=all_segments,
            query=query,
            max_budget_tokens=4096,
        )

        decision = router.route(context)
        print(f"\n查询: {query}")
        print(f"  Segment 类型: {[s.type.value for s in all_segments]}")
        print(f"  路由到: {decision.selected_model.model_id}")
        print(f"  匹配规则: {decision.matched_rule}")


def main() -> None:
    """运行所有演示。"""
    print("\n路由模块完整演示")
    print("展示复杂度估计、规则路由、LLM 路由和多 Agent 协调")
    print()

    demo_complexity_estimation()
    demo_rule_based_routing()
    demo_context_bus()
    demo_custom_routing_rules()

    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
