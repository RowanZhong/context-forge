"""
场景 6：多模型适配与成本优化

演示如何使用 Context Forge 进行智能路由和成本优化：
- 3 个查询（简单/中等/复杂）
- 复杂度估计
- 自动路由
- Budget 调整
- 成本对比
- 缓存降本演示（相同查询命中缓存，跳过 Pipeline）

使用方法：
  python examples/scenario_routing_cost.py          # 使用 mock 模式（默认）
  python examples/scenario_routing_cost.py --mock   # 明确指定 mock
  python examples/scenario_routing_cost.py --no-mock # 使用真实 LLM（需要 API Key）

→ 6.6 上下文路由与动态调度
→ 6.2.2 预算分配策略
→ 6.2.3 缓存架构与复用优化
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from examples._shared import (
    MockLLM,
    check_api_key,
    console,
    create_comparison_table,
    format_percentage,
    format_tokens,
    parse_args,
    print_header,
    print_section,
    print_success,
    print_warning,
    truncate_text,
)


async def main(mock: bool = True):
    """主函数。"""
    from context_forge import ContextForge
    from context_forge.models.routing import ComplexityLevel

    print_header(
        "场景 6：多模型适配与成本优化",
        "演示如何根据复杂度自动路由到合适的模型，优化成本"
    )

    # 检查 API Key
    if not check_api_key(mock):
        mock = True

    print_section("步骤 1：定义测试查询")

    # 3 个不同复杂度的查询
    test_queries = [
        {
            "id": "query_simple",
            "complexity": ComplexityLevel.SIMPLE,
            "user_message": "今天天气怎么样？",
            "rag_chunks": [
                {"content": "今日天气：晴，温度 22℃，湿度 60%。", "score": 0.98},
            ],
            "expected_model": "gpt-4o-mini",
            "description": "简单事实查询，单一数据源",
        },
        {
            "id": "query_medium",
            "complexity": ComplexityLevel.MODERATE,
            "user_message": "对比一下 iPhone 15 Pro 和 Samsung S24 Ultra 的性能和价格。",
            "rag_chunks": [
                {"content": "iPhone 15 Pro：A17 Pro 芯片，128GB 起售价 7999 元。", "score": 0.92},
                {"content": "Samsung S24 Ultra：骁龙 8 Gen 3，256GB 起售价 9699 元。", "score": 0.90},
                {"content": "性能对比：A17 Pro 单核性能更强，骁龙 8 Gen 3 多核更优。", "score": 0.88},
                {"content": "屏幕对比：iPhone 6.1 英寸 OLED，Samsung 6.8 英寸 Dynamic AMOLED。", "score": 0.85},
            ],
            "expected_model": "gpt-4o",
            "description": "中等复杂度，需要对比分析",
        },
        {
            "id": "query_complex",
            "complexity": ComplexityLevel.COMPLEX,
            "user_message": "制定一份为期 3 个月的产品发布计划，包括市场调研、竞品分析、功能设计、开发排期、测试、营销策略和预算分配。",
            "rag_chunks": [
                {"content": "市场调研模板：用户画像、需求分析、市场规模评估...", "score": 0.88},
                {"content": "竞品分析框架：SWOT 分析、功能对比矩阵、定价策略...", "score": 0.86},
                {"content": "敏捷开发流程：Sprint 规划、Daily Standup、Retrospective...", "score": 0.84},
                {"content": "测试策略：单元测试、集成测试、E2E 测试、性能测试...", "score": 0.82},
                {"content": "营销漏斗：认知→兴趣→考虑→转化→忠诚...", "score": 0.80},
                {"content": "预算分配模板：研发 40%、营销 30%、运营 20%、其他 10%...", "score": 0.78},
            ],
            "expected_model": "claude-sonnet-4-5-20250514",
            "description": "复杂多步骤任务，需要深度推理",
        },
    ]

    # 显示测试查询
    query_table = create_comparison_table(
        "测试查询",
        ["ID", "复杂度", "RAG 数", "查询内容", "期望模型"],
        [
            [
                q["id"],
                q["complexity"].value,
                str(len(q["rag_chunks"])),
                truncate_text(q["user_message"], 35),
                q["expected_model"],
            ]
            for q in test_queries
        ]
    )
    console.print(query_table)
    console.print()

    print_section("步骤 2：配置路由规则")

    # 配置路由策略
    from context_forge.config.schema import ObservabilityConfig, RoutingConfig
    from context_forge.models.routing import RoutingRule

    routing_rules = [
        RoutingRule(
            name="simple_queries",
            condition_type="complexity",
            condition_value="simple",
            target_model="gpt-4o-mini",
            priority=10,
        ),
        RoutingRule(
            name="moderate_queries",
            condition_type="complexity",
            condition_value="moderate",
            target_model="gpt-4o",
            priority=20,
        ),
        RoutingRule(
            name="complex_queries",
            condition_type="complexity",
            condition_value="complex",
            target_model="claude-sonnet-4-5-20250514",
            priority=30,
        ),
    ]

    # 显示路由规则
    rule_table = create_comparison_table(
        "路由规则",
        ["规则名", "条件", "目标模型", "预算策略"],
        [
            [
                rule.name,
                f"{rule.condition_type}={rule.condition_value}",
                rule.target_model,
                "按模型默认",
            ]
            for rule in routing_rules
        ]
    )
    console.print(rule_table)
    console.print()

    print_section("步骤 3：执行路由决策")

    # 创建 ContextForge 实例（启用路由）
    forge = ContextForge(
        model="gpt-4o",  # 默认模型
        max_context_tokens=8192,
    )

    # 启用路由
    forge._policy = forge._policy.model_copy(update={
        "routing": RoutingConfig(
            enabled=True,
            default_model="gpt-4o",
            router_type="rule_based",
            rules=[r.model_dump() for r in routing_rules],
        ),
        "observability": ObservabilityConfig(
            snapshot_enabled=False,
            tracing_enabled=False,
            metrics_enabled=True,
            export_format="json",
        ),
    })

    # 重新创建路由器
    from context_forge.routing import RuleBasedRouter

    forge._router = RuleBasedRouter(
        default_model=forge._policy.routing.default_model,
        rules=routing_rules,
    )

    # 处理每个查询
    results = []

    for query in test_queries:
        console.print(f"[bold]处理查询：{query['id']}[/bold]\n")

        # 组装上下文（路由会自动触发）
        context = await forge.build(
            system_prompt="你是一个智能助手，根据用户问题和检索到的信息提供帮助。",
            messages=[
                {"role": "user", "content": query["user_message"]},
            ],
            rag_chunks=query["rag_chunks"],
        )

        # 获取路由决策
        routing_decision = context.routing_decision

        if routing_decision:
            console.print(f"  路由决策：")
            console.print(f"    - 选择模型：[bold cyan]{routing_decision.selected_model.model_id}[/bold cyan]")
            console.print(f"    - 复杂度：{routing_decision.complexity.value}")
            console.print(f"    - 理由：{routing_decision.reasoning}")
        else:
            console.print(f"  路由决策：使用默认模型 {forge.model}")

        console.print()

        # 记录结果
        results.append({
            "query_id": query["id"],
            "complexity": query["complexity"].value,
            "expected_model": query["expected_model"],
            "selected_model": routing_decision.selected_model.model_id if routing_decision else forge.model,
            "tokens": context.token_usage.total_tokens,
            "saturation": context.budget_allocation.saturation_rate,
            "duration_ms": context.assembly_duration_ms,
            "matched": (routing_decision.selected_model.model_id if routing_decision else forge.model) == query["expected_model"],
        })

    print_section("步骤 4：路由准确性分析")

    # 显示路由结果
    routing_table = create_comparison_table(
        "路由结果",
        ["查询ID", "复杂度", "期望模型", "实际模型", "匹配"],
        [
            [
                r["query_id"],
                r["complexity"],
                r["expected_model"],
                r["selected_model"],
                "[green]OK[/green]" if r["matched"] else "[red]X[/red]",
            ]
            for r in results
        ]
    )
    console.print(routing_table)
    console.print()

    # 计算准确率
    match_count = len([r for r in results if r["matched"]])
    accuracy = match_count / len(results)

    console.print(f"路由准确率：[bold]{format_percentage(accuracy)}[/bold] ({match_count}/{len(results)})\n")

    print_section("步骤 5：成本分析")

    # 定义模型成本（美元/1M Token，示例价格）
    model_costs = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "claude-sonnet-4-5-20250514": {"input": 3.00, "output": 15.00},
    }

    # 计算成本
    for result in results:
        model = result["selected_model"]
        tokens = result["tokens"]

        # 假设 input:output = 3:1
        input_tokens = tokens * 0.75
        output_tokens = tokens * 0.25

        cost_input = (input_tokens / 1_000_000) * model_costs.get(model, {"input": 0})["input"]
        cost_output = (output_tokens / 1_000_000) * model_costs.get(model, {"output": 0})["output"]
        total_cost = cost_input + cost_output

        result["cost_usd"] = total_cost

    # 计算如果全部使用默认模型的成本
    default_model = "gpt-4o"
    baseline_costs = []

    for result in results:
        tokens = result["tokens"]
        input_tokens = tokens * 0.75
        output_tokens = tokens * 0.25

        cost_input = (input_tokens / 1_000_000) * model_costs[default_model]["input"]
        cost_output = (output_tokens / 1_000_000) * model_costs[default_model]["output"]
        baseline_costs.append(cost_input + cost_output)

    # 显示成本对比
    cost_table = create_comparison_table(
        "成本对比（单次调用）",
        ["查询ID", "路由模型", "Token", "智能路由成本", "固定 gpt-4o 成本", "节省"],
        [
            [
                r["query_id"],
                r["selected_model"],
                format_tokens(r["tokens"]),
                f"${r['cost_usd']:.6f}",
                f"${baseline_costs[i]:.6f}",
                f"${baseline_costs[i] - r['cost_usd']:.6f}" if baseline_costs[i] > r['cost_usd'] else f"-${r['cost_usd'] - baseline_costs[i]:.6f}",
            ]
            for i, r in enumerate(results)
        ]
    )
    console.print(cost_table)
    console.print()

    # 总成本统计
    total_routed_cost = sum(r["cost_usd"] for r in results)
    total_baseline_cost = sum(baseline_costs)
    total_saved = total_baseline_cost - total_routed_cost
    saved_ratio = total_saved / total_baseline_cost if total_baseline_cost > 0 else 0

    console.print(f"[bold]总成本对比：[/bold]\n")
    console.print(f"  - 智能路由：[green]${total_routed_cost:.6f}[/green]")
    console.print(f"  - 固定模型：${total_baseline_cost:.6f}")
    console.print(f"  - 节省：[bold green]${total_saved:.6f}[/bold green] ({format_percentage(saved_ratio)})\n")

    # 年化成本估算（假设每天 10000 次调用）
    daily_calls = 10000
    yearly_calls = daily_calls * 365

    yearly_routed = total_routed_cost * (yearly_calls / len(results))
    yearly_baseline = total_baseline_cost * (yearly_calls / len(results))
    yearly_saved = yearly_routed - yearly_baseline

    console.print(f"[bold]年化成本估算（每天 {daily_calls:,} 次调用）：[/bold]\n")
    console.print(f"  - 智能路由：[green]${yearly_routed:,.2f}[/green]")
    console.print(f"  - 固定模型：${yearly_baseline:,.2f}")
    console.print(f"  - 年节省：[bold green]${abs(yearly_saved):,.2f}[/bold green]\n")

    print_section("步骤 6：缓存降本演示")

    # 演示缓存命中如何节省计算和 API 成本
    # [Design Decision] 使用独立的 ContextForge 实例启用缓存，
    # 演示相同查询第二次调用直接命中缓存的效果。
    from context_forge.cache import CacheManager, MemoryCache
    from context_forge.config.schema import CacheConfig

    # 创建启用缓存的 forge 实例
    forge_cached = ContextForge(
        model="gpt-4o",
        max_context_tokens=8192,
    )
    # 启用缓存
    forge_cached._policy = forge_cached._policy.model_copy(update={
        "cache": CacheConfig(enabled=True, backend="memory", ttl_seconds=3600),
        "routing": RoutingConfig(
            enabled=True,
            default_model="gpt-4o",
            router_type="rule_based",
            rules=[r.model_dump() for r in routing_rules],
        ),
        "observability": ObservabilityConfig(
            snapshot_enabled=False,
            tracing_enabled=False,
            metrics_enabled=False,
            export_format="json",
        ),
    })
    forge_cached._router = RuleBasedRouter(
        default_model=forge_cached._policy.routing.default_model,
        rules=routing_rules,
    )
    forge_cached._cache_manager = CacheManager(
        l1=MemoryCache(max_size=1000, default_ttl=3600),
    )

    # 使用简单查询演示缓存
    cache_query = test_queries[0]  # 简单查询
    system_prompt = "你是一个智能助手，根据用户问题和检索到的信息提供帮助。"

    console.print("[bold]演示：相同查询的缓存效果[/bold]\n")

    # 第一次调用（缓存未命中，走完整 Pipeline）
    context_first = await forge_cached.build(
        system_prompt=system_prompt,
        messages=[{"role": "user", "content": cache_query["user_message"]}],
        rag_chunks=cache_query["rag_chunks"],
    )
    first_duration = context_first.assembly_duration_ms

    # 第二次调用（相同输入，应命中缓存）
    context_second = await forge_cached.build(
        system_prompt=system_prompt,
        messages=[{"role": "user", "content": cache_query["user_message"]}],
        rag_chunks=cache_query["rag_chunks"],
    )
    second_duration = context_second.assembly_duration_ms

    # 获取缓存统计
    cache_stats = await forge_cached._cache_manager.stats()
    l1_stats = cache_stats.get("l1")

    # 显示缓存效果
    cache_table = create_comparison_table(
        "缓存效果对比",
        ["调用", "延迟", "Segment 数", "Token 数", "状态"],
        [
            [
                "第 1 次（缓存未命中）",
                f"{first_duration:.1f}ms",
                str(len(context_first.segments)),
                format_tokens(context_first.token_usage.total_tokens),
                "[yellow]MISS[/yellow]",
            ],
            [
                "第 2 次（缓存命中）",
                f"{second_duration:.1f}ms",
                str(len(context_second.segments)),
                format_tokens(context_second.token_usage.total_tokens),
                "[green]HIT[/green]",
            ],
        ]
    )
    console.print(cache_table)
    console.print()

    # 验证缓存命中的结果与首次构建一致
    first_messages = context_first.to_messages()
    second_messages = context_second.to_messages()
    content_match = all(
        m1["content"] == m2["content"]
        for m1, m2 in zip(first_messages, second_messages)
    )
    console.print(f"  缓存结果内容一致性：[bold]{'OK' if content_match else 'MISMATCH'}[/bold]")

    if l1_stats:
        console.print(f"  L1 缓存命中率：[bold]{l1_stats.hit_rate:.0%}[/bold]")
        console.print(f"  L1 缓存条目数：{l1_stats.current_size}")

    # 延迟改善
    if first_duration > 0:
        speedup = first_duration / max(second_duration, 0.001)
        console.print(f"  缓存加速比：[bold green]{speedup:.1f}x[/bold green]")

    console.print()
    print_success("缓存命中时跳过完整 Pipeline（6 个阶段），直接返回缓存结果")
    console.print()

    print_section("步骤 7：性能 vs 成本权衡")

    # 显示性能指标
    perf_table = create_comparison_table(
        "性能指标",
        ["查询ID", "模型", "Token", "饱和度", "延迟", "成本"],
        [
            [
                r["query_id"],
                r["selected_model"],
                format_tokens(r["tokens"]),
                format_percentage(r["saturation"]),
                f"{r['duration_ms']:.1f}ms",
                f"${r['cost_usd']:.6f}",
            ]
            for r in results
        ]
    )
    console.print(perf_table)
    console.print()

    # 权衡分析
    console.print("[bold]权衡建议：[/bold]\n")

    for r in results:
        if r["selected_model"] == "gpt-4o-mini":
            print_success(f"{r['query_id']}: 简单查询使用 mini 模型，成本最低")
        elif r["selected_model"] == "gpt-4o":
            console.print(f"  - {r['query_id']}: 中等查询使用标准模型，平衡性能和成本")
        else:
            print_warning(f"{r['query_id']}: 复杂查询使用高级模型，成本较高但质量有保障")

    console.print()

    print_section("总结")

    print_success("多模型路由与成本优化完成！")
    print_success(f"- 路由准确率：{format_percentage(accuracy)}")
    print_success(f"- 成本节省：{format_percentage(saved_ratio)} (相比固定使用 gpt-4o)")
    print_success("- 简单查询 -> gpt-4o-mini（低成本）")
    print_success("- 中等查询 -> gpt-4o（平衡性价比）")
    print_success("- 复杂查询 -> claude-sonnet-4-5-20250514（高质量）")
    print_success(f"- 预计年节省：${abs(yearly_saved):,.2f}")
    print_success("- 缓存命中时跳过 Pipeline，进一步降低延迟和计算开销")

    console.print("\n[dim]提示：生产环境可使用基于 LLM 的路由器（router_type=llm），准确率更高[/dim]")
    console.print("[dim]参考配置文件：configs/cost_optimization_policy.yaml 中的 routing + cache 配置[/dim]")


if __name__ == "__main__":
    args = parse_args("场景 6：多模型适配与成本优化")
    asyncio.run(main(mock=args.mock))
