"""
场景 5：Prompt 版本管理与回归

演示如何使用 Context Forge 进行 Prompt 版本管理：
- System Prompt v1 vs v2
- Snapshot 保存/加载
- 结构化 Diff
- Golden Set 回归
- Metrics 对比

使用方法：
  python examples/scenario_versioning.py          # 使用 mock 模式（默认）
  python examples/scenario_versioning.py --mock   # 明确指定 mock
  python examples/scenario_versioning.py --no-mock # 使用真实 LLM（需要 API Key）

→ 6.5.1 Context Snapshot
→ 6.5.2 Prompt Diff
→ 6.5.3 Golden Set 回归测试
"""

import asyncio
import json
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

    print_header(
        "场景 5：Prompt 版本管理与回归",
        "演示如何使用 Snapshot、Diff、Golden Set 管理 Prompt 版本变更"
    )

    # 检查 API Key
    if not check_api_key(mock):
        mock = True

    print_section("步骤 1：创建 Prompt v1（基线版本）")

    # v1：基础版本，较为通用
    system_prompt_v1 = """你是一个客服助手，负责回答用户关于产品和服务的问题。

你应该：
- 礼貌友好地回答问题
- 提供准确的信息
- 如果不确定，诚实告知用户"""

    console.print("[bold]System Prompt v1:[/bold]\n")
    console.print(f"[dim]{system_prompt_v1}[/dim]\n")

    # 创建 ContextForge 实例（启用 Snapshot）
    from context_forge.config.schema import ObservabilityConfig

    forge_v1 = ContextForge(
        model="gpt-4o",
        max_context_tokens=8192,
    )

    # 启用快照
    forge_v1._policy = forge_v1._policy.model_copy(update={
        "observability": ObservabilityConfig(
            snapshot_enabled=True,
            tracing_enabled=False,
            metrics_enabled=True,
            export_format="json",
            snapshot_dir=".context_forge/snapshots",
        ),
    })

    # 重新创建组件
    from context_forge.observability import SnapshotManager
    forge_v1._snapshot_manager = SnapshotManager(
        storage_dir=forge_v1._policy.observability.snapshot_dir
    )

    # 测试用例
    test_cases = [
        {
            "id": "test_1",
            "messages": [
                {"role": "user", "content": "你们的退货政策是什么？"},
            ],
            "rag_chunks": [
                {"content": "退货政策：7 天内无理由退货。", "score": 0.95},
                {"content": "退款流程：审核通过后 3-5 个工作日到账。", "score": 0.88},
            ],
        },
        {
            "id": "test_2",
            "messages": [
                {"role": "user", "content": "如何联系客服？"},
            ],
            "rag_chunks": [
                {"content": "客服电话：400-123-4567，工作时间 9:00-18:00。", "score": 0.92},
                {"content": "在线客服：点击右下角对话框。", "score": 0.85},
            ],
        },
    ]

    # 使用 v1 组装上下文
    context_v1 = await forge_v1.build(
        system_prompt=system_prompt_v1,
        messages=test_cases[0]["messages"],
        rag_chunks=test_cases[0]["rag_chunks"],
    )

    # 保存 v1 快照
    snapshot_id_v1 = await forge_v1.save_snapshot(context_v1)

    print_success(f"Prompt v1 快照已保存：{snapshot_id_v1}\n")

    # 显示 v1 指标
    v1_metrics = create_comparison_table(
        "v1 指标",
        ["指标", "值"],
        [
            ["Segment 数量", str(len(context_v1.segments))],
            ["总 Token", format_tokens(context_v1.token_usage.total_tokens)],
            ["预算饱和度", format_percentage(context_v1.budget_allocation.saturation_rate)],
            ["组装耗时", f"{context_v1.assembly_duration_ms:.1f}ms"],
        ]
    )
    console.print(v1_metrics)
    console.print()

    print_section("步骤 2：创建 Prompt v2（改进版本）")

    # v2：改进版本，更具体、更有针对性
    system_prompt_v2 = """你是 XYZ 电商平台的智能客服助手，专注于提供专业、高效的客户支持。

核心职责：
1. 根据知识库内容准确回答用户问题
2. 主动识别用户需求，提供个性化建议
3. 对于复杂问题，引导用户联系人工客服

回答规范：
- 使用友好、专业的语气
- 优先引用知识库中的信息
- 如果知识库无相关内容，明确告知用户并提供替代方案
- 回答控制在 150 字以内，简洁明了

禁止事项：
- 不得提供未经验证的信息
- 不得承诺超出职责范围的事项
- 不得泄露其他用户的信息"""

    console.print("[bold]System Prompt v2:[/bold]\n")
    console.print(f"[dim]{system_prompt_v2}[/dim]\n")

    # 使用 v2 组装上下文
    forge_v2 = ContextForge(
        model="gpt-4o",
        max_context_tokens=8192,
    )

    # 启用快照
    forge_v2._policy = forge_v2._policy.model_copy(update={
        "observability": ObservabilityConfig(
            snapshot_enabled=True,
            tracing_enabled=False,
            metrics_enabled=True,
            export_format="json",
            snapshot_dir=".context_forge/snapshots",
        ),
    })

    forge_v2._snapshot_manager = SnapshotManager(
        storage_dir=forge_v2._policy.observability.snapshot_dir
    )

    context_v2 = await forge_v2.build(
        system_prompt=system_prompt_v2,
        messages=test_cases[0]["messages"],
        rag_chunks=test_cases[0]["rag_chunks"],
    )

    # 保存 v2 快照
    snapshot_id_v2 = await forge_v2.save_snapshot(context_v2)

    print_success(f"Prompt v2 快照已保存：{snapshot_id_v2}\n")

    # 显示 v2 指标
    v2_metrics = create_comparison_table(
        "v2 指标",
        ["指标", "值"],
        [
            ["Segment 数量", str(len(context_v2.segments))],
            ["总 Token", format_tokens(context_v2.token_usage.total_tokens)],
            ["预算饱和度", format_percentage(context_v2.budget_allocation.saturation_rate)],
            ["组装耗时", f"{context_v2.assembly_duration_ms:.1f}ms"],
        ]
    )
    console.print(v2_metrics)
    console.print()

    print_section("步骤 3：Diff 分析（v1 vs v2）")

    # 使用 DiffEngine 对比
    diff_result = await forge_v2.diff_snapshots(snapshot_id_v1, snapshot_id_v2)

    console.print("[bold]变更摘要：[/bold]\n")

    # 显示 Segment 变更
    summary = diff_result.get("summary", {})
    console.print(f"  - 新增 Segment：[green]{summary.get('added', 0)}[/green] 个")
    console.print(f"  - 删除 Segment：[red]{summary.get('removed', 0)}[/red] 个")
    console.print(f"  - 修改 Segment：[yellow]{summary.get('modified', 0)}[/yellow] 个")
    console.print()

    # 显示 Token 变化
    token_change = context_v2.token_usage.total_tokens - context_v1.token_usage.total_tokens
    token_change_pct = token_change / context_v1.token_usage.total_tokens if context_v1.token_usage.total_tokens > 0 else 0

    token_color = "green" if token_change < 0 else "red"
    console.print(f"  - Token 变化：[{token_color}]{token_change:+d}[/{token_color}] ({format_percentage(abs(token_change_pct))})")
    console.print()

    # 显示预算变化
    saturation_change = context_v2.budget_allocation.saturation_rate - context_v1.budget_allocation.saturation_rate

    console.print(f"  - 预算饱和度变化：{saturation_change:+.1%}\n")

    # 详细变更表
    modified_entries = [e for e in diff_result.get("entries", []) if e.get("type") == "modified"]
    if modified_entries:
        console.print("[bold]修改详情（System Prompt）：[/bold]\n")

        for change in modified_entries[:1]:
            console.print(f"  Path: [dim]{change.get('path', 'unknown')}[/dim]")
            console.print(f"  类型: {change.get('type', 'unknown')}")
            console.print(f"  描述: {change.get('description', '')}\n")

    print_section("步骤 4：Golden Set 回归测试")

    # 将 v1 作为 Golden（基线）
    console.print(f"[bold]使用 v1 作为 Golden Set 基线[/bold]\n")

    # 对 v2 进行回归测试
    regression_result = await forge_v2.validate_against_golden(
        golden_snapshot_id=snapshot_id_v1,
        current_package=context_v2,
    )

    # 显示回归结果
    passed = regression_result.get("passed", False)
    issues = regression_result.get("entries", [])

    result_color = "green" if passed else "red"
    result_text = "OK 通过" if passed else "X 失败"

    console.print(f"回归测试：[{result_color}]{result_text}[/{result_color}]\n")

    if issues:
        console.print(f"[bold yellow]发现 {len(issues)} 个问题：[/bold yellow]\n")
        for issue in issues[:5]:
            print_warning(f"{issue.get('type', 'unknown')}: {issue.get('description', '')}")
        console.print()
    else:
        print_success("未发现回归问题")
        console.print()

    # 回归指标对比
    regression_metrics = create_comparison_table(
        "回归指标对比",
        ["指标", "Golden (v1)", "Current (v2)", "差异", "状态"],
        [
            [
                "Segment 数量",
                str(len(context_v1.segments)),
                str(len(context_v2.segments)),
                f"{len(context_v2.segments) - len(context_v1.segments):+d}",
                "OK" if abs(len(context_v2.segments) - len(context_v1.segments)) <= 2 else "!",
            ],
            [
                "总 Token",
                format_tokens(context_v1.token_usage.total_tokens),
                format_tokens(context_v2.token_usage.total_tokens),
                f"{context_v2.token_usage.total_tokens - context_v1.token_usage.total_tokens:+d}",
                "OK" if abs(token_change_pct) < 0.2 else "!",
            ],
            [
                "预算饱和度",
                format_percentage(context_v1.budget_allocation.saturation_rate),
                format_percentage(context_v2.budget_allocation.saturation_rate),
                f"{saturation_change:+.1%}",
                "OK" if abs(saturation_change) < 0.1 else "!",
            ],
        ]
    )
    console.print(regression_metrics)
    console.print()

    print_section("步骤 5：批量测试（Golden Set）")

    # 使用所有测试用例
    console.print("[bold]运行所有测试用例...[/bold]\n")

    golden_results = []

    for i, test_case in enumerate(test_cases):
        # v1（Golden）
        ctx_v1 = await forge_v1.build(
            system_prompt=system_prompt_v1,
            messages=test_case["messages"],
            rag_chunks=test_case["rag_chunks"],
        )

        # v2（Current）
        ctx_v2 = await forge_v2.build(
            system_prompt=system_prompt_v2,
            messages=test_case["messages"],
            rag_chunks=test_case["rag_chunks"],
        )

        # 对比
        token_diff = ctx_v2.token_usage.total_tokens - ctx_v1.token_usage.total_tokens
        saturation_diff = ctx_v2.budget_allocation.saturation_rate - ctx_v1.budget_allocation.saturation_rate

        golden_results.append({
            "id": test_case["id"],
            "v1_tokens": ctx_v1.token_usage.total_tokens,
            "v2_tokens": ctx_v2.token_usage.total_tokens,
            "token_diff": token_diff,
            "saturation_diff": saturation_diff,
            "passed": abs(token_diff / ctx_v1.token_usage.total_tokens) < 0.3,  # 允许 30% 波动
        })

    # 显示结果
    golden_table = create_comparison_table(
        "Golden Set 测试结果",
        ["测试ID", "v1 Token", "v2 Token", "差异", "状态"],
        [
            [
                result["id"],
                format_tokens(result["v1_tokens"]),
                format_tokens(result["v2_tokens"]),
                f"{result['token_diff']:+d}",
                "[green]OK PASS[/green]" if result["passed"] else "[red]X FAIL[/red]",
            ]
            for result in golden_results
        ]
    )
    console.print(golden_table)
    console.print()

    # 统计通过率
    pass_count = len([r for r in golden_results if r["passed"]])
    pass_rate = pass_count / len(golden_results)

    console.print(f"通过率：[bold]{format_percentage(pass_rate)}[/bold] ({pass_count}/{len(golden_results)})\n")

    print_section("步骤 6：版本管理建议")

    # 基于对比结果给出建议
    avg_token_diff = sum(r["token_diff"] for r in golden_results) / len(golden_results)
    avg_saturation_diff = sum(r["saturation_diff"] for r in golden_results) / len(golden_results)

    console.print("[bold]变更影响分析：[/bold]\n")

    if avg_token_diff > 0:
        print_warning(f"v2 平均增加 {abs(avg_token_diff):.0f} Token，成本上升约 {format_percentage(abs(avg_token_diff) / context_v1.token_usage.total_tokens)}")
    else:
        print_success(f"v2 平均节省 {abs(avg_token_diff):.0f} Token，成本下降约 {format_percentage(abs(avg_token_diff) / context_v1.token_usage.total_tokens)}")

    if abs(avg_saturation_diff) < 0.05:
        print_success("预算饱和度变化很小，窗口利用率稳定")
    else:
        print_warning(f"预算饱和度变化 {avg_saturation_diff:+.1%}，需要关注")

    console.print()

    console.print("[bold]部署建议：[/bold]\n")

    if pass_rate >= 0.9:
        print_success("OK 回归测试通过率 ≥ 90%，建议部署")
    elif pass_rate >= 0.7:
        print_warning("! 回归测试通过率 70-90%，建议人工复审")
    else:
        from examples._shared import print_error
        print_error("X 回归测试通过率 < 70%，不建议部署")

    console.print()

    print_section("总结")

    print_success(f"Prompt 版本管理完成！")
    print_success(f"- v1 和 v2 快照已保存，可随时回溯")
    print_success(f"- Diff 分析显示：System Prompt 增加 {len(system_prompt_v2) - len(system_prompt_v1)} 字符")
    print_success(f"- Golden Set 测试通过率：{format_percentage(pass_rate)}")
    print_success(f"- Token 变化：{avg_token_diff:+.0f} 平均")
    print_success(f"- 预算饱和度变化：{avg_saturation_diff:+.1%}")

    console.print(f"\n[dim]提示：快照已保存到 {forge_v1._policy.observability.snapshot_dir}[/dim]")
    console.print(f"[dim]可使用 `context-forge diff {snapshot_id_v1} {snapshot_id_v2}` 查看详细差异[/dim]")


if __name__ == "__main__":
    args = parse_args("场景 5：Prompt 版本管理与回归")
    asyncio.run(main(mock=args.mock))
