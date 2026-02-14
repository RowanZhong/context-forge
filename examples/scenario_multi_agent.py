"""
场景 3：多 Agent 上下文协调

演示如何在多 Agent 系统中管理上下文：
- 3 个 Agent：Planner → Executor → Reviewer
- Namespace 隔离
- Publish/Subscribe 模式
- Handoff 机制
- Visibility 控制

使用方法：
  python examples/scenario_multi_agent.py          # 使用 mock 模式（默认）
  python examples/scenario_multi_agent.py --mock   # 明确指定 mock
  python examples/scenario_multi_agent.py --no-mock # 使用真实 LLM（需要 API Key）

→ 6.3.1.1 Namespace Isolation
→ 6.3.4 Isolate：多 Agent 隔离与协调
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from examples._shared import (
    MockLLM,
    check_api_key,
    console,
    create_comparison_table,
    create_tree,
    format_tokens,
    parse_args,
    print_header,
    print_section,
    print_success,
    truncate_text,
)


async def main(mock: bool = True):
    """主函数。"""
    from context_forge import ContextForge
    from context_forge.models.control import ControlFlags, Visibility
    from context_forge.models.segment import Priority, Segment, SegmentType
    from context_forge.models.provenance import Provenance, SourceType
    from context_forge.models.metadata import SegmentMetadata

    print_header(
        "场景 3：多 Agent 上下文协调",
        "演示如何使用 Namespace 隔离、Visibility 控制实现 Agent 协作"
    )

    # 检查 API Key
    if not check_api_key(mock):
        mock = True

    print_section("步骤 1：定义 Agent 层级结构")

    # Agent 层级：
    # - Planner（规划者）：接收用户需求，制定执行计划
    # - Executor（执行者）：根据计划执行具体任务
    # - Reviewer（审核者）：检查执行结果，提供反馈

    console.print("[bold]Agent 层级树：[/bold]\n")

    agent_tree = create_tree("系统（root）")
    planner_node = agent_tree.add("[cyan]Planner[/cyan] - 规划者")
    planner_node.add("[dim]职责：制定执行计划[/dim]")
    planner_node.add("[dim]命名空间：planner[/dim]")

    executor_node = agent_tree.add("[green]Executor[/green] - 执行者")
    executor_node.add("[dim]职责：执行具体任务[/dim]")
    executor_node.add("[dim]命名空间：executor[/dim]")

    reviewer_node = agent_tree.add("[yellow]Reviewer[/yellow] - 审核者")
    reviewer_node.add("[dim]职责：质量检查与反馈[/dim]")
    reviewer_node.add("[dim]命名空间：reviewer[/dim]")

    console.print(agent_tree)
    console.print()

    print_section("步骤 2：创建各 Agent 的上下文片段")

    now = datetime.now()

    # === Planner 的上下文 ===
    planner_segments = [
        # Planner 的系统提示（仅对自己可见）
        Segment(
            type=SegmentType.SYSTEM,
            content="你是一个任务规划专家，负责将用户需求分解为可执行的步骤。输出格式为 JSON 列表。",
            role="system",
            priority=Priority.CRITICAL,
            control=ControlFlags(
                namespace="planner",
                visibility=Visibility.NAMESPACE,  # 仅 planner 命名空间可见
                lock_position=True,
                must_keep=True,
            ),
            provenance=Provenance(
                source_id="planner_system",
                source_type=SourceType.SYSTEM_CONFIG,
            ),
        ),
        # Planner 的工作结果（对下游可见）
        Segment(
            type=SegmentType.ASSISTANT,
            content="""规划完成，执行步骤如下：
1. 调研目标用户群体和市场需求
2. 设计产品核心功能和架构
3. 制定开发计划和里程碑
4. 评估资源需求和预算""",
            role="assistant",
            priority=Priority.HIGH,
            control=ControlFlags(
                namespace="planner",
                visibility=Visibility.DOWNSTREAM,  # 对下游 Agent 可见
                handoff_to="executor",  # 交接给 Executor
            ),
            provenance=Provenance(
                source_id="planner_output_001",
                source_type=SourceType.SYSTEM_CONFIG,
                created_at=now - timedelta(minutes=30),
            ),
            metadata=SegmentMetadata(
                turn_number=1,
                injected_at=now - timedelta(minutes=30),
                debug_labels={"agent_name": "Planner"},
            ),
        ),
    ]

    # === Executor 的上下文 ===
    executor_segments = [
        # Executor 的系统提示
        Segment(
            type=SegmentType.SYSTEM,
            content="你是一个任务执行专家，根据规划步骤完成具体工作，输出详细的执行报告。",
            role="system",
            priority=Priority.CRITICAL,
            control=ControlFlags(
                namespace="executor",
                visibility=Visibility.NAMESPACE,
                lock_position=True,
                must_keep=True,
            ),
            provenance=Provenance(
                source_id="executor_system",
                source_type=SourceType.SYSTEM_CONFIG,
            ),
        ),
        # Executor 的工作进度（仅对自己可见）
        Segment(
            type=SegmentType.STATE,
            content="""当前进度：
- [完成] 步骤 1：用户调研（50 份问卷）
- [完成] 步骤 2：功能设计（原型图已完成）
- [进行中] 步骤 3：开发计划（预计明天完成）
- [待开始] 步骤 4：资源评估""",
            role="system",
            priority=Priority.HIGH,
            control=ControlFlags(
                namespace="executor",
                visibility=Visibility.NAMESPACE,  # 内部状态，不对外
            ),
            provenance=Provenance(
                source_id="executor_state",
                source_type=SourceType.SYSTEM_CONFIG,
                created_at=now - timedelta(minutes=15),
            ),
        ),
        # Executor 的执行结果（对下游可见）
        Segment(
            type=SegmentType.ASSISTANT,
            content="""执行报告：
步骤 1-2 已完成：
- 目标用户：25-40 岁城市白领，需要高效的时间管理工具
- 核心功能：日历管理、任务追踪、智能提醒、数据分析
- 技术架构：React + FastAPI + PostgreSQL

步骤 3 进行中，预计明天交付开发计划。""",
            role="assistant",
            priority=Priority.HIGH,
            control=ControlFlags(
                namespace="executor",
                visibility=Visibility.DOWNSTREAM,
                handoff_to="reviewer",
            ),
            provenance=Provenance(
                source_id="executor_output_001",
                source_type=SourceType.SYSTEM_CONFIG,
                created_at=now - timedelta(minutes=10),
            ),
            metadata=SegmentMetadata(
                turn_number=2,
                injected_at=now - timedelta(minutes=10),
                debug_labels={"agent_name": "Executor"},
            ),
        ),
    ]

    # === Reviewer 的上下文 ===
    reviewer_segments = [
        # Reviewer 的系统提示
        Segment(
            type=SegmentType.SYSTEM,
            content="你是一个质量审核专家，检查执行结果是否符合规划要求，提供改进建议。",
            role="system",
            priority=Priority.CRITICAL,
            control=ControlFlags(
                namespace="reviewer",
                visibility=Visibility.NAMESPACE,
                lock_position=True,
                must_keep=True,
            ),
            provenance=Provenance(
                source_id="reviewer_system",
                source_type=SourceType.SYSTEM_CONFIG,
            ),
        ),
        # Reviewer 的检查清单（内部资料）
        Segment(
            type=SegmentType.SCHEMA,
            content="""审核清单：
- [ ] 是否覆盖所有规划步骤
- [ ] 用户调研样本量是否足够
- [ ] 功能设计是否符合需求
- [ ] 技术方案是否可行
- [ ] 时间估算是否合理""",
            role="system",
            priority=Priority.MEDIUM,
            control=ControlFlags(
                namespace="reviewer",
                visibility=Visibility.NAMESPACE,  # 内部检查清单
            ),
            provenance=Provenance(
                source_id="reviewer_checklist",
                source_type=SourceType.SYSTEM_CONFIG,
            ),
        ),
        # Reviewer 的反馈（全局可见）
        Segment(
            type=SegmentType.ASSISTANT,
            content="""审核反馈：

OK 优点：
- 用户调研样本量充足（50 份）
- 功能设计贴合需求，有原型支撑
- 技术栈成熟，风险可控

! 改进建议：
- 建议补充竞品分析（对比 3-5 款同类产品）
- 开发计划应包含详细的时间节点和人力分配
- 预算评估建议分为开发成本、运营成本、营销成本三部分

总体评价：B+（良好，有待完善）""",
            role="assistant",
            priority=Priority.HIGH,
            control=ControlFlags(
                namespace="reviewer",
                visibility=Visibility.GLOBAL,  # 全局可见，所有 Agent 都能看到
                publish=True,  # 发布到全局上下文
            ),
            provenance=Provenance(
                source_id="reviewer_output_001",
                source_type=SourceType.SYSTEM_CONFIG,
                created_at=now - timedelta(minutes=5),
            ),
            metadata=SegmentMetadata(
                turn_number=3,
                injected_at=now - timedelta(minutes=5),
                debug_labels={"agent_name": "Reviewer"},
            ),
        ),
    ]

    # 显示各 Agent 的上下文统计
    agent_stats = create_comparison_table(
        "各 Agent 上下文统计",
        ["Agent", "Segment 数", "可见性配置", "说明"],
        [
            [
                "Planner",
                str(len(planner_segments)),
                "NAMESPACE + DOWNSTREAM",
                "系统提示私有，输出对下游可见",
            ],
            [
                "Executor",
                str(len(executor_segments)),
                "NAMESPACE + DOWNSTREAM",
                "内部状态私有，执行报告对下游可见",
            ],
            [
                "Reviewer",
                str(len(reviewer_segments)),
                "NAMESPACE + GLOBAL",
                "检查清单私有，审核反馈全局发布",
            ],
        ]
    )
    console.print(agent_stats)
    console.print()

    print_section("步骤 3：为各 Agent 组装上下文")

    # 创建 ContextForge 实例
    forge = ContextForge(
        model="gpt-4o",
        max_context_tokens=8192,
    )

    # === 组装 Planner 的上下文 ===
    console.print("[bold cyan]1. Planner 视角[/bold cyan]\n")

    planner_context = await forge.build(
        system_prompt="",  # 使用 Segment 中的系统提示
        messages=[
            {"role": "user", "content": "我想开发一款时间管理 App，请帮我制定产品规划。"},
        ],
        extra_segments=planner_segments,
        namespace="planner",
    )

    planner_visible = [s for s in planner_context.segments if s.control.namespace == "planner" or s.control.visibility == Visibility.GLOBAL]
    console.print(f"  可见 Segment：[bold]{len(planner_visible)}[/bold] 个")
    console.print(f"  总 Token：[bold]{format_tokens(planner_context.token_usage.total_tokens)}[/bold]\n")

    # === 组装 Executor 的上下文 ===
    console.print("[bold green]2. Executor 视角[/bold green]\n")

    # Executor 能看到：自己的 Segment + 上游（Planner）的 DOWNSTREAM Segment
    executor_all_segments = executor_segments + [s for s in planner_segments if s.control.visibility in (Visibility.DOWNSTREAM, Visibility.GLOBAL)]

    executor_context = await forge.build(
        system_prompt="",
        messages=[
            {"role": "user", "content": "根据规划，开始执行任务。"},
        ],
        extra_segments=executor_all_segments,
        namespace="executor",
    )

    executor_visible = [s for s in executor_context.segments if s.control.namespace in ("executor", "") or s.control.visibility in (Visibility.DOWNSTREAM, Visibility.GLOBAL)]
    console.print(f"  可见 Segment：[bold]{len(executor_visible)}[/bold] 个（含上游 Planner 的输出）")
    console.print(f"  总 Token：[bold]{format_tokens(executor_context.token_usage.total_tokens)}[/bold]\n")

    # === 组装 Reviewer 的上下文 ===
    console.print("[bold yellow]3. Reviewer 视角[/bold yellow]\n")

    # Reviewer 能看到：自己的 Segment + 上游（Planner + Executor）的 DOWNSTREAM Segment + GLOBAL Segment
    reviewer_all_segments = (
        reviewer_segments +
        [s for s in planner_segments if s.control.visibility in (Visibility.DOWNSTREAM, Visibility.GLOBAL)] +
        [s for s in executor_segments if s.control.visibility in (Visibility.DOWNSTREAM, Visibility.GLOBAL)]
    )

    reviewer_context = await forge.build(
        system_prompt="",
        messages=[
            {"role": "user", "content": "请审核执行结果，提供反馈。"},
        ],
        extra_segments=reviewer_all_segments,
        namespace="reviewer",
    )

    reviewer_visible = [s for s in reviewer_context.segments if s.control.namespace in ("reviewer", "") or s.control.visibility in (Visibility.DOWNSTREAM, Visibility.GLOBAL)]
    console.print(f"  可见 Segment：[bold]{len(reviewer_visible)}[/bold] 个（含上游 Planner 和 Executor 的输出）")
    console.print(f"  总 Token：[bold]{format_tokens(reviewer_context.token_usage.total_tokens)}[/bold]\n")

    print_section("步骤 4：可见性矩阵")

    # 构建可见性矩阵
    visibility_matrix = create_comparison_table(
        "Segment 可见性矩阵",
        ["Segment", "命名空间", "可见性", "Planner", "Executor", "Reviewer"],
        [
            ["Planner 系统提示", "planner", "NAMESPACE", "OK", "X", "X"],
            ["Planner 输出", "planner", "DOWNSTREAM", "OK", "OK", "OK"],
            ["Executor 系统提示", "executor", "NAMESPACE", "X", "OK", "X"],
            ["Executor 状态", "executor", "NAMESPACE", "X", "OK", "X"],
            ["Executor 报告", "executor", "DOWNSTREAM", "X", "OK", "OK"],
            ["Reviewer 系统提示", "reviewer", "NAMESPACE", "X", "X", "OK"],
            ["Reviewer 清单", "reviewer", "NAMESPACE", "X", "X", "OK"],
            ["Reviewer 反馈", "reviewer", "GLOBAL", "OK", "OK", "OK"],
        ]
    )
    console.print(visibility_matrix)
    console.print()

    print_section("步骤 5：Handoff 事件时间线")

    # 收集所有 Handoff 事件
    handoff_events = []
    for segments, agent_name in [
        (planner_segments, "Planner"),
        (executor_segments, "Executor"),
        (reviewer_segments, "Reviewer"),
    ]:
        for seg in segments:
            if seg.control.handoff_to:
                handoff_events.append({
                    "from": agent_name,
                    "to": seg.control.handoff_to,
                    "time": seg.metadata.injected_at if seg.metadata else now,
                    "content": truncate_text(seg.content, 50),
                })

    # 按时间排序
    handoff_events.sort(key=lambda e: e["time"])

    if handoff_events:
        handoff_table = create_comparison_table(
            "Handoff 事件时间线",
            ["时间", "发起方", "接收方", "内容预览"],
            [
                [
                    event["time"].strftime("%H:%M:%S"),
                    event["from"],
                    event["to"],
                    event["content"],
                ]
                for event in handoff_events
            ]
        )
        console.print(handoff_table)
        console.print()

    print_section("步骤 6：发布/订阅模式")

    # 显示 GLOBAL 可见性的 Segment（发布到全局）
    global_segments = []
    for segments, agent_name in [
        (planner_segments, "Planner"),
        (executor_segments, "Executor"),
        (reviewer_segments, "Reviewer"),
    ]:
        for seg in segments:
            if seg.control.visibility == Visibility.GLOBAL or seg.control.publish:
                global_segments.append({
                    "publisher": agent_name,
                    "content": truncate_text(seg.content, 60),
                    "subscribers": "所有 Agent",
                })

    if global_segments:
        publish_table = create_comparison_table(
            "全局发布事件",
            ["发布者", "内容", "订阅者"],
            [
                [event["publisher"], event["content"], event["subscribers"]]
                for event in global_segments
            ]
        )
        console.print(publish_table)
        console.print()

    print_section("总结")

    print_success(f"多 Agent 上下文协调完成！")
    print_success(f"- Planner → Executor → Reviewer 三层协作流程")
    print_success(f"- Namespace 隔离确保各 Agent 上下文独立")
    print_success(f"- DOWNSTREAM 可见性实现上下游信息传递")
    print_success(f"- GLOBAL 可见性实现全局事件广播")
    print_success(f"- Handoff 机制确保任务交接清晰")

    console.print(f"\n[bold]Token 使用汇总：[/bold]")
    console.print(f"  - Planner：{format_tokens(planner_context.token_usage.total_tokens)}")
    console.print(f"  - Executor：{format_tokens(executor_context.token_usage.total_tokens)}")
    console.print(f"  - Reviewer：{format_tokens(reviewer_context.token_usage.total_tokens)}")

    console.print(f"\n[dim]提示：实际生产中建议使用 Context Bus 统一管理多 Agent 上下文（第三轮待实现）[/dim]")


if __name__ == "__main__":
    args = parse_args("场景 3：多 Agent 上下文协调")
    asyncio.run(main(mock=args.mock))
