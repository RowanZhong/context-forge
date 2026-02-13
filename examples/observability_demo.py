"""
可观测性模块演示 — Snapshot、Diff、Golden Set、Metrics。

展示如何使用 Context Forge 的可观测性套件：
1. 保存和加载 Context Snapshot
2. 比对两次上下文组装的差异
3. 运行 Golden Set 回归测试
4. 收集和统计性能指标
"""

import asyncio
from pathlib import Path

from context_forge.models.context_package import ContextPackage
from context_forge.models.segment import Segment, SegmentType
from context_forge.observability import (
    GoldenCase,
    GoldenTolerance,
    create_observability_suite,
)


async def main():
    print("═══ Context Forge 可观测性演示 ═══\n")

    # 创建可观测性套件
    suite = create_observability_suite(
        snapshot_dir="./snapshots_demo",
        metrics_max_points=1000,
        enable_tracing=False,
    )

    # ==================== 场景 1: Context Snapshot ====================
    print("── 场景 1: Context Snapshot ──")

    # 创建示例 ContextPackage
    package1 = ContextPackage(
        segments=[
            Segment(
                type=SegmentType.SYSTEM,
                role="system",
                content="你是一个有帮助的 AI 助手。",
            ),
            Segment(
                type=SegmentType.USER,
                role="user",
                content="介绍一下 Python 的主要特性。",
            ),
        ],
        model="gpt-4o",
        policy_version="v1.0",
        assembly_duration_ms=35.2,
    )

    # 保存 Snapshot
    snapshot_id = await suite.snapshot_manager.save(
        package=package1,
        build_inputs={
            "system_prompt": "你是一个有帮助的 AI 助手。",
            "messages": [{"role": "user", "content": "介绍一下 Python 的主要特性。"}],
        },
        tags={"env": "demo", "scenario": "basic_query"},
    )

    print(f"[OK] 保存 Snapshot: {snapshot_id}")

    # 加载 Snapshot
    snapshot = await suite.snapshot_manager.load(snapshot_id)
    print(f"[OK] 加载 Snapshot: request_id={snapshot.metadata.request_id}")
    print(f"  模型: {snapshot.metadata.model}")
    print(f"  标签: {snapshot.metadata.tags}")
    print()

    # ==================== 场景 2: Prompt Diff ====================
    print("── 场景 2: Prompt Diff ──")

    # 创建修改后的 Package（添加了一条消息）
    package2 = ContextPackage(
        segments=package1.segments
        + [
            Segment(
                type=SegmentType.ASSISTANT,
                role="assistant",
                content="Python 是一种高级编程语言，具有简洁优雅的语法...",
            )
        ],
        model="gpt-4o",
        policy_version="v1.1",
        assembly_duration_ms=42.8,
    )

    # 比对差异
    diff = await suite.diff_engine.diff(package1, package2)

    print(f"[OK] Diff 完成")
    print(f"  汇总: {diff.summary}")
    print()
    print("  变更详情:")
    for entry in diff.entries[:5]:  # 最多显示 5 条
        print(f"    [{entry.diff_type.value}] {entry.description}")
    print()

    # ==================== 场景 3: Golden Set 回归测试 ====================
    print("── 场景 3: Golden Set 回归测试 ──")

    # 创建 mock build 函数
    async def mock_build(**kwargs):
        """模拟上下文组装函数。"""
        segments = []
        if "system_prompt" in kwargs:
            segments.append(
                Segment(
                    type=SegmentType.SYSTEM,
                    role="system",
                    content=kwargs["system_prompt"],
                )
            )
        if "messages" in kwargs:
            for msg in kwargs["messages"]:
                segments.append(
                    Segment(
                        type=SegmentType.USER,
                        role=msg["role"],
                        content=msg["content"],
                    )
                )

        return ContextPackage(
            segments=segments,
            model="gpt-4o",
            assembly_duration_ms=40.0,
        )

    # 添加 Golden Case
    suite.golden_runner.add_case(
        GoldenCase(
            name="basic_query",
            description="基础查询场景",
            build_inputs={
                "system_prompt": "你是一个助手",
                "messages": [{"role": "user", "content": "你好"}],
            },
            expected_outputs={
                "segment_count": 2,
                "dropped_count": 0,
            },
            tolerance=GoldenTolerance(allow_token_delta=0.1),
        )
    )

    # 运行测试
    results = await suite.golden_runner.run(mock_build)

    print(f"[OK] Golden Set 测试完成")
    print(f"  通过: {suite.golden_runner.passed_count(results)}/{len(results)}")

    for result in results:
        status = "[OK]" if result.passed else "[FAIL]"
        print(f"  {status} {result.case.name}: {result.case.description}")

    print()

    # ==================== 场景 4: 指标收集 ====================
    print("── 场景 4: 指标收集 ──")

    # 从 Package 收集指标
    suite.metrics_collector.collect_from_package(package1)
    suite.metrics_collector.collect_from_package(package2)

    # 查看指标
    for metric_name in ["assembly_latency_ms", "total_tokens", "segment_count"]:
        summary = suite.metrics_collector.summary(metric_name)
        if summary:
            print(f"[OK] {metric_name}:")
            print(f"    平均值: {summary.mean:.1f}")
            print(f"    P50: {summary.p50:.1f}")
            print(f"    P95: {summary.p95:.1f}")

    print()

    # ==================== 清理 ====================
    print("── 清理 ──")

    # 删除演示 Snapshot
    await suite.snapshot_manager.delete(snapshot_id)
    print(f"[OK] 已删除 Snapshot: {snapshot_id}")

    # 清理目录
    snapshot_dir = Path("./snapshots_demo")
    if snapshot_dir.exists():
        import shutil

        shutil.rmtree(snapshot_dir)
        print(f"[OK] 已删除目录: {snapshot_dir}")

    print("\n═══ 演示完成 ═══")


if __name__ == "__main__":
    asyncio.run(main())
