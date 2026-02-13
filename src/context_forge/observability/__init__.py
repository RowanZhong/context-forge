"""
可观测性模块 — Context Snapshot、Prompt Diff、Golden Set 回归测试、指标监控。

→ 6.5 上下文可观测性：Snapshot、Diff、Golden Set 与核心指标

这个模块提供了完整的可观测性解决方案,解决生产环境中的核心问题:

1. **Context Snapshot（快照）** — 将每次上下文组装的完整状态持久化存储,
   支持后续回放、对比、回归测试。(→ 6.5.1)

2. **Prompt Diff（差异对比）** — 结构化地对比两次上下文组装的差异,
   快速识别策略修改的影响。(→ 6.5.2)

3. **Golden Set（回归测试）** — 建立一组代表性测试用例,
   在修改策略或升级版本后自动运行,确保关键指标在容差范围内。(→ 6.5.3)

4. **MetricsCollector（指标收集）** — 轻量级的指标收集和统计,
   无需外部依赖即可监控性能和质量指标。(→ 6.5.4)

5. **TracingMiddleware（分布式追踪）** — 可选的 OpenTelemetry 集成,
   将上下文组装的各个阶段纳入分布式调用链。(→ 6.5.5)

基本用法::

    from context_forge.observability import create_observability_suite

    # 创建完整的可观测性套件
    suite = create_observability_suite(
        snapshot_dir="./snapshots",
        enable_metrics=True,
        enable_tracing=False,
    )

    # 使用 Snapshot
    snapshot_id = await suite.snapshot_manager.save(package)

    # 使用 Diff
    diff = await suite.diff_engine.diff(old_package, new_package)

    # 使用 Golden Set
    results = await suite.golden_runner.run(forge.build)

    # 使用 Metrics
    suite.metrics_collector.collect_from_package(package)

模块结构:
- snapshot.py: SnapshotManager、Snapshot、SnapshotMetadata
- diff.py: DiffEngine、ContextDiff、DiffEntry
- golden_set.py: GoldenSetRunner、GoldenCase、GoldenResult、GoldenTolerance
- metrics.py: MetricsCollector、MetricsSummary
- tracing.py: TracingMiddleware（可选 OpenTelemetry 集成）
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from context_forge.observability.diff import ContextDiff, DiffEngine, DiffEntry, DiffType
from context_forge.observability.golden_set import (
    AssertionResult,
    ComparisonOperator,
    GoldenCase,
    GoldenResult,
    GoldenSetRunner,
    GoldenTolerance,
)
from context_forge.observability.metrics import (
    MetricPoint,
    MetricsCollector,
    MetricsSummary,
    get_global_collector,
    reset_global_collector,
)
from context_forge.observability.snapshot import (
    Snapshot,
    SnapshotManager,
    SnapshotMetadata,
)
from context_forge.observability.tracing import (
    TracingMiddleware,
    auto_configure_otel,
    configure_global_middleware,
    get_global_middleware,
    reset_global_middleware,
)

__all__ = [
    # Snapshot
    "Snapshot",
    "SnapshotMetadata",
    "SnapshotManager",
    # Diff
    "DiffType",
    "DiffEntry",
    "ContextDiff",
    "DiffEngine",
    # Golden Set
    "ComparisonOperator",
    "GoldenTolerance",
    "GoldenCase",
    "AssertionResult",
    "GoldenResult",
    "GoldenSetRunner",
    # Metrics
    "MetricPoint",
    "MetricsSummary",
    "MetricsCollector",
    "get_global_collector",
    "reset_global_collector",
    # Tracing
    "TracingMiddleware",
    "get_global_middleware",
    "configure_global_middleware",
    "reset_global_middleware",
    "auto_configure_otel",
    # Factory
    "ObservabilitySuite",
    "create_observability_suite",
]


@dataclass
class ObservabilitySuite:
    """
    可观测性套件 — 包含所有可观测性组件的容器。

    → 6.5 上下文可观测性

    # [DX Decision] 提供一站式的可观测性套件,
    # 用户无需单独初始化各个组件。

    属性:
        snapshot_manager: Snapshot 管理器
        diff_engine: Diff 引擎
        golden_runner: Golden Set 运行器
        metrics_collector: 指标收集器
        tracing_middleware: 追踪中间件
    """

    snapshot_manager: SnapshotManager
    diff_engine: DiffEngine
    golden_runner: GoldenSetRunner
    metrics_collector: MetricsCollector
    tracing_middleware: TracingMiddleware


def create_observability_suite(
    snapshot_dir: str | Path = "./snapshots",
    snapshot_auto_cleanup_days: int = 0,
    diff_ignore_fields: list[str] | None = None,
    metrics_max_points: int = 10000,
    enable_tracing: bool = False,
    tracing_service_name: str = "context_forge",
    tracing_exporter_endpoint: str | None = None,
) -> ObservabilitySuite:
    """
    创建完整的可观测性套件。

    → 6.5 上下文可观测性

    # [DX Decision] 这是推荐的初始化方式,
    # 一次性配置所有可观测性组件。

    参数:
        snapshot_dir: Snapshot 存储目录
        snapshot_auto_cleanup_days: 自动清理 N 天前的 Snapshot（0=不清理）
        diff_ignore_fields: Diff 时忽略的字段列表
        metrics_max_points: 每个指标保留的最大数据点数量
        enable_tracing: 是否启用分布式追踪
        tracing_service_name: 追踪服务名称
        tracing_exporter_endpoint: 追踪导出端点（如 "http://localhost:4317"）

    返回:
        ObservabilitySuite 实例

    用法::

        # 最简配置（仅本地 Snapshot + Metrics）
        suite = create_observability_suite()

        # 生产配置（启用追踪）
        suite = create_observability_suite(
            snapshot_dir="/data/snapshots",
            snapshot_auto_cleanup_days=7,
            enable_tracing=True,
            tracing_exporter_endpoint="http://jaeger:4317",
        )
    """
    # 创建 SnapshotManager
    snapshot_manager = SnapshotManager(
        storage_dir=snapshot_dir,
        auto_cleanup_days=snapshot_auto_cleanup_days,
    )

    # 创建 DiffEngine
    diff_engine = DiffEngine(ignore_fields=diff_ignore_fields)

    # 创建 GoldenSetRunner
    golden_runner = GoldenSetRunner()

    # 创建 MetricsCollector
    metrics_collector = MetricsCollector(max_points=metrics_max_points)

    # 创建 TracingMiddleware
    if enable_tracing:
        tracer = auto_configure_otel(
            service_name=tracing_service_name,
            exporter_endpoint=tracing_exporter_endpoint,
        )
        tracing_middleware = TracingMiddleware(tracer=tracer)
    else:
        tracing_middleware = TracingMiddleware()

    return ObservabilitySuite(
        snapshot_manager=snapshot_manager,
        diff_engine=diff_engine,
        golden_runner=golden_runner,
        metrics_collector=metrics_collector,
        tracing_middleware=tracing_middleware,
    )
