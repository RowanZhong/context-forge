"""
MetricsCollector — 核心指标收集与统计。

→ 6.5.4 核心指标监控

生产环境中,上下文组装的性能和质量需要持续监控。MetricsCollector 提供了
轻量级的指标收集能力,无需外部依赖即可统计关键指标:

- 性能指标: 组装延迟(P50/P95/P99)、Token 计数耗时
- 质量指标: Token 利用率、压缩比、丢弃数量、告警数量
- 缓存指标: 缓存命中率、命中延迟

指标数据保存在内存中的循环缓冲区(deque with maxlen),避免内存溢出,
适合在生产环境中长期运行。

⚠️ 反模式对照:不收集指标的系统无法发现性能退化和质量下降,
只能在用户投诉后才被动排查,极大降低了系统可靠性。
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from context_forge.models.context_package import ContextPackage


@dataclass
class MetricPoint:
    """
    单个指标数据点。

    # [Design Decision] 不使用 frozen=True,因为需要在收集后更新时间戳。

    属性:
        name: 指标名称
        value: 指标值
        timestamp: 时间戳
        tags: 标签（用于分组和过滤）
    """

    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    tags: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricsSummary:
    """
    指标汇总统计。

    属性:
        metric_name: 指标名称
        count: 数据点数量
        min: 最小值
        max: 最大值
        mean: 平均值
        p50: P50 百分位数（中位数）
        p95: P95 百分位数
        p99: P99 百分位数
    """

    metric_name: str
    count: int
    min: float
    max: float
    mean: float
    p50: float
    p95: float
    p99: float


class MetricsCollector:
    """
    指标收集器 — 收集和统计上下文组装的核心指标。

    → 6.5.4 核心指标监控

    基本用法::

        collector = MetricsCollector(max_points=10000)

        # 记录性能指标
        collector.record("assembly_latency_ms", 42.5, tags={"model": "gpt-4o"})

        # 从 ContextPackage 自动提取指标
        collector.collect_from_package(package)

        # 获取统计信息
        summary = collector.summary("assembly_latency_ms")
        print(f"P99 延迟: {summary.p99:.1f}ms")

        # 导出所有指标
        metrics = collector.export()

    属性:
        max_points: 每个指标保留的最大数据点数量
        metrics: 指标存储（指标名 -> deque）
    """

    def __init__(self, max_points: int = 10000) -> None:
        """
        初始化 MetricsCollector。

        参数:
            max_points: 每个指标保留的最大数据点数量（循环缓冲区大小）
        """
        self.max_points = max_points
        self.metrics: dict[str, deque[MetricPoint]] = {}

    def record(
        self,
        name: str,
        value: float,
        tags: dict[str, str] | None = None,
    ) -> None:
        """
        记录一个指标数据点。

        参数:
            name: 指标名称
            value: 指标值
            tags: 标签（用于分组和过滤）
        """
        if name not in self.metrics:
            self.metrics[name] = deque(maxlen=self.max_points)

        point = MetricPoint(
            name=name,
            value=value,
            tags=tags or {},
        )

        self.metrics[name].append(point)

    def collect_from_package(self, package: ContextPackage) -> None:
        """
        从 ContextPackage 自动提取并记录指标。

        → 6.5.4.1 自动指标提取

        提取的指标包括:
        - assembly_latency_ms: 组装延迟
        - total_tokens: 总 Token 数
        - segment_count: Segment 数量
        - dropped_count: 丢弃数量
        - warning_count: 警告数量
        - token_utilization: Token 利用率（实际使用 / 预算）

        参数:
            package: ContextPackage 实例
        """
        tags = {
            "model": package.model,
            "policy_version": package.policy_version,
        }

        # 组装延迟
        self.record("assembly_latency_ms", package.assembly_duration_ms, tags=tags)

        # Token 统计
        usage = package.token_usage
        self.record("total_tokens", float(usage.total_tokens), tags=tags)
        self.record("segment_count", float(usage.segment_count), tags=tags)

        # 质量指标
        self.record("dropped_count", float(len(package.dropped_segments)), tags=tags)
        self.record("warning_count", float(len(package.warnings)), tags=tags)

        # Token 利用率
        if package.budget_allocation:
            total_budget = package.budget_allocation.total_budget
            if total_budget > 0:
                utilization = usage.total_tokens / total_budget
                self.record("token_utilization", utilization, tags=tags)

        # 压缩比（如果有截断/压缩）
        truncated_count = len(package.truncated_segments)
        if truncated_count > 0:
            self.record("truncated_count", float(truncated_count), tags=tags)

    def summary(
        self,
        name: str,
        tags: dict[str, str] | None = None,
    ) -> MetricsSummary | None:
        """
        获取指标的汇总统计。

        → 6.5.4.2 百分位数计算

        参数:
            name: 指标名称
            tags: 标签过滤条件（只统计匹配的数据点）

        返回:
            MetricsSummary 实例,如果指标不存在则返回 None
        """
        if name not in self.metrics:
            return None

        # 过滤数据点
        points = self.metrics[name]
        if tags:
            points = deque(p for p in points if self._match_tags(p.tags, tags))

        if not points:
            return None

        values = sorted(p.value for p in points)
        count = len(values)

        return MetricsSummary(
            metric_name=name,
            count=count,
            min=values[0],
            max=values[-1],
            mean=sum(values) / count,
            p50=self._percentile(values, 0.50),
            p95=self._percentile(values, 0.95),
            p99=self._percentile(values, 0.99),
        )

    def export(self) -> dict[str, list[dict[str, Any]]]:
        """
        导出所有指标数据。

        返回:
            指标字典（指标名 -> 数据点列表）
        """
        result: dict[str, list[dict[str, Any]]] = {}

        for name, points in self.metrics.items():
            result[name] = [
                {
                    "value": p.value,
                    "timestamp": p.timestamp,
                    "tags": p.tags,
                }
                for p in points
            ]

        return result

    def reset(self) -> None:
        """清空所有指标数据。"""
        self.metrics.clear()

    def get_metric_names(self) -> list[str]:
        """获取所有已记录的指标名称。"""
        return list(self.metrics.keys())

    def get_point_count(self, name: str) -> int:
        """
        获取指定指标的数据点数量。

        参数:
            name: 指标名称

        返回:
            数据点数量
        """
        if name not in self.metrics:
            return 0
        return len(self.metrics[name])

    # --- 内部方法 ---

    def _percentile(self, values: list[float], p: float) -> float:
        """
        计算百分位数。

        # [Design Decision] 使用简单的线性插值法,无需外部依赖。

        参数:
            values: 已排序的数值列表
            p: 百分位数（0.0 ~ 1.0）

        返回:
            百分位数值
        """
        if not values:
            return 0.0

        if p <= 0:
            return values[0]
        if p >= 1:
            return values[-1]

        index = p * (len(values) - 1)
        lower_index = int(index)
        upper_index = lower_index + 1

        if upper_index >= len(values):
            return values[lower_index]

        # 线性插值
        fraction = index - lower_index
        return values[lower_index] * (1 - fraction) + values[upper_index] * fraction

    def _match_tags(self, point_tags: dict[str, str], filter_tags: dict[str, str]) -> bool:
        """检查数据点的标签是否匹配过滤条件。"""
        return all(point_tags.get(k) == v for k, v in filter_tags.items())


# --- 全局单例实例（可选） ---

_global_collector: MetricsCollector | None = None


def get_global_collector() -> MetricsCollector:
    """
    获取全局 MetricsCollector 单例实例。

    # [DX Decision] 提供全局单例方便在多个模块中共享指标收集器,
    # 避免在每个函数调用中传递 collector 参数。

    返回:
        全局 MetricsCollector 实例
    """
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector


def reset_global_collector() -> None:
    """
    重置全局 MetricsCollector 单例实例。

    在测试场景中使用,确保每个测试用例使用独立的指标收集器。
    """
    global _global_collector
    _global_collector = None
