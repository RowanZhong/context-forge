"""
TracingMiddleware — 可选的 OpenTelemetry 集成。

→ 6.5.5 分布式追踪集成

在微服务架构中,上下文组装往往是整个请求链路的一个环节。
通过集成 OpenTelemetry,可以将上下文组装的各个阶段（Normalize、Sanitize、
Rerank、Allocate、Assemble）作为独立的 Span 记录,并与外部系统（如 API Gateway、
LLM Provider）的 Trace 串联起来,形成完整的分布式调用链。

这个模块是完全可选的：
- 如果用户没有安装 OpenTelemetry,会自动降级为无操作模式
- 如果用户安装了但未配置,会发出 Warning 但不会阻塞流程
- 如果用户配置正确,会自动记录 Trace 并导出到配置的后端（Jaeger/Zipkin 等）

⚠️ 反模式对照:在复杂系统中不提供分布式追踪能力,
排查跨服务的性能问题时只能靠日志拼凑,效率极低。
"""

from __future__ import annotations

import warnings
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from context_forge.models.context_package import ContextPackage

# [Design Decision] 使用 lazy import,避免强依赖 OpenTelemetry
_OTEL_AVAILABLE = False
try:
    from opentelemetry import trace
    from opentelemetry.trace import SpanKind, Status, StatusCode

    _OTEL_AVAILABLE = True
except ImportError:
    # OpenTelemetry 未安装,降级为无操作模式
    pass


class TracingMiddleware:
    """
    OpenTelemetry 追踪中间件。

    → 6.5.5 分布式追踪集成

    基本用法::

        # 创建 Tracer（需要先配置 OpenTelemetry）
        from opentelemetry import trace
        tracer = trace.get_tracer("context_forge")

        middleware = TracingMiddleware(tracer=tracer)

        # 在组装流程中使用
        async with middleware.trace_build(request_id="req_123") as span:
            # 记录输入参数
            middleware.add_event(span, "build_started", {"model": "gpt-4o"})

            # 执行组装...
            package = await forge.build(...)

            # 记录结果
            middleware.record_package(span, package)

    属性:
        tracer: OpenTelemetry Tracer 实例（可选）
        enabled: 是否启用追踪
    """

    def __init__(self, tracer: Any = None) -> None:
        """
        初始化 TracingMiddleware。

        参数:
            tracer: OpenTelemetry Tracer 实例（可选）
        """
        self.tracer = tracer
        self.enabled = _OTEL_AVAILABLE and tracer is not None

        if not _OTEL_AVAILABLE and tracer is not None:
            warnings.warn(
                "OpenTelemetry 未安装,TracingMiddleware 将以无操作模式运行。"
                "如需启用追踪,请安装: pip install opentelemetry-api opentelemetry-sdk"
            )

    @asynccontextmanager
    async def trace_build(
        self,
        request_id: str,
        model: str = "",
    ) -> AsyncIterator[Any]:
        """
        追踪一次完整的 build 操作。

        → 6.5.5.1 Build Span

        参数:
            request_id: 请求 ID
            model: 目标模型 ID

        Yields:
            Span 实例（如果未启用则为 None）
        """
        if not self.enabled:
            yield None
            return

        with self.tracer.start_as_current_span(
            "context_forge.build",
            kind=SpanKind.INTERNAL,
        ) as span:
            span.set_attribute("request_id", request_id)
            if model:
                span.set_attribute("model", model)
            yield span

    @asynccontextmanager
    async def trace_stage(
        self,
        stage_name: str,
        segment_count: int = 0,
    ) -> AsyncIterator[Any]:
        """
        追踪流水线的单个阶段。

        → 6.5.5.2 Stage Span

        参数:
            stage_name: 阶段名称（如 "normalize", "sanitize"）
            segment_count: 输入 Segment 数量

        Yields:
            Span 实例（如果未启用则为 None）
        """
        if not self.enabled:
            yield None
            return

        with self.tracer.start_as_current_span(
            f"context_forge.pipeline.{stage_name}",
            kind=SpanKind.INTERNAL,
        ) as span:
            span.set_attribute("stage_name", stage_name)
            span.set_attribute("input_segment_count", segment_count)
            yield span

    def record_package(self, span: Any, package: ContextPackage) -> None:
        """
        记录 ContextPackage 的关键信息到 Span。

        → 6.5.5.3 Package 属性记录

        参数:
            span: Span 实例
            package: ContextPackage 实例
        """
        if not self.enabled or span is None:
            return

        usage = package.token_usage

        span.set_attribute("package.total_tokens", usage.total_tokens)
        span.set_attribute("package.segment_count", usage.segment_count)
        span.set_attribute("package.dropped_count", len(package.dropped_segments))
        span.set_attribute("package.warning_count", len(package.warnings))
        span.set_attribute("package.assembly_duration_ms", package.assembly_duration_ms)

        # 如果有预算分配
        if package.budget_allocation:
            span.set_attribute("package.total_budget", package.budget_allocation.total_budget)
            span.set_attribute(
                "package.token_utilization",
                usage.total_tokens / package.budget_allocation.total_budget,
            )

    def add_event(
        self,
        span: Any,
        event_name: str,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """
        向 Span 添加事件。

        参数:
            span: Span 实例
            event_name: 事件名称
            attributes: 事件属性
        """
        if not self.enabled or span is None:
            return

        span.add_event(event_name, attributes=attributes or {})

    def set_error(self, span: Any, error: Exception) -> None:
        """
        记录错误到 Span。

        参数:
            span: Span 实例
            error: 异常实例
        """
        if not self.enabled or span is None:
            return

        span.set_status(Status(StatusCode.ERROR, str(error)))
        span.record_exception(error)


# --- 全局单例实例（可选） ---

_global_middleware: TracingMiddleware | None = None


def get_global_middleware() -> TracingMiddleware:
    """
    获取全局 TracingMiddleware 单例实例。

    # [DX Decision] 提供全局单例方便在多个模块中共享追踪中间件。

    返回:
        全局 TracingMiddleware 实例（可能未启用）
    """
    global _global_middleware
    if _global_middleware is None:
        _global_middleware = TracingMiddleware()
    return _global_middleware


def configure_global_middleware(tracer: Any) -> None:
    """
    配置全局 TracingMiddleware 单例实例。

    参数:
        tracer: OpenTelemetry Tracer 实例
    """
    global _global_middleware
    _global_middleware = TracingMiddleware(tracer=tracer)


def reset_global_middleware() -> None:
    """
    重置全局 TracingMiddleware 单例实例。

    在测试场景中使用。
    """
    global _global_middleware
    _global_middleware = None


# --- 辅助函数：自动配置 OpenTelemetry ---


def auto_configure_otel(
    service_name: str = "context_forge",
    exporter_endpoint: str | None = None,
) -> Any | None:
    """
    自动配置 OpenTelemetry（仅在依赖可用时）。

    # 🏭 生产提示：这是简化版配置,生产环境中应使用更完善的配置方案,
    # 包括 Resource 属性、采样策略、批量导出等。

    参数:
        service_name: 服务名称
        exporter_endpoint: 导出端点（如 "http://localhost:4317"）

    返回:
        Tracer 实例,如果 OpenTelemetry 不可用则返回 None
    """
    if not _OTEL_AVAILABLE:
        return None

    try:
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

        # 创建 TracerProvider
        provider = TracerProvider()

        # 配置导出器
        if exporter_endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )

                exporter = OTLPSpanExporter(endpoint=exporter_endpoint)
            except ImportError:
                warnings.warn(
                    "opentelemetry-exporter-otlp 未安装,降级为控制台输出。"
                    "如需 OTLP 导出,请安装: pip install opentelemetry-exporter-otlp"
                )
                exporter = ConsoleSpanExporter()
        else:
            # 默认使用控制台输出
            exporter = ConsoleSpanExporter()

        # 添加批量处理器
        provider.add_span_processor(BatchSpanProcessor(exporter))

        # 设置全局 TracerProvider
        trace.set_tracer_provider(provider)

        # 返回 Tracer
        return trace.get_tracer(service_name)

    except Exception as e:
        warnings.warn(f"OpenTelemetry 自动配置失败: {e}")
        return None
