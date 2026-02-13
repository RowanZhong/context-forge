"""
Context Forge — 高性能动态上下文组装引擎。

Context Forge 之于 LLM 应用，就像 ORM 之于数据库应用。
它把散落在业务代码中的字符串拼接式上下文组装，
提升为一个声明式的、可配置的、可观测的工程层。

快速上手::

    from context_forge import ContextForge

    forge = ContextForge(model="gpt-4o")
    context = await forge.build(
        system_prompt="你是一个有用的助手。",
        messages=[{"role": "user", "content": "你好"}],
    )
    messages = context.to_messages()  # → 直接传给 LLM API

同步用法::

    context = forge.build_sync(
        system_prompt="你是一个有用的助手。",
        messages=[{"role": "user", "content": "你好"}],
    )

文档：https://context-forge.github.io
源码：https://github.com/context-forge/context-forge
"""

# 第四轮新增导出：反模式检测
from context_forge.antipattern import (
    AntiPatternDetector,
    AntiPatternRule,
    AntiPatternSeverity,
    DetectionContext,
    DetectionResult,
    create_default_detector,
)
from context_forge.cache import CacheManager, MemoryCache

# 第三轮新增导出：压缩、缓存、路由、可观测性
from context_forge.compress import (
    CompressEngine,
    CompressionResult,
    Compressor,
    TruncationCompressor,
)
from context_forge.facade import ContextForge
from context_forge.models import (
    AuditEntry,
    BudgetAllocation,
    BudgetPolicy,
    ContextPackage,
    ControlFlags,
    DecisionType,
    Priority,
    Provenance,
    ReasonCode,
    Segment,
    SegmentMetadata,
    SegmentType,
    SourceType,
    TokenUsage,
    Visibility,
)
from context_forge.observability import (
    DiffEngine,
    GoldenSetRunner,
    MetricsCollector,
    SnapshotManager,
)
from context_forge.routing import ContextBus, Router, RuleBasedRouter

__version__ = "0.1.0"

__all__ = [
    # 顶层入口
    "ContextForge",
    # 数据模型
    "Segment",
    "SegmentType",
    "Priority",
    "Provenance",
    "SourceType",
    "ControlFlags",
    "Visibility",
    "SegmentMetadata",
    "ContextPackage",
    "TokenUsage",
    "BudgetPolicy",
    "BudgetAllocation",
    "AuditEntry",
    "DecisionType",
    "ReasonCode",
    # 压缩模块（第三轮）
    "Compressor",
    "CompressionResult",
    "TruncationCompressor",
    "CompressEngine",
    # 缓存模块（第三轮）
    "CacheManager",
    "MemoryCache",
    # 路由模块（第三轮）
    "Router",
    "RuleBasedRouter",
    "ContextBus",
    # 可观测性模块（第三轮）
    "SnapshotManager",
    "DiffEngine",
    "GoldenSetRunner",
    "MetricsCollector",
    # 反模式检测模块（第四轮）
    "AntiPatternDetector",
    "AntiPatternRule",
    "AntiPatternSeverity",
    "DetectionContext",
    "DetectionResult",
    "create_default_detector",
    # 版本
    "__version__",
]
