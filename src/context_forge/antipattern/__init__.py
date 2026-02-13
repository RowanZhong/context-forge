"""
反模式检测模块。

→ 6.7 反模式检测与诊断

本模块提供反模式检测功能，帮助开发者识别常见的上下文组装问题。

核心组件：
- AntiPatternDetector: 检测器主类
- AntiPatternRule: 检测规则 Protocol
- DetectionResult: 检测结果
- DetectionContext: 检测上下文
- create_default_detector: 工厂函数

检测规则（10 个）：
- MissingTokenCountRule (CRITICAL): 缺失 Token 计数
- NamespaceLeakageRule (CRITICAL): 命名空间泄漏
- CircularDependencyRule (CRITICAL): 循环依赖
- OveruseCriticalRule (WARNING): CRITICAL 优先级滥用
- RigidBudgetTooLargeRule (WARNING): 刚性预算过大
- ExpiredDataRule (WARNING): 过期数据未清理
- OverCompressionRule (WARNING): 过度压缩
- IneffectiveRoutingRule (INFO): 无效的路由决策
- CacheKeyCollisionRule (INFO): 缓存键冲突风险
- UnusedSanitizerRule (INFO): 未使用的清洗规则

使用示例::

    from context_forge import ContextForge
    from context_forge.antipattern import create_default_detector

    # 方式 1: 通过 Facade API
    forge = ContextForge(model="gpt-4o")
    package = await forge.build(...)
    report = forge.detect_antipatterns(package, format="text")
    print(report)

    # 方式 2: 直接使用 Detector
    detector = create_default_detector()
    results = detector.detect_from_package(package)
    print(detector.format_report(results, format="text"))

    # 方式 3: 自定义规则
    detector = AntiPatternDetector()
    detector.register_rule(MissingTokenCountRule())
    detector.register_rule(MyCustomRule())
    results = detector.detect(context)
"""

from context_forge.antipattern.base import (
    AntiPatternRule,
    AntiPatternSeverity,
    DetectionContext,
    DetectionResult,
)
from context_forge.antipattern.detector import (
    AntiPatternDetector,
    create_default_detector,
)
from context_forge.antipattern.rules import (
    CacheKeyCollisionRule,
    CircularDependencyRule,
    ExpiredDataRule,
    IneffectiveRoutingRule,
    MissingTokenCountRule,
    NamespaceLeakageRule,
    OverCompressionRule,
    OveruseCriticalRule,
    RigidBudgetTooLargeRule,
    UnusedSanitizerRule,
)

__all__ = [
    # 核心类型
    "AntiPatternDetector",
    "AntiPatternRule",
    "AntiPatternSeverity",
    "DetectionContext",
    "DetectionResult",
    # 工厂函数
    "create_default_detector",
    # 所有检测规则
    "MissingTokenCountRule",
    "NamespaceLeakageRule",
    "CircularDependencyRule",
    "OveruseCriticalRule",
    "RigidBudgetTooLargeRule",
    "ExpiredDataRule",
    "OverCompressionRule",
    "IneffectiveRoutingRule",
    "CacheKeyCollisionRule",
    "UnusedSanitizerRule",
]
