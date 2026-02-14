"""
反模式检测基础类型与协议。

→ 6.7 反模式检测与诊断

本模块定义反模式检测器的核心抽象：
- AntiPatternSeverity: 严重性级别（INFO/WARNING/CRITICAL）
- DetectionResult: 单条检测结果（frozen dataclass）
- DetectionContext: 检测上下文（包含 Segment 列表、预算、审计日志等）
- AntiPatternRule: 检测规则 Protocol（所有规则实现的接口）

设计理念：**规则即插件**。每个反模式检测规则是一个独立的可插拔组件，
遵循统一的 Protocol 接口。这允许用户禁用特定规则、调整阈值，
或注册自定义的反模式检测器。

# [Design Decision] 使用 Protocol 而非抽象类，
# 遵循 Python 的"鸭子类型"理念，降低耦合度。
# 用户可以实现自己的规则类，只要符合 Protocol 即可注册。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from context_forge.models.audit import AuditEntry
    from context_forge.models.budget import BudgetAllocation, BudgetPolicy
    from context_forge.models.segment import Segment


class AntiPatternSeverity(str, Enum):
    """
    反模式严重性级别。

    → 6.7.1 反模式分类与严重性

    三级严重性分类：
    - INFO: 优化建议，不影响正确性
    - WARNING: 潜在问题，可能导致效率低下或成本浪费
    - CRITICAL: 严重问题，可能导致功能异常或安全风险
    """

    INFO = "info"
    """信息级别 — 优化建议，不影响功能正确性"""

    WARNING = "warning"
    """警告级别 — 潜在问题，可能影响性能或成本"""

    CRITICAL = "critical"
    """严重级别 — 重大问题，可能导致功能异常或安全风险"""


@dataclass(frozen=True)
class DetectionResult:
    """
    单条反模式检测结果。

    → 6.7.2 ~ 6.7.4 各反模式规则

    每个检测规则返回 0 到多个 DetectionResult，每个结果描述一个发现的反模式实例。

    属性:
        rule_name: 规则名称（如 "MissingTokenCountRule"）
        severity: 严重性级别
        title: 反模式标题（简短描述，如 "缺失 Token 计数"）
        message: 详细诊断信息（What: 发生了什么问题）
        why: 为什么这是个问题（Why: 影响和后果）
        how: 如何修复（How: 具体的修复建议）
        segment_ids: 涉及的 Segment ID 列表
        metadata: 额外的诊断数据（如具体阈值、计算值等）
    """

    rule_name: str
    severity: AntiPatternSeverity
    title: str
    message: str
    why: str
    how: str
    segment_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def format_text(self) -> str:
        """格式化为文本报告（用于终端输出）。"""
        severity_symbol = {
            AntiPatternSeverity.INFO: "[i]",
            AntiPatternSeverity.WARNING: "[!]",
            AntiPatternSeverity.CRITICAL: "[X]",
        }[self.severity]

        lines = [
            f"{severity_symbol} [{self.severity.value.upper()}] {self.title}",
            f"   规则: {self.rule_name}",
            "",
            f"   问题: {self.message}",
            f"   原因: {self.why}",
            f"   修复: {self.how}",
        ]

        if self.segment_ids:
            lines.append(f"   涉及 Segment: {', '.join(self.segment_ids[:5])}")
            if len(self.segment_ids) > 5:
                lines.append(f"                  ... 及其他 {len(self.segment_ids) - 5} 个")

        if self.metadata:
            lines.append("   详细信息:")
            for key, value in self.metadata.items():
                lines.append(f"     - {key}: {value}")

        return "\n".join(lines)


@dataclass(frozen=True)
class DetectionContext:
    """
    反模式检测上下文 — 传递给检测规则的输入数据。

    → 6.7 反模式检测框架

    DetectionContext 封装了检测规则需要的所有输入信息，
    包括 Segment 列表、预算策略、审计日志等。

    # [DX Decision] 使用独立的上下文对象而非直接传递 ContextPackage，
    # 这样规则只能访问检测所需的信息，避免耦合到 ContextPackage 的内部实现。

    属性:
        segments: 最终保留的 Segment 列表
        budget_policy: 预算策略配置
        budget_allocation: 实际的预算分配结果
        audit_log: 完整的审计日志
        model: 目标模型名称
        policy_version: 策略版本号
        config: 反模式检测配置
    """

    segments: list[Segment]
    budget_policy: BudgetPolicy | None = None
    budget_allocation: BudgetAllocation | None = None
    audit_log: list[AuditEntry] = field(default_factory=list)
    model: str = ""
    policy_version: str = "1.0"
    config: dict[str, Any] = field(default_factory=dict)


class AntiPatternRule(Protocol):
    """
    反模式检测规则 Protocol。

    → 6.7 反模式检测框架

    所有反模式检测规则必须实现此 Protocol。
    每个规则负责检测一种特定的反模式，返回 0 到多个 DetectionResult。

    基本实现示例::

        class MyCustomRule:
            @property
            def name(self) -> str:
                return "MyCustomRule"

            @property
            def severity(self) -> AntiPatternSeverity:
                return AntiPatternSeverity.WARNING

            def detect(self, context: DetectionContext) -> list[DetectionResult]:
                # 实现检测逻辑
                if some_condition:
                    return [DetectionResult(
                        rule_name=self.name,
                        severity=self.severity,
                        title="发现某反模式",
                        message="描述问题",
                        why="为什么有问题",
                        how="如何修复",
                    )]
                return []

    注意事项:
    - 规则必须处理缺失数据（如 budget_allocation=None）的情况
    - 返回空列表表示未检测到问题，不要抛异常
    - 检测逻辑应保守，避免误报
    """

    @property
    def name(self) -> str:
        """规则名称（唯一标识符）。"""
        ...

    @property
    def severity(self) -> AntiPatternSeverity:
        """此规则检测到的反模式的默认严重性级别。"""
        ...

    def detect(self, context: DetectionContext) -> list[DetectionResult]:
        """
        执行检测。

        参数:
            context: 检测上下文

        返回:
            检测结果列表（空列表表示未检测到问题）
        """
        ...
