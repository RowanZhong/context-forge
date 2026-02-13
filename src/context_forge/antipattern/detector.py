"""
反模式检测器 — 编排所有检测规则。

→ 6.7 反模式检测与诊断

本模块提供 AntiPatternDetector 类，它是反模式检测的主入口。
Detector 负责：
1. 注册和管理所有检测规则
2. 按优先级顺序执行规则
3. 汇总检测结果
4. 生成多格式报告（text/json/rich）

使用示例::

    from context_forge.antipattern import AntiPatternDetector, create_default_detector

    # 使用默认规则集
    detector = create_default_detector()

    # 从 ContextPackage 检测
    results = detector.detect_from_package(context_package)

    # 生成报告
    print(detector.format_report(results, format="text"))

# [Design Decision] 检测器使用插件化架构，
# 所有规则通过 register_rule() 动态注册，方便用户禁用特定规则或添加自定义规则。
"""

from __future__ import annotations

import json
from typing import Any

from context_forge.antipattern.base import (
    AntiPatternRule,
    AntiPatternSeverity,
    DetectionContext,
    DetectionResult,
)
from context_forge.models.context_package import ContextPackage


class AntiPatternDetector:
    """
    反模式检测器 — 编排所有检测规则。

    → 6.7 反模式检测框架

    Detector 管理一组检测规则，按优先级顺序执行，
    汇总结果并生成报告。

    基本用法::

        detector = AntiPatternDetector()
        detector.register_rule(MissingTokenCountRule())
        detector.register_rule(OveruseCriticalRule())

        results = detector.detect(context)
        report = detector.format_report(results, format="text")

    属性:
        rules: 已注册的规则列表
    """

    def __init__(self) -> None:
        """初始化检测器（无规则）。"""
        self._rules: list[AntiPatternRule] = []

    def register_rule(self, rule: AntiPatternRule) -> None:
        """
        注册一个检测规则。

        参数:
            rule: 实现 AntiPatternRule Protocol 的规则实例
        """
        self._rules.append(rule)

    def unregister_rule(self, rule_name: str) -> None:
        """
        注销一个检测规则。

        参数:
            rule_name: 规则名称
        """
        self._rules = [r for r in self._rules if r.name != rule_name]

    def detect(self, context: DetectionContext) -> list[DetectionResult]:
        """
        执行所有注册的检测规则。

        # [Design Decision] 检测器按严重性从高到低执行规则：
        # CRITICAL → WARNING → INFO
        # 这样用户可以优先看到最严重的问题。

        参数:
            context: 检测上下文

        返回:
            所有规则的检测结果列表（按严重性排序）
        """
        all_results: list[DetectionResult] = []

        # 按严重性排序规则（CRITICAL > WARNING > INFO）
        severity_order = {
            AntiPatternSeverity.CRITICAL: 0,
            AntiPatternSeverity.WARNING: 1,
            AntiPatternSeverity.INFO: 2,
        }
        sorted_rules = sorted(self._rules, key=lambda r: severity_order[r.severity])

        # 执行规则
        for rule in sorted_rules:
            try:
                results = rule.detect(context)
                all_results.extend(results)
            except Exception as e:
                # [DX Decision] 检测器不应因单个规则失败而中断。
                # 记录错误并继续执行其他规则。
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"检测规则 {rule.name} 执行失败: {e}",
                    exc_info=True,
                )

        return all_results

    def detect_from_package(
        self,
        package: ContextPackage,
        config: dict[str, Any] | None = None,
    ) -> list[DetectionResult]:
        """
        从 ContextPackage 创建上下文并检测反模式。

        # [DX Decision] 提供便捷方法直接从 ContextPackage 检测，
        # 避免用户手动构建 DetectionContext。

        参数:
            package: 组装完成的 ContextPackage
            config: 额外的配置参数（如阈值覆盖）

        返回:
            检测结果列表
        """
        # 从 package 中提取信息构建 DetectionContext
        context = DetectionContext(
            segments=package.segments,
            budget_policy=None,  # ContextPackage 不直接包含 BudgetPolicy
            budget_allocation=package.budget_allocation,
            audit_log=package.audit_log,
            model=package.model,
            policy_version=package.policy_version,
            config=config or {},
        )

        return self.detect(context)

    def format_report(
        self,
        results: list[DetectionResult],
        format: str = "text",
    ) -> str:
        """
        格式化检测结果为报告。

        → 6.7 反模式检测与诊断

        参数:
            results: 检测结果列表
            format: 输出格式（"text" / "json" / "rich"）

        返回:
            格式化后的报告字符串
        """
        if format == "json":
            return self._format_json(results)
        elif format == "rich":
            return self._format_rich(results)
        else:  # text
            return self._format_text(results)

    def _format_text(self, results: list[DetectionResult]) -> str:
        """格式化为纯文本报告。"""
        if not results:
            return "[OK] 未检测到反模式。"

        lines = ["=" * 70, "反模式检测报告", "=" * 70, ""]

        # 按严重性分组
        by_severity: dict[AntiPatternSeverity, list[DetectionResult]] = {
            AntiPatternSeverity.CRITICAL: [],
            AntiPatternSeverity.WARNING: [],
            AntiPatternSeverity.INFO: [],
        }

        for result in results:
            by_severity[result.severity].append(result)

        # 输出统计
        critical_count = len(by_severity[AntiPatternSeverity.CRITICAL])
        warning_count = len(by_severity[AntiPatternSeverity.WARNING])
        info_count = len(by_severity[AntiPatternSeverity.INFO])

        lines.append(f"检测到 {len(results)} 个问题:")
        lines.append(f"  [!] CRITICAL: {critical_count}")
        lines.append(f"  [!] WARNING:  {warning_count}")
        lines.append(f"  [i] INFO:     {info_count}")
        lines.append("")

        # 输出详细结果
        for severity in [AntiPatternSeverity.CRITICAL, AntiPatternSeverity.WARNING, AntiPatternSeverity.INFO]:
            severity_results = by_severity[severity]
            if not severity_results:
                continue

            lines.append("-" * 70)
            lines.append(f"{severity.value.upper()} 级别问题 ({len(severity_results)} 个)")
            lines.append("-" * 70)
            lines.append("")

            for result in severity_results:
                lines.append(result.format_text())
                lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)

    def _format_json(self, results: list[DetectionResult]) -> str:
        """格式化为 JSON 报告。"""
        report = {
            "total": len(results),
            "by_severity": {
                "critical": len([r for r in results if r.severity == AntiPatternSeverity.CRITICAL]),
                "warning": len([r for r in results if r.severity == AntiPatternSeverity.WARNING]),
                "info": len([r for r in results if r.severity == AntiPatternSeverity.INFO]),
            },
            "results": [
                {
                    "rule_name": r.rule_name,
                    "severity": r.severity.value,
                    "title": r.title,
                    "message": r.message,
                    "why": r.why,
                    "how": r.how,
                    "segment_ids": r.segment_ids,
                    "metadata": r.metadata,
                }
                for r in results
            ],
        }

        return json.dumps(report, ensure_ascii=False, indent=2)

    def _format_rich(self, results: list[DetectionResult]) -> str:
        """
        格式化为 Rich 库兼容的报告。

        # 🏭 生产提示：实际生产中应直接返回 rich.console.Console 对象，
        # 这里简化为返回 Markdown 格式字符串。
        """
        if not results:
            return "[OK] **未检测到反模式**"

        lines = ["# 反模式检测报告\n"]

        # 统计
        critical_count = len([r for r in results if r.severity == AntiPatternSeverity.CRITICAL])
        warning_count = len([r for r in results if r.severity == AntiPatternSeverity.WARNING])
        info_count = len([r for r in results if r.severity == AntiPatternSeverity.INFO])

        lines.append(f"检测到 **{len(results)}** 个问题:\n")
        lines.append(f"- [X] **CRITICAL**: {critical_count}")
        lines.append(f"- [!] **WARNING**: {warning_count}")
        lines.append(f"- [i] **INFO**: {info_count}\n")

        # 详细结果
        for result in results:
            severity_emoji = {
                AntiPatternSeverity.CRITICAL: "[X]",
                AntiPatternSeverity.WARNING: "[!]",
                AntiPatternSeverity.INFO: "[i]",
            }[result.severity]

            lines.append(f"## {severity_emoji} {result.title}\n")
            lines.append(f"**规则**: {result.rule_name}  ")
            lines.append(f"**级别**: {result.severity.value.upper()}\n")
            lines.append(f"**问题**: {result.message}\n")
            lines.append(f"**原因**: {result.why}\n")
            lines.append(f"**修复**: {result.how}\n")

            if result.segment_ids:
                ids_str = ", ".join(f"`{sid}`" for sid in result.segment_ids[:5])
                if len(result.segment_ids) > 5:
                    ids_str += f" ... 及其他 {len(result.segment_ids) - 5} 个"
                lines.append(f"**涉及 Segment**: {ids_str}\n")

            if result.metadata:
                lines.append("**详细信息**:\n")
                for key, value in result.metadata.items():
                    lines.append(f"- {key}: `{value}`")
                lines.append("")

            lines.append("---\n")

        return "\n".join(lines)

    @property
    def rules(self) -> list[AntiPatternRule]:
        """返回已注册的规则列表。"""
        return self._rules.copy()

    def __repr__(self) -> str:
        return f"AntiPatternDetector(rules={len(self._rules)})"


def create_default_detector(config: dict[str, Any] | None = None) -> AntiPatternDetector:
    """
    创建包含所有默认规则的检测器。

    → 6.7 反模式检测框架

    # [DX Decision] 提供工厂函数快速创建检测器，
    # 用户可以直接使用默认规则集，也可以在此基础上调整。

    参数:
        config: 可选的配置字典（如阈值覆盖）

    返回:
        配置好的 AntiPatternDetector 实例

    使用示例::

        # 使用默认配置
        detector = create_default_detector()

        # 覆盖阈值
        detector = create_default_detector(config={
            "critical_ratio_threshold": 0.3,
            "rigid_budget_threshold": 0.6,
        })

        # 禁用特定规则
        detector.unregister_rule("UnusedSanitizerRule")
    """
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

    detector = AntiPatternDetector()

    # 注册所有默认规则（按优先级顺序）
    # CRITICAL 级别
    detector.register_rule(MissingTokenCountRule())
    detector.register_rule(NamespaceLeakageRule())
    detector.register_rule(CircularDependencyRule())

    # WARNING 级别
    detector.register_rule(OveruseCriticalRule())
    detector.register_rule(RigidBudgetTooLargeRule())
    detector.register_rule(ExpiredDataRule())
    detector.register_rule(OverCompressionRule())

    # INFO 级别
    detector.register_rule(IneffectiveRoutingRule())
    detector.register_rule(CacheKeyCollisionRule())
    detector.register_rule(UnusedSanitizerRule())

    return detector
