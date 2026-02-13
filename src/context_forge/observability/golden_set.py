"""
GoldenSetRunner — 回归测试框架。

→ 6.5.3 Golden Set 回归测试

Golden Set 是一组精心设计的测试用例,每个用例包含输入参数和期望的输出特征。
在修改上下文组装策略或升级引擎版本后,通过运行 Golden Set 可以快速发现回归问题。

这解决了生产环境中最常见的风险:"修改了一个策略参数,不知道对现有场景有什么影响。"
通过 Golden Set,你可以建立一组代表性场景(如"RAG 查询"、"多轮对话"、"Tool Calling"等),
每次修改后自动运行,确保关键指标(Token 使用、丢弃数量、缓存命中率等)在容差范围内。

⚠️ 反模式对照:不建立回归测试集的系统在修改策略时只能靠人工抽查,
既耗时又容易遗漏边界情况,导致上线后才发现问题。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from context_forge.models.context_package import ContextPackage


class ComparisonOperator(str, Enum):
    """比较操作符。"""

    EQUAL = "equal"
    """相等"""

    LESS_THAN = "less_than"
    """小于"""

    LESS_EQUAL = "less_equal"
    """小于等于"""

    GREATER_THAN = "greater_than"
    """大于"""

    GREATER_EQUAL = "greater_equal"
    """大于等于"""

    IN_RANGE = "in_range"
    """在范围内（百分比容差）"""


@dataclass(frozen=True)
class GoldenTolerance:
    """
    Golden Case 的容差配置。

    # [Design Decision] 使用 frozen dataclass 保证不可变性。

    用于定义"什么程度的差异是可接受的"。例如:
    - Token 总数允许 ±5% 浮动
    - 丢弃数量必须完全相等
    - Segment 顺序可以不同

    属性:
        allow_token_delta: 允许的 Token 数量差异（百分比,0.05 = 5%）
        allow_reordering: 是否允许 Segment 重排
        ignore_fields: 忽略的字段列表（如 request_id, created_at）
        custom_assertions: 自定义断言函数列表
    """

    allow_token_delta: float = 0.05
    allow_reordering: bool = False
    ignore_fields: list[str] = field(default_factory=lambda: ["request_id", "created_at"])
    custom_assertions: list[Callable[[ContextPackage], bool]] = field(default_factory=list)


@dataclass(frozen=True)
class GoldenCase:
    """
    单个 Golden 测试用例。

    → 6.5.3.1 Golden Case 定义

    每个 Golden Case 包含:
    1. 输入参数（用于调用 ContextForge.build()）
    2. 期望的输出特征（Token 数量、Segment 类型分布等）
    3. 容差配置（允许的偏差范围）

    属性:
        name: 用例名称
        description: 用例描述
        build_inputs: 构建输入参数
        expected_outputs: 期望的输出特征
        tolerance: 容差配置
        tags: 用例标签（用于分类和过滤）
    """

    name: str
    description: str
    build_inputs: dict[str, Any]
    expected_outputs: dict[str, Any]
    tolerance: GoldenTolerance = field(default_factory=GoldenTolerance)
    tags: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AssertionResult:
    """
    单个断言的结果。

    属性:
        assertion_name: 断言名称
        passed: 是否通过
        expected: 期望值
        actual: 实际值
        message: 详细信息
    """

    assertion_name: str
    passed: bool
    expected: Any
    actual: Any
    message: str


@dataclass(frozen=True)
class GoldenResult:
    """
    Golden Case 的执行结果。

    → 6.5.3.2 Golden 运行结果

    属性:
        case: Golden Case 实例
        package: 实际生成的 ContextPackage
        assertions: 断言结果列表
        passed: 是否全部通过
        error: 执行错误信息（如果有）
    """

    case: GoldenCase
    package: ContextPackage | None
    assertions: list[AssertionResult]
    passed: bool
    error: str = ""


class GoldenSetRunner:
    """
    Golden Set 回归测试运行器。

    → 6.5.3 Golden Set 回归测试

    基本用法::

        runner = GoldenSetRunner()

        # 添加 Golden Case
        runner.add_case(
            GoldenCase(
                name="basic_rag_query",
                description="基础 RAG 查询场景",
                build_inputs={
                    "system_prompt": "你是一个助手",
                    "messages": [{"role": "user", "content": "介绍一下 Python"}],
                    "rag_chunks": ["Python 是一种编程语言...", "..."],
                },
                expected_outputs={
                    "total_tokens": 500,
                    "segment_count": 5,
                    "dropped_count": 0,
                },
                tolerance=GoldenTolerance(allow_token_delta=0.1),
            )
        )

        # 运行测试
        results = await runner.run(forge)

        # 查看结果
        print(f"通过: {runner.passed_count(results)}/{len(results)}")

    属性:
        cases: Golden Case 列表
    """

    def __init__(self) -> None:
        """初始化 GoldenSetRunner。"""
        self.cases: list[GoldenCase] = []

    def add_case(self, case: GoldenCase) -> None:
        """
        添加 Golden Case。

        参数:
            case: Golden Case 实例
        """
        self.cases.append(case)

    def add_cases(self, cases: list[GoldenCase]) -> None:
        """
        批量添加 Golden Case。

        参数:
            cases: Golden Case 列表
        """
        self.cases.extend(cases)

    async def run(
        self,
        build_fn: Callable[[dict[str, Any]], Any],
        filter_tags: dict[str, str] | None = None,
    ) -> list[GoldenResult]:
        """
        运行 Golden Set 测试。

        → 6.5.3.2 Golden 运行与比对

        参数:
            build_fn: 构建函数（通常是 ContextForge.build）
            filter_tags: 标签过滤条件（只运行匹配的用例）

        返回:
            GoldenResult 列表
        """
        results: list[GoldenResult] = []

        for case in self.cases:
            # 应用标签过滤
            if filter_tags and not self._match_tags(case.tags, filter_tags):
                continue

            result = await self._run_case(case, build_fn)
            results.append(result)

        return results

    def passed_count(self, results: list[GoldenResult]) -> int:
        """
        统计通过的用例数量。

        参数:
            results: GoldenResult 列表

        返回:
            通过的用例数量
        """
        return sum(1 for r in results if r.passed)

    def failed_cases(self, results: list[GoldenResult]) -> list[GoldenResult]:
        """
        获取失败的用例。

        参数:
            results: GoldenResult 列表

        返回:
            失败的 GoldenResult 列表
        """
        return [r for r in results if not r.passed]

    def summary(self, results: list[GoldenResult]) -> str:
        """
        生成人类可读的测试摘要。

        参数:
            results: GoldenResult 列表

        返回:
            格式化后的摘要文本
        """
        total = len(results)
        passed = self.passed_count(results)
        failed = total - passed

        lines = [
            "═══ Golden Set 测试结果 ═══",
            f"总数: {total}",
            f"✓ 通过: {passed}",
            f"✗ 失败: {failed}",
        ]

        if failed > 0:
            lines.append("")
            lines.append("── 失败的用例 ──")
            for result in self.failed_cases(results):
                lines.append(f"  ✗ {result.case.name}: {result.case.description}")
                if result.error:
                    lines.append(f"    错误: {result.error}")
                for assertion in result.assertions:
                    if not assertion.passed:
                        lines.append(f"    - {assertion.assertion_name}: {assertion.message}")

        return "\n".join(lines)

    # --- 内部方法 ---

    async def _run_case(
        self,
        case: GoldenCase,
        build_fn: Callable[[dict[str, Any]], Any],
    ) -> GoldenResult:
        """运行单个 Golden Case。"""
        try:
            # 调用构建函数
            package = await build_fn(**case.build_inputs)

            # 执行断言
            assertions = self._assert_outputs(package, case.expected_outputs, case.tolerance)

            # 判断是否全部通过
            passed = all(a.passed for a in assertions)

            return GoldenResult(
                case=case,
                package=package,
                assertions=assertions,
                passed=passed,
            )

        except Exception as e:
            # 构建失败
            return GoldenResult(
                case=case,
                package=None,
                assertions=[],
                passed=False,
                error=str(e),
            )

    def _assert_outputs(
        self,
        package: ContextPackage,
        expected: dict[str, Any],
        tolerance: GoldenTolerance,
    ) -> list[AssertionResult]:
        """执行输出断言。"""
        assertions: list[AssertionResult] = []

        # 断言: total_tokens
        if "total_tokens" in expected:
            expected_tokens = expected["total_tokens"]
            actual_tokens = package.token_usage.total_tokens

            passed, message = self._compare_with_tolerance(
                expected_tokens,
                actual_tokens,
                tolerance.allow_token_delta,
            )

            assertions.append(
                AssertionResult(
                    assertion_name="total_tokens",
                    passed=passed,
                    expected=expected_tokens,
                    actual=actual_tokens,
                    message=message,
                )
            )

        # 断言: segment_count
        if "segment_count" in expected:
            expected_count = expected["segment_count"]
            actual_count = len(package.segments)

            passed = expected_count == actual_count
            message = (
                "符合预期"
                if passed
                else f"期望 {expected_count} 个 Segment,实际 {actual_count} 个"
            )

            assertions.append(
                AssertionResult(
                    assertion_name="segment_count",
                    passed=passed,
                    expected=expected_count,
                    actual=actual_count,
                    message=message,
                )
            )

        # 断言: dropped_count
        if "dropped_count" in expected:
            expected_drops = expected["dropped_count"]
            actual_drops = len(package.dropped_segments)

            passed = expected_drops == actual_drops
            message = (
                "符合预期"
                if passed
                else f"期望丢弃 {expected_drops} 个 Segment,实际丢弃 {actual_drops} 个"
            )

            assertions.append(
                AssertionResult(
                    assertion_name="dropped_count",
                    passed=passed,
                    expected=expected_drops,
                    actual=actual_drops,
                    message=message,
                )
            )

        # 断言: segment_types (按类型统计)
        if "segment_types" in expected:
            expected_types = expected["segment_types"]
            actual_types = {}
            for seg in package.segments:
                seg_type = seg.type.value
                actual_types[seg_type] = actual_types.get(seg_type, 0) + 1

            passed = expected_types == actual_types
            message = (
                "符合预期"
                if passed
                else f"期望 {expected_types},实际 {actual_types}"
            )

            assertions.append(
                AssertionResult(
                    assertion_name="segment_types",
                    passed=passed,
                    expected=expected_types,
                    actual=actual_types,
                    message=message,
                )
            )

        # 执行自定义断言
        for i, custom_fn in enumerate(tolerance.custom_assertions):
            try:
                passed = custom_fn(package)
                message = "自定义断言通过" if passed else "自定义断言失败"
            except Exception as e:
                passed = False
                message = f"自定义断言执行失败: {e}"

            assertions.append(
                AssertionResult(
                    assertion_name=f"custom_assertion_{i}",
                    passed=passed,
                    expected="True",
                    actual=str(passed),
                    message=message,
                )
            )

        return assertions

    def _compare_with_tolerance(
        self,
        expected: int | float,
        actual: int | float,
        tolerance: float,
    ) -> tuple[bool, str]:
        """
        带容差比较两个数值。

        参数:
            expected: 期望值
            actual: 实际值
            tolerance: 容差（百分比,0.05 = 5%）

        返回:
            (是否通过, 详细信息)
        """
        delta = abs(expected - actual)
        delta_percent = delta / expected if expected != 0 else 0

        if delta_percent <= tolerance:
            return True, f"符合预期（差异 {delta_percent * 100:.1f}%）"
        else:
            return False, f"超出容差（期望 {expected},实际 {actual},差异 {delta_percent * 100:.1f}%）"

    def _match_tags(self, case_tags: dict[str, str], filter_tags: dict[str, str]) -> bool:
        """检查用例的标签是否匹配过滤条件。"""
        return all(case_tags.get(k) == v for k, v in filter_tags.items())
