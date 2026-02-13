"""
基于规则的路由器——零 LLM 依赖的默认路由实现。

→ 6.6.1 意图驱动路由

规则引擎通过可配置的规则列表进行路由决策。每条规则定义了条件和目标模型，
按优先级顺序匹配。这是 Context Forge 的默认路由器，无需任何外部 API 调用。

# [Design Decision] 规则引擎是默认路由器而非 LLM 路由器，因为：
# 1. 零延迟 — 纯计算，不需要等待 API 响应
# 2. 零成本 — 不产生 LLM API 费用
# 3. 可解释性强 — 每条规则都是显式的，便于调试和审计
# 4. 准确率足够 — 在明确定义的路由策略下，规则引擎的准确率接近 100%
#
# 适用场景：80% 的生产路由需求可以用规则描述（复杂度分流、关键词路由、领域路由）。
# 不适用场景：需要理解用户意图的语义路由（如情感分析、细粒度分类）。
"""

from __future__ import annotations

import re
from typing import Any

from context_forge.config.defaults import MODEL_REGISTRY, resolve_model
from context_forge.errors.exceptions import ModelNotFoundError, RoutingError
from context_forge.models.routing import (
    ComplexityLevel,
    ModelConfig,
    RoutingDecision,
    RoutingRule,
)
from context_forge.routing.base import RoutingContext
from context_forge.routing.complexity import ComplexityEstimator


class RuleBasedRouter:
    """
    基于规则的路由器。

    → 6.6.1 意图驱动路由

    规则按优先级顺序匹配（数值越大越优先）。第一条匹配的规则决定路由目标。
    如果没有规则匹配，使用默认模型。

    基本用法::

        from context_forge.models.routing import RoutingRule

        rules = [
            RoutingRule(
                name="simple_to_mini",
                condition_type="complexity",
                condition_value="simple",
                target_model="gpt-4o-mini",
                priority=10,
            ),
            RoutingRule(
                name="code_to_sonnet",
                condition_type="keyword",
                condition_value="代码|code|编程",
                target_model="claude-sonnet-4-5-20250514",
                priority=20,
            ),
        ]

        router = RuleBasedRouter(rules=rules, default_model="gpt-4o")
        decision = router.route(context)

    支持的条件类型:
        - complexity: 复杂度等级（simple / moderate / complex / expert）
        - keyword: 关键词正则匹配（支持中英文、管道符分隔）
        - token_count: Token 数量范围（如 ">1000" 或 "500-2000"）
        - segment_type_present: 是否包含指定类型的 Segment（如 "rag" 或 "tool_call"）
    """

    def __init__(
        self,
        rules: list[RoutingRule] | None = None,
        default_model: str = "gpt-4o",
        enable_fallback: bool = True,
        complexity_estimator: ComplexityEstimator | None = None,
    ) -> None:
        """
        初始化规则路由器。

        参数:
            rules: 路由规则列表（按 priority 降序排序）
            default_model: 默认模型 ID（无规则匹配时使用）
            enable_fallback: 是否启用降级路径（目标模型不可用时使用 fallback_model）
            complexity_estimator: 复杂度估计器（None 时使用默认配置）
        """
        self.rules = sorted(rules or [], key=lambda r: r.priority, reverse=True)
        self.default_model_id = default_model
        self.enable_fallback = enable_fallback
        self.complexity_estimator = complexity_estimator or ComplexityEstimator()

        # 解析默认模型
        self.default_model = self._resolve_model(default_model)

    def route(self, context: RoutingContext) -> RoutingDecision:
        """
        执行路由决策。

        参数:
            context: 路由上下文

        返回:
            路由决策结果

        异常:
            RoutingError: 路由失败时抛出
        """
        # 1. 估算复杂度
        complexity_signals = self.complexity_estimator.estimate_with_signals(context.query)
        complexity = complexity_signals.estimated_level

        # 2. 按优先级顺序匹配规则
        for rule in self.rules:
            if self._match_rule(rule, context, complexity):
                # 规则匹配，尝试使用目标模型
                selected_model = self._resolve_model_with_fallback(rule)
                return RoutingDecision(
                    selected_model=selected_model,
                    complexity=complexity,
                    matched_rule=rule.name,
                    is_fallback=(
                        selected_model.model_id != rule.target_model
                        if self.enable_fallback
                        else False
                    ),
                    confidence=complexity_signals.confidence,
                    reasoning=self._build_reasoning(rule, complexity_signals),
                    estimated_cost=0.0,  # 成本在实际使用时计算
                )

        # 3. 无规则匹配，使用默认模型
        return RoutingDecision(
            selected_model=self.default_model,
            complexity=complexity,
            matched_rule="default",
            is_fallback=False,
            confidence=complexity_signals.confidence,
            reasoning=f"无匹配规则，使用默认模型 {self.default_model.model_id}",
            estimated_cost=0.0,
        )

    def _match_rule(
        self,
        rule: RoutingRule,
        context: RoutingContext,
        complexity: ComplexityLevel,
    ) -> bool:
        """
        判断规则是否匹配。

        参数:
            rule: 路由规则
            context: 路由上下文
            complexity: 判定的复杂度等级

        返回:
            True 表示匹配
        """
        condition_type = rule.condition_type.lower()
        condition_value = rule.condition_value

        # → 6.6.1.1 复杂度分流
        if condition_type == "complexity":
            return complexity.value == condition_value.lower()

        # → 6.6.1.2 关键词路由
        if condition_type == "keyword":
            # 支持管道符分隔的多关键词（正则匹配）
            pattern = re.compile(condition_value, re.IGNORECASE)
            return pattern.search(context.query) is not None

        # → 6.6.1.3 Token 数量路由
        if condition_type == "token_count":
            total_tokens = context.total_tokens
            return self._match_token_range(total_tokens, condition_value)

        # → 6.6.1.4 Segment 类型路由
        if condition_type == "segment_type_present":
            return context.has_segment_type(condition_value)

        # 未知条件类型，不匹配
        return False

    def _match_token_range(self, count: int, range_expr: str) -> bool:
        """
        匹配 Token 数量范围表达式。

        支持格式:
            - ">1000" — 大于 1000
            - "<500" — 小于 500
            - "500-2000" — 500 到 2000 之间
            - "1000" — 精确等于 1000

        参数:
            count: 实际 Token 数量
            range_expr: 范围表达式

        返回:
            True 表示匹配
        """
        range_expr = range_expr.strip()

        # 大于
        if range_expr.startswith(">"):
            threshold = int(range_expr[1:].strip())
            return count > threshold

        # 小于
        if range_expr.startswith("<"):
            threshold = int(range_expr[1:].strip())
            return count < threshold

        # 区间
        if "-" in range_expr:
            parts = range_expr.split("-")
            if len(parts) == 2:
                min_val = int(parts[0].strip())
                max_val = int(parts[1].strip())
                return min_val <= count <= max_val

        # 精确匹配
        try:
            target = int(range_expr)
            return count == target
        except ValueError:
            return False

    def _resolve_model(self, model_id: str) -> ModelConfig:
        """
        解析模型 ID 到 ModelConfig。

        参数:
            model_id: 模型标识符

        返回:
            模型配置对象

        异常:
            RoutingError: 模型未找到时抛出
        """
        try:
            model = resolve_model(model_id)
        except ModelNotFoundError as e:
            raise RoutingError(
                what=f"路由失败：模型 '{model_id}' 未找到",
                why="指定的模型不在注册表中，可能是拼写错误或模型未注册",
                how=(
                    f"请检查模型 ID 是否正确，或使用以下已注册的模型之一：\n"
                    f"{', '.join(list(MODEL_REGISTRY.keys())[:10])}..."
                ),
            ) from e
        return model

    def _resolve_model_with_fallback(self, rule: RoutingRule) -> ModelConfig:
        """
        解析模型 ID，支持降级路径。

        参数:
            rule: 路由规则

        返回:
            模型配置对象
        """
        try:
            return self._resolve_model(rule.target_model)
        except RoutingError:
            if self.enable_fallback and rule.fallback_model:
                # 尝试使用降级模型
                try:
                    return self._resolve_model(rule.fallback_model)
                except RoutingError:
                    pass
            # 降级失败或未启用降级，使用默认模型
            return self.default_model

    def _build_reasoning(
        self,
        rule: RoutingRule,
        signals: Any,
    ) -> str:
        """
        构建决策推理说明。

        参数:
            rule: 匹配的规则
            signals: 复杂度信号

        返回:
            决策推理文本
        """
        parts = [
            f"匹配规则: {rule.name}",
            f"条件类型: {rule.condition_type}",
            f"条件值: {rule.condition_value}",
            f"目标模型: {rule.target_model}",
        ]

        # 添加复杂度信号
        parts.append(f"复杂度: {signals.estimated_level.value}")
        parts.append(f"置信度: {signals.confidence:.2f}")

        # 添加关键特征
        features = []
        if signals.has_complex_task_words:
            features.append("包含复杂任务词")
        if signals.has_comparison_words:
            features.append("包含对比词")
        if signals.code_block_count > 0:
            features.append(f"{signals.code_block_count} 个代码块")
        if signals.math_symbol_count > 0:
            features.append(f"{signals.math_symbol_count} 个数学符号")

        if features:
            parts.append(f"关键特征: {', '.join(features)}")

        return " | ".join(parts)


def create_default_complexity_rules(
    simple_model: str = "gpt-4o-mini",
    moderate_model: str = "gpt-4o",
    complex_model: str = "claude-sonnet-4-5-20250514",
    expert_model: str = "claude-opus-4-20250115",
) -> list[RoutingRule]:
    """
    创建默认的复杂度路由规则。

    → 6.6.1.1 复杂度分流

    这是最常用的路由策略：简单查询用便宜的小模型，复杂查询用旗舰模型。

    参数:
        simple_model: SIMPLE 级别使用的模型
        moderate_model: MODERATE 级别使用的模型
        complex_model: COMPLEX 级别使用的模型
        expert_model: EXPERT 级别使用的模型

    返回:
        路由规则列表
    """
    return [
        RoutingRule(
            name="expert_to_opus",
            condition_type="complexity",
            condition_value="expert",
            target_model=expert_model,
            priority=40,
            fallback_model=complex_model,
        ),
        RoutingRule(
            name="complex_to_sonnet",
            condition_type="complexity",
            condition_value="complex",
            target_model=complex_model,
            priority=30,
            fallback_model=moderate_model,
        ),
        RoutingRule(
            name="moderate_to_gpt4o",
            condition_type="complexity",
            condition_value="moderate",
            target_model=moderate_model,
            priority=20,
            fallback_model=simple_model,
        ),
        RoutingRule(
            name="simple_to_mini",
            condition_type="complexity",
            condition_value="simple",
            target_model=simple_model,
            priority=10,
            fallback_model=moderate_model,
        ),
    ]
