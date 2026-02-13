"""
路由与调度模块。

→ 6.6 上下文路由与动态调度

路由模块解决"不同查询应该用不同模型"的核心问题。它包含：

1. **基础协议** — Router, RoutingContext, AgentContext
2. **复杂度估计** — ComplexityEstimator（启发式规则）
3. **规则路由器** — RuleBasedRouter（零 LLM 依赖，默认实现）
4. **LLM 路由器** — LLMRouter（可选的高级路由）
5. **上下文总线** — ContextBus（多 Agent 协调）

# [DX Decision] 提供工厂函数 create_default_router()，
# 让用户无需了解内部实现就能获得开箱即用的路由器。
"""

from __future__ import annotations

from context_forge.routing.base import AgentContext, Router, RoutingContext
from context_forge.routing.complexity import ComplexityEstimator, ComplexitySignals
from context_forge.routing.context_bus import (
    ContextBus,
    ContextEvent,
    HandoffRequest,
)
from context_forge.routing.llm_router import LLMRouter, create_mock_llm_call_fn
from context_forge.routing.rule_based import (
    RuleBasedRouter,
    create_default_complexity_rules,
)

__all__ = [
    # 基础协议
    "Router",
    "RoutingContext",
    "AgentContext",
    # 复杂度估计
    "ComplexityEstimator",
    "ComplexitySignals",
    # 规则路由器
    "RuleBasedRouter",
    "create_default_complexity_rules",
    # LLM 路由器
    "LLMRouter",
    "create_mock_llm_call_fn",
    # 上下文总线
    "ContextBus",
    "ContextEvent",
    "HandoffRequest",
    # 工厂函数
    "create_default_router",
]


def create_default_router(
    router_type: str = "rule",
    simple_model: str = "gpt-4o-mini",
    moderate_model: str = "gpt-4o",
    complex_model: str = "claude-sonnet-4-5-20250514",
    expert_model: str = "claude-opus-4-20250115",
) -> Router:
    """
    创建默认路由器。

    → 6.6.1 意图驱动路由

    这是最便捷的路由器创建方式，封装了默认配置。

    基本用法::

        # 使用规则路由器（默认）
        router = create_default_router()

        # 使用 LLM 路由器（需要提供 LLM 调用函数）
        router = create_default_router(router_type="llm")

        # 自定义模型映射
        router = create_default_router(
            simple_model="gpt-4o-mini",
            complex_model="claude-sonnet-4-5-20250514",
        )

    参数:
        router_type: 路由器类型（"rule" 或 "llm"）
        simple_model: SIMPLE 级别使用的模型 ID
        moderate_model: MODERATE 级别使用的模型 ID
        complex_model: COMPLEX 级别使用的模型 ID
        expert_model: EXPERT 级别使用的模型 ID

    返回:
        路由器实例

    异常:
        ValueError: router_type 无效时抛出
    """
    # 创建默认的复杂度路由规则
    rules = create_default_complexity_rules(
        simple_model=simple_model,
        moderate_model=moderate_model,
        complex_model=complex_model,
        expert_model=expert_model,
    )

    if router_type == "rule":
        # 规则路由器（默认）
        return RuleBasedRouter(
            rules=rules,
            default_model=moderate_model,
            enable_fallback=True,
        )
    elif router_type == "llm":
        # LLM 路由器（使用 Mock 调用函数）
        # 🏭 生产提示：用户应该替换为真实的 LLM Provider
        return LLMRouter(
            llm_call_fn=create_mock_llm_call_fn(),
            fallback_router=RuleBasedRouter(
                rules=rules,
                default_model=moderate_model,
            ),
        )
    else:
        raise ValueError(
            f"无效的 router_type: {router_type}。"
            f"支持的类型: 'rule', 'llm'"
        )
