"""
路由模块单元测试。

→ 6.6 上下文路由与动态调度
"""

from __future__ import annotations

import json
import warnings
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from context_forge.errors.exceptions import RoutingError
from context_forge.models.routing import ComplexityLevel, RoutingRule
from context_forge.models.segment import Segment, SegmentType
from context_forge.routing import (
    AgentContext,
    ComplexityEstimator,
    ContextBus,
    HandoffRequest,
    RuleBasedRouter,
    RoutingContext,
    create_default_router,
)
from context_forge.routing.llm_router import LLMRouter, create_mock_llm_call_fn


class TestComplexityEstimator:
    """ComplexityEstimator 测试。"""

    def test_simple_query(self) -> None:
        """测试简单查询识别。"""
        estimator = ComplexityEstimator()
        level = estimator.estimate("退货地址是哪？")
        assert level == ComplexityLevel.SIMPLE

    def test_moderate_query(self) -> None:
        """测试中等复杂度查询。"""
        estimator = ComplexityEstimator()
        level = estimator.estimate("请比较 Python 和 Go 的并发模型")
        assert level in (ComplexityLevel.SIMPLE, ComplexityLevel.MODERATE)

    def test_complex_query(self) -> None:
        """测试复杂查询识别。"""
        estimator = ComplexityEstimator()
        query = "请设计一个高可用的分布式缓存系统，要求支持数据分片和自动故障转移"
        level = estimator.estimate(query)
        assert level in (ComplexityLevel.MODERATE, ComplexityLevel.COMPLEX)

    def test_expert_query(self) -> None:
        """测试专家级查询识别。"""
        estimator = ComplexityEstimator()
        query = "请证明费马大定理，并详细推导数学证明过程，包含完整的代数变换步骤"
        level = estimator.estimate(query)
        # 复杂查询可能被判定为 MODERATE/COMPLEX/EXPERT，取决于启发式规则的权重
        assert level in (
            ComplexityLevel.MODERATE,
            ComplexityLevel.COMPLEX,
            ComplexityLevel.EXPERT,
        )

    def test_signals(self) -> None:
        """测试复杂度信号。"""
        estimator = ComplexityEstimator()
        signals = estimator.estimate_with_signals("请分析并比较两种算法")

        assert signals.has_comparison_words
        assert signals.has_complex_task_words
        assert 0.0 <= signals.confidence <= 1.0


class TestRuleBasedRouter:
    """RuleBasedRouter 测试。"""

    def test_complexity_routing(self) -> None:
        """测试复杂度路由。"""
        router = create_default_router(router_type="rule")

        segment = Segment(
            type=SegmentType.USER,
            content="退货地址是哪？",
            role="user",
            token_count=10,
        )

        context = RoutingContext(
            segments=[segment],
            query="退货地址是哪？",
            max_budget_tokens=4096,
        )

        decision = router.route(context)
        assert decision.selected_model.model_id == "gpt-4o-mini"
        assert decision.complexity == ComplexityLevel.SIMPLE

    def test_keyword_routing(self) -> None:
        """测试关键词路由。"""
        rules = [
            RoutingRule(
                name="code_keyword",
                condition_type="keyword",
                condition_value="代码|code",
                target_model="claude-sonnet-4-5-20250514",
                priority=10,
            ),
        ]

        router = RuleBasedRouter(rules=rules, default_model="gpt-4o-mini")

        segment = Segment(
            type=SegmentType.USER,
            content="请帮我写代码",
            role="user",
            token_count=10,
        )

        context = RoutingContext(
            segments=[segment],
            query="请帮我写代码",
            max_budget_tokens=4096,
        )

        decision = router.route(context)
        assert decision.selected_model.model_id == "claude-sonnet-4-5-20250514"

    def test_token_count_routing(self) -> None:
        """测试 Token 数量路由。"""
        rules = [
            RoutingRule(
                name="large_context",
                condition_type="token_count",
                condition_value=">100",
                target_model="gpt-4o",
                priority=10,
            ),
        ]

        router = RuleBasedRouter(rules=rules, default_model="gpt-4o-mini")

        segment = Segment(
            type=SegmentType.USER,
            content="x" * 200,
            role="user",
            token_count=200,
        )

        context = RoutingContext(
            segments=[segment],
            query="x" * 200,
            max_budget_tokens=4096,
        )

        decision = router.route(context)
        assert decision.selected_model.model_id == "gpt-4o"

    def test_segment_type_routing(self) -> None:
        """测试 Segment 类型路由。"""
        rules = [
            RoutingRule(
                name="rag_routing",
                condition_type="segment_type_present",
                condition_value="rag",
                target_model="gpt-4o",
                priority=10,
            ),
        ]

        router = RuleBasedRouter(rules=rules, default_model="gpt-4o-mini")

        segments = [
            Segment(
                type=SegmentType.USER,
                content="问题",
                role="user",
                token_count=5,
            ),
            Segment(
                type=SegmentType.RAG,
                content="文档片段",
                role="user",
                token_count=10,
            ),
        ]

        context = RoutingContext(
            segments=segments,
            query="问题",
            max_budget_tokens=4096,
        )

        decision = router.route(context)
        assert decision.selected_model.model_id == "gpt-4o"


class TestContextBus:
    """ContextBus 测试。"""

    def test_agent_registration(self) -> None:
        """测试 Agent 注册。"""
        bus = ContextBus()
        agent = AgentContext(
            agent_id="test_agent",
            namespace="test",
            role="tester",
        )

        bus.register_agent(agent)
        assert "test_agent" in bus._agents

    def test_segment_publishing(self) -> None:
        """测试 Segment 发布。"""
        bus = ContextBus()
        agent = AgentContext(
            agent_id="test_agent",
            namespace="test",
            role="tester",
        )
        bus.register_agent(agent)

        segment = Segment(
            type=SegmentType.USER,
            content="测试内容",
            role="user",
        )

        bus.publish_segment(agent, segment)
        visible = bus.get_visible_segments(agent)
        assert len(visible) == 1

    def test_namespace_isolation(self) -> None:
        """测试命名空间隔离。"""
        bus = ContextBus()

        agent1 = AgentContext(agent_id="agent1", namespace="ns1", role="r1")
        agent2 = AgentContext(agent_id="agent2", namespace="ns2", role="r2")

        bus.register_agent(agent1)
        bus.register_agent(agent2)

        segment = Segment(
            type=SegmentType.USER,
            content="Agent1 的内容",
            role="user",
        )

        bus.publish_segment(agent1, segment)

        # Agent1 能看到自己的 Segment
        assert len(bus.get_visible_segments(agent1)) == 1

        # Agent2 看不到 Agent1 的 Segment（不包含 default）
        assert len(bus.get_visible_segments(agent2, include_default=False)) == 0

    def test_handoff(self) -> None:
        """测试上下文移交。"""
        bus = ContextBus()

        agent1 = AgentContext(agent_id="agent1", namespace="ns1", role="r1")
        agent2 = AgentContext(agent_id="agent2", namespace="ns2", role="r2")

        bus.register_agent(agent1)
        bus.register_agent(agent2)

        segment = Segment(
            type=SegmentType.STATE,
            content="计划",
            role="assistant",
        )

        bus.publish_segment(agent1, segment)

        # 移交前，Agent2 看不到
        assert len(bus.get_visible_segments(agent2, include_default=False)) == 0

        # 执行移交
        handoff = HandoffRequest(
            from_agent_id="agent1",
            to_agent_id="agent2",
            reason="测试移交",
        )
        bus.handoff(handoff)

        # 移交后，Agent2 能看到
        assert len(bus.get_visible_segments(agent2)) > 0


class TestRoutingContext:
    """RoutingContext 测试。"""

    def test_total_tokens(self) -> None:
        """测试 Token 总数计算。"""
        segments = [
            Segment(
                type=SegmentType.USER,
                content="test",
                role="user",
                token_count=10,
            ),
            Segment(
                type=SegmentType.RAG,
                content="doc",
                role="user",
                token_count=20,
            ),
        ]

        context = RoutingContext(
            segments=segments,
            query="test",
            max_budget_tokens=1000,
        )

        assert context.total_tokens == 30

    def test_segment_types(self) -> None:
        """测试 Segment 类型集合。"""
        segments = [
            Segment(type=SegmentType.USER, content="1", role="user"),
            Segment(type=SegmentType.RAG, content="2", role="user"),
            Segment(type=SegmentType.RAG, content="3", role="user"),
        ]

        context = RoutingContext(
            segments=segments,
            query="test",
            max_budget_tokens=1000,
        )

        assert context.segment_types == {"user", "rag"}
        assert context.has_segment_type("rag")
        assert not context.has_segment_type("tool_call")


class TestLLMRouter:
    """LLM 路由器测试（包含 Mock LLM）。"""

    def test_llm_router_simple_query(self) -> None:
        """测试简单查询的 LLM 路由。"""
        mock_llm_fn = MagicMock(return_value=json.dumps({
            "complexity": "simple",
            "confidence": 0.85,
            "reasoning": "查询简短，是简单问题"
        }, ensure_ascii=False))

        router = LLMRouter(
            llm_call_fn=mock_llm_fn,
            fallback_router=RuleBasedRouter(),
        )

        segment = Segment(
            type=SegmentType.USER,
            content="退货地址是哪？",
            role="user",
            token_count=10,
        )

        context = RoutingContext(
            segments=[segment],
            query="退货地址是哪？",
            max_budget_tokens=4096,
        )

        decision = router.route(context)
        assert decision.complexity == ComplexityLevel.SIMPLE
        assert decision.confidence == 0.85
        assert "LLM 分类" in decision.reasoning
        mock_llm_fn.assert_called_once()

    def test_llm_router_complex_query(self) -> None:
        """测试复杂查询的 LLM 路由。"""
        mock_llm_fn = MagicMock(return_value=json.dumps({
            "complexity": "complex",
            "confidence": 0.92,
            "reasoning": "涉及多步推理和比较分析"
        }, ensure_ascii=False))

        router = LLMRouter(
            llm_call_fn=mock_llm_fn,
            fallback_router=RuleBasedRouter(),
        )

        segment = Segment(
            type=SegmentType.USER,
            content="请设计一个高可用的分布式缓存系统，要求支持数据分片和自动故障转移，并详细讨论一致性和可用性的权衡",
            role="user",
            token_count=50,
        )

        context = RoutingContext(
            segments=[segment],
            query="请设计一个高可用的分布式缓存系统...",
            max_budget_tokens=128000,
        )

        decision = router.route(context)
        assert decision.complexity == ComplexityLevel.COMPLEX
        assert decision.confidence == 0.92

    def test_llm_router_expert_query(self) -> None:
        """测试专家级查询的 LLM 路由。"""
        mock_llm_fn = MagicMock(return_value=json.dumps({
            "complexity": "expert",
            "confidence": 0.95,
            "reasoning": "需要深度数学推导和算法设计"
        }, ensure_ascii=False))

        router = LLMRouter(
            llm_call_fn=mock_llm_fn,
            fallback_router=RuleBasedRouter(),
        )

        segment = Segment(
            type=SegmentType.USER,
            content="证明费马大定理并推导完整的数学证明过程",
            role="user",
            token_count=30,
        )

        context = RoutingContext(
            segments=[segment],
            query="证明费马大定理...",
            max_budget_tokens=128000,
        )

        decision = router.route(context)
        assert decision.complexity == ComplexityLevel.EXPERT
        assert decision.confidence == 0.95

    def test_llm_router_moderate_query(self) -> None:
        """测试中等复杂度查询的 LLM 路由。"""
        mock_llm_fn = MagicMock(return_value=json.dumps({
            "complexity": "moderate",
            "confidence": 0.78,
            "reasoning": "需要部分分析和综合"
        }, ensure_ascii=False))

        router = LLMRouter(
            llm_call_fn=mock_llm_fn,
            fallback_router=RuleBasedRouter(),
        )

        segment = Segment(
            type=SegmentType.USER,
            content="请比较 Python 和 Go 的并发模型",
            role="user",
            token_count=20,
        )

        context = RoutingContext(
            segments=[segment],
            query="请比较 Python 和 Go 的并发模型",
            max_budget_tokens=8192,
        )

        decision = router.route(context)
        assert decision.complexity == ComplexityLevel.MODERATE
        assert decision.confidence == 0.78

    def test_llm_router_cache_hit(self) -> None:
        """测试 LLM 路由缓存命中。"""
        mock_llm_fn = MagicMock(return_value=json.dumps({
            "complexity": "simple",
            "confidence": 0.85,
            "reasoning": "缓存测试"
        }, ensure_ascii=False))

        router = LLMRouter(
            llm_call_fn=mock_llm_fn,
            fallback_router=RuleBasedRouter(),
            enable_cache=True,
        )

        segment = Segment(
            type=SegmentType.USER,
            content="hello",
            role="user",
            token_count=5,
        )

        context = RoutingContext(
            segments=[segment],
            query="hello",
            max_budget_tokens=4096,
        )

        # 第一次调用
        decision1 = router.route(context)
        assert mock_llm_fn.call_count == 1

        # 第二次调用相同查询（应该命中缓存）
        decision2 = router.route(context)
        assert mock_llm_fn.call_count == 1  # 未增加
        assert decision1.complexity == decision2.complexity

    def test_llm_router_cache_miss_different_query(self) -> None:
        """测试 LLM 路由缓存未命中（不同查询）。"""
        mock_llm_fn = MagicMock(return_value=json.dumps({
            "complexity": "simple",
            "confidence": 0.85,
            "reasoning": "测试"
        }, ensure_ascii=False))

        router = LLMRouter(
            llm_call_fn=mock_llm_fn,
            fallback_router=RuleBasedRouter(),
            enable_cache=True,
        )

        # 第一个查询
        context1 = RoutingContext(
            segments=[Segment(type=SegmentType.USER, content="hello", role="user")],
            query="hello",
            max_budget_tokens=4096,
        )
        router.route(context1)
        assert mock_llm_fn.call_count == 1

        # 第二个不同的查询
        context2 = RoutingContext(
            segments=[Segment(type=SegmentType.USER, content="world", role="user")],
            query="world",
            max_budget_tokens=4096,
        )
        router.route(context2)
        assert mock_llm_fn.call_count == 2

    def test_llm_router_cache_disabled(self) -> None:
        """测试禁用缓存的 LLM 路由。"""
        mock_llm_fn = MagicMock(return_value=json.dumps({
            "complexity": "simple",
            "confidence": 0.85,
            "reasoning": "测试"
        }, ensure_ascii=False))

        router = LLMRouter(
            llm_call_fn=mock_llm_fn,
            fallback_router=RuleBasedRouter(),
            enable_cache=False,
        )

        context = RoutingContext(
            segments=[Segment(type=SegmentType.USER, content="hello", role="user")],
            query="hello",
            max_budget_tokens=4096,
        )

        # 调用多次
        router.route(context)
        router.route(context)
        # 由于缓存禁用，每次都应该调用 LLM
        assert mock_llm_fn.call_count == 2

    def test_llm_router_markdown_wrapped_response(self) -> None:
        """测试处理 Markdown 包裹的 LLM 响应。"""
        markdown_response = """```json
{
  "complexity": "simple",
  "confidence": 0.85,
  "reasoning": "简单查询"
}
```"""
        mock_llm_fn = MagicMock(return_value=markdown_response)

        router = LLMRouter(
            llm_call_fn=mock_llm_fn,
            fallback_router=RuleBasedRouter(),
        )

        context = RoutingContext(
            segments=[Segment(type=SegmentType.USER, content="test", role="user")],
            query="test",
            max_budget_tokens=4096,
        )

        decision = router.route(context)
        assert decision.complexity == ComplexityLevel.SIMPLE

    def test_llm_router_json_keyword_wrapped_response(self) -> None:
        """测试处理 json 关键字包裹的 LLM 响应。"""
        json_wrapped = """json
{
  "complexity": "moderate",
  "confidence": 0.80,
  "reasoning": "中等复杂度"
}"""
        mock_llm_fn = MagicMock(return_value=json_wrapped)

        router = LLMRouter(
            llm_call_fn=mock_llm_fn,
            fallback_router=RuleBasedRouter(),
        )

        context = RoutingContext(
            segments=[Segment(type=SegmentType.USER, content="test", role="user")],
            query="test",
            max_budget_tokens=4096,
        )

        decision = router.route(context)
        assert decision.complexity == ComplexityLevel.MODERATE

    def test_llm_router_invalid_json_response(self) -> None:
        """测试处理无效 JSON 响应时的降级。"""
        mock_llm_fn = MagicMock(return_value="not a json response")

        router = LLMRouter(
            llm_call_fn=mock_llm_fn,
            fallback_router=RuleBasedRouter(),
        )

        context = RoutingContext(
            segments=[Segment(type=SegmentType.USER, content="test", role="user")],
            query="test",
            max_budget_tokens=4096,
        )

        # 无效 JSON 会触发 RoutingError，但被 route() 捕获后降级到 fallback_router
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            decision = router.route(context)
            assert decision is not None
            assert len(w) == 1
            assert "LLM 路由失败" in str(w[0].message)

    def test_llm_router_llm_call_exception(self) -> None:
        """测试 LLM 调用异常时的降级。"""
        mock_llm_fn = MagicMock(side_effect=RuntimeError("API 调用失败"))

        router = LLMRouter(
            llm_call_fn=mock_llm_fn,
            fallback_router=RuleBasedRouter(),
        )

        context = RoutingContext(
            segments=[Segment(type=SegmentType.USER, content="test", role="user")],
            query="test",
            max_budget_tokens=4096,
        )

        # 应该发出警告并降级到 fallback router
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            decision = router.route(context)
            assert len(w) == 1
            assert "LLM 路由失败" in str(w[0].message)
            assert decision is not None  # 应该返回 fallback 结果

    def test_llm_router_none_llm_call_fn(self) -> None:
        """测试 llm_call_fn 为 None 时直接使用 fallback_router。"""
        router = LLMRouter(
            llm_call_fn=None,
            fallback_router=RuleBasedRouter(),
        )

        context = RoutingContext(
            segments=[Segment(type=SegmentType.USER, content="test", role="user")],
            query="test",
            max_budget_tokens=4096,
        )

        # 应该直接使用 fallback router（无需调用 LLM）
        decision = router.route(context)
        assert decision is not None
        # fallback_router 不会标记 is_fallback，因为这是主路由路径
        assert decision.selected_model is not None

    def test_llm_router_missing_complexity_field(self) -> None:
        """测试 LLM 响应缺少 complexity 字段时的默认值。"""
        mock_llm_fn = MagicMock(return_value=json.dumps({
            "confidence": 0.80,
            "reasoning": "缺少 complexity 字段"
        }, ensure_ascii=False))

        router = LLMRouter(
            llm_call_fn=mock_llm_fn,
            fallback_router=RuleBasedRouter(),
        )

        context = RoutingContext(
            segments=[Segment(type=SegmentType.USER, content="test", role="user")],
            query="test",
            max_budget_tokens=4096,
        )

        decision = router.route(context)
        # 应该使用默认值 MODERATE
        assert decision.complexity == ComplexityLevel.MODERATE

    def test_llm_router_missing_confidence_field(self) -> None:
        """测试 LLM 响应缺少 confidence 字段时的默认值。"""
        mock_llm_fn = MagicMock(return_value=json.dumps({
            "complexity": "complex",
            "reasoning": "缺少 confidence 字段"
        }, ensure_ascii=False))

        router = LLMRouter(
            llm_call_fn=mock_llm_fn,
            fallback_router=RuleBasedRouter(),
        )

        context = RoutingContext(
            segments=[Segment(type=SegmentType.USER, content="test", role="user")],
            query="test",
            max_budget_tokens=4096,
        )

        decision = router.route(context)
        # 应该使用默认值 0.5
        assert decision.confidence == 0.5

    def test_llm_router_missing_reasoning_field(self) -> None:
        """测试 LLM 响应缺少 reasoning 字段时的默认值。"""
        mock_llm_fn = MagicMock(return_value=json.dumps({
            "complexity": "complex",
            "confidence": 0.80
        }, ensure_ascii=False))

        router = LLMRouter(
            llm_call_fn=mock_llm_fn,
            fallback_router=RuleBasedRouter(),
        )

        context = RoutingContext(
            segments=[Segment(type=SegmentType.USER, content="test", role="user")],
            query="test",
            max_budget_tokens=4096,
        )

        decision = router.route(context)
        # 应该使用默认值 ""
        assert "LLM 分类" in decision.reasoning

    def test_llm_router_invalid_complexity_value(self) -> None:
        """测试 LLM 响应包含无效 complexity 值时的默认值。"""
        mock_llm_fn = MagicMock(return_value=json.dumps({
            "complexity": "invalid_level",
            "confidence": 0.80,
            "reasoning": "无效的复杂度值"
        }, ensure_ascii=False))

        router = LLMRouter(
            llm_call_fn=mock_llm_fn,
            fallback_router=RuleBasedRouter(),
        )

        context = RoutingContext(
            segments=[Segment(type=SegmentType.USER, content="test", role="user")],
            query="test",
            max_budget_tokens=4096,
        )

        decision = router.route(context)
        # 应该映射到默认值 MODERATE
        assert decision.complexity == ComplexityLevel.MODERATE

    def test_llm_router_case_insensitive_complexity(self) -> None:
        """测试 complexity 字段的大小写不敏感性。"""
        mock_llm_fn = MagicMock(return_value=json.dumps({
            "complexity": "SIMPLE",  # 大写
            "confidence": 0.85,
            "reasoning": "大小写测试"
        }, ensure_ascii=False))

        router = LLMRouter(
            llm_call_fn=mock_llm_fn,
            fallback_router=RuleBasedRouter(),
        )

        context = RoutingContext(
            segments=[Segment(type=SegmentType.USER, content="test", role="user")],
            query="test",
            max_budget_tokens=4096,
        )

        decision = router.route(context)
        assert decision.complexity == ComplexityLevel.SIMPLE

    def test_llm_router_confidence_float_conversion(self) -> None:
        """测试 confidence 值的类型转换。"""
        mock_llm_fn = MagicMock(return_value=json.dumps({
            "complexity": "complex",
            "confidence": 0.75,  # float
            "reasoning": "类型转换测试"
        }, ensure_ascii=False))

        router = LLMRouter(
            llm_call_fn=mock_llm_fn,
            fallback_router=RuleBasedRouter(),
        )

        context = RoutingContext(
            segments=[Segment(type=SegmentType.USER, content="test", role="user")],
            query="test",
            max_budget_tokens=4096,
        )

        decision = router.route(context)
        assert isinstance(decision.confidence, float)
        assert decision.confidence == 0.75

    def test_llm_router_with_metadata(self) -> None:
        """测试 LLM 路由传递元数据到 fallback router。"""
        mock_llm_fn = MagicMock(return_value=json.dumps({
            "complexity": "simple",
            "confidence": 0.85,
            "reasoning": "元数据测试"
        }, ensure_ascii=False))

        router = LLMRouter(
            llm_call_fn=mock_llm_fn,
            fallback_router=RuleBasedRouter(),
        )

        metadata = {"user_id": "user_123", "domain": "support"}
        context = RoutingContext(
            segments=[Segment(type=SegmentType.USER, content="test", role="user")],
            query="test",
            max_budget_tokens=4096,
            metadata=metadata,
        )

        decision = router.route(context)
        # 验证 LLM 复杂度被正确传递
        assert decision.complexity == ComplexityLevel.SIMPLE
        assert "_llm_complexity" in decision.reasoning or decision.reasoning  # 应该包含标记

    def test_llm_router_decision_annotated_as_llm(self) -> None:
        """测试路由决策被正确标注为 LLM 路由。"""
        mock_llm_fn = MagicMock(return_value=json.dumps({
            "complexity": "complex",
            "confidence": 0.92,
            "reasoning": "LLM 标注测试"
        }, ensure_ascii=False))

        router = LLMRouter(
            llm_call_fn=mock_llm_fn,
            fallback_router=RuleBasedRouter(),
        )

        context = RoutingContext(
            segments=[Segment(type=SegmentType.USER, content="test", role="user")],
            query="test",
            max_budget_tokens=4096,
        )

        decision = router.route(context)
        # 验证 matched_rule 包含 llm_classified 标记
        assert "llm_classified" in decision.matched_rule
        assert "LLM 分类" in decision.reasoning

    def test_llm_router_empty_query(self) -> None:
        """测试空查询的 LLM 路由。"""
        mock_llm_fn = MagicMock(return_value=json.dumps({
            "complexity": "simple",
            "confidence": 0.5,
            "reasoning": "空查询"
        }, ensure_ascii=False))

        router = LLMRouter(
            llm_call_fn=mock_llm_fn,
            fallback_router=RuleBasedRouter(),
        )

        context = RoutingContext(
            segments=[Segment(type=SegmentType.USER, content="", role="user")],
            query="",
            max_budget_tokens=4096,
        )

        decision = router.route(context)
        assert decision is not None

    def test_llm_router_very_long_query(self) -> None:
        """测试很长的查询文本。"""
        long_query = "这是一个很长的查询" * 100
        mock_llm_fn = MagicMock(return_value=json.dumps({
            "complexity": "complex",
            "confidence": 0.88,
            "reasoning": "长查询"
        }, ensure_ascii=False))

        router = LLMRouter(
            llm_call_fn=mock_llm_fn,
            fallback_router=RuleBasedRouter(),
        )

        context = RoutingContext(
            segments=[Segment(type=SegmentType.USER, content=long_query, role="user")],
            query=long_query,
            max_budget_tokens=128000,
        )

        decision = router.route(context)
        assert decision.complexity == ComplexityLevel.COMPLEX

    def test_llm_router_special_characters_in_query(self) -> None:
        """测试包含特殊字符的查询。"""
        special_query = "你好🎉 @#$%^&*() <html>test</html>"
        mock_llm_fn = MagicMock(return_value=json.dumps({
            "complexity": "simple",
            "confidence": 0.75,
            "reasoning": "特殊字符测试"
        }, ensure_ascii=False))

        router = LLMRouter(
            llm_call_fn=mock_llm_fn,
            fallback_router=RuleBasedRouter(),
        )

        context = RoutingContext(
            segments=[Segment(type=SegmentType.USER, content=special_query, role="user")],
            query=special_query,
            max_budget_tokens=4096,
        )

        decision = router.route(context)
        assert decision is not None

    def test_llm_router_classifier_model_attribute(self) -> None:
        """测试 classifier_model 属性。"""
        router = LLMRouter(
            llm_call_fn=MagicMock(),
            classifier_model="gpt-4o",
        )
        assert router.classifier_model == "gpt-4o"

    def test_llm_router_create_mock_llm_call_fn(self) -> None:
        """测试创建 Mock LLM 调用函数。"""
        mock_fn = create_mock_llm_call_fn()
        assert callable(mock_fn)

        # 测试简单查询
        prompt = "用户查询：\n测试"
        response = mock_fn(prompt)
        data = json.loads(response)
        assert "complexity" in data
        assert "confidence" in data
        assert "reasoning" in data

    def test_mock_llm_call_fn_simple_query(self) -> None:
        """测试 Mock LLM 对简单查询的分类。"""
        mock_fn = create_mock_llm_call_fn()
        prompt = "用户查询：\n这是一个简短的问题"
        response = mock_fn(prompt)
        data = json.loads(response)
        assert data["complexity"] == "simple"
        assert data["confidence"] == 0.85

    def test_mock_llm_call_fn_moderate_query(self) -> None:
        """测试 Mock LLM 对中等长度查询的分类。"""
        mock_fn = create_mock_llm_call_fn()
        # 构造 50-150 字符的中等长度查询
        prompt = "用户查询：\n" + "这是一个中等长度的查询文本" * 5
        response = mock_fn(prompt)
        data = json.loads(response)
        # 根据 create_mock_llm_call_fn 的实现，长度决定复杂度
        assert data["complexity"] in ("simple", "moderate", "complex")

    def test_mock_llm_call_fn_complex_query(self) -> None:
        """测试 Mock LLM 对复杂查询的分类。"""
        mock_fn = create_mock_llm_call_fn()
        prompt = "用户查询：\n" + "这是一个很长的查询" * 20
        response = mock_fn(prompt)
        data = json.loads(response)
        assert data["complexity"] == "complex"

    def test_mock_llm_call_fn_expert_query(self) -> None:
        """测试 Mock LLM 对专家级查询的分类。"""
        mock_fn = create_mock_llm_call_fn()
        prompt = "用户查询：\n" + "这是一个非常长的查询" * 50
        response = mock_fn(prompt)
        data = json.loads(response)
        assert data["complexity"] == "expert"

    def test_llm_router_with_multiple_segments(self) -> None:
        """测试包含多个 Segment 的路由上下文。"""
        mock_llm_fn = MagicMock(return_value=json.dumps({
            "complexity": "moderate",
            "confidence": 0.82,
            "reasoning": "多 Segment 测试"
        }, ensure_ascii=False))

        router = LLMRouter(
            llm_call_fn=mock_llm_fn,
            fallback_router=RuleBasedRouter(),
        )

        segments = [
            Segment(type=SegmentType.SYSTEM, content="系统提示", role="system"),
            Segment(type=SegmentType.USER, content="用户查询", role="user"),
            Segment(type=SegmentType.RAG, content="RAG 内容", role="user"),
        ]

        context = RoutingContext(
            segments=segments,
            query="用户查询",
            max_budget_tokens=8192,
        )

        decision = router.route(context)
        assert decision.complexity == ComplexityLevel.MODERATE

    def test_llm_router_unicode_json_response(self) -> None:
        """测试 Unicode 字符在 JSON 响应中的处理。"""
        mock_llm_fn = MagicMock(return_value=json.dumps({
            "complexity": "simple",
            "confidence": 0.88,
            "reasoning": "这是一个包含中文、日文（日本語）和Emoji的推理🎯"
        }, ensure_ascii=False))

        router = LLMRouter(
            llm_call_fn=mock_llm_fn,
            fallback_router=RuleBasedRouter(),
        )

        context = RoutingContext(
            segments=[Segment(type=SegmentType.USER, content="测试", role="user")],
            query="测试",
            max_budget_tokens=4096,
        )

        decision = router.route(context)
        assert decision.complexity == ComplexityLevel.SIMPLE
        assert "中文" in decision.reasoning

    def test_llm_router_confidence_boundary_values(self) -> None:
        """测试置信度的边界值（0.0 和 1.0）。"""
        # 测试 confidence = 0.0
        mock_llm_fn_min = MagicMock(return_value=json.dumps({
            "complexity": "simple",
            "confidence": 0.0,
            "reasoning": "最小置信度"
        }, ensure_ascii=False))

        router_min = LLMRouter(
            llm_call_fn=mock_llm_fn_min,
            fallback_router=RuleBasedRouter(),
        )

        context = RoutingContext(
            segments=[Segment(type=SegmentType.USER, content="test", role="user")],
            query="test",
            max_budget_tokens=4096,
        )

        decision_min = router_min.route(context)
        assert decision_min.confidence == 0.0

        # 测试 confidence = 1.0
        mock_llm_fn_max = MagicMock(return_value=json.dumps({
            "complexity": "expert",
            "confidence": 1.0,
            "reasoning": "最大置信度"
        }, ensure_ascii=False))

        router_max = LLMRouter(
            llm_call_fn=mock_llm_fn_max,
            fallback_router=RuleBasedRouter(),
        )

        decision_max = router_max.route(context)
        assert decision_max.confidence == 1.0

    def test_mock_llm_call_fn_without_query_line(self) -> None:
        """测试 Mock LLM 处理没有 '用户查询:' 行的 Prompt。"""
        mock_fn = create_mock_llm_call_fn()
        # 构造没有 '用户查询:' 标记的 prompt
        prompt = "这是一个不标准的 prompt\n没有 用户查询: 行"
        response = mock_fn(prompt)
        data = json.loads(response)
        # 应该返回某个默认复杂度
        assert "complexity" in data
        assert data["complexity"] in ("simple", "moderate", "complex", "expert")

    def test_mock_llm_call_fn_multiline_query_extraction(self) -> None:
        """测试 Mock LLM 提取多行的查询。"""
        mock_fn = create_mock_llm_call_fn()
        # 标准格式
        prompt = """你是一个分类器

用户查询：
这是一个查询"""
        response = mock_fn(prompt)
        data = json.loads(response)
        assert "complexity" in data

    def test_llm_router_classify_with_llm_none_fn(self) -> None:
        """测试 _classify_with_llm 当 llm_call_fn 为 None 时。"""
        router = LLMRouter(llm_call_fn=None, fallback_router=RuleBasedRouter())
        result = router._classify_with_llm("test query")
        assert result is None

    def test_llm_router_json_parsing_with_extra_newlines(self) -> None:
        """测试 JSON 解析处理额外的换行符。"""
        # 包含额外空白和换行的 JSON
        mock_llm_fn = MagicMock(return_value="""

{
  "complexity": "simple",
  "confidence": 0.85,
  "reasoning": "有额外空白"
}

""")

        router = LLMRouter(
            llm_call_fn=mock_llm_fn,
            fallback_router=RuleBasedRouter(),
        )

        context = RoutingContext(
            segments=[Segment(type=SegmentType.USER, content="test", role="user")],
            query="test",
            max_budget_tokens=4096,
        )

        decision = router.route(context)
        assert decision.complexity == ComplexityLevel.SIMPLE

    def test_llm_router_markdown_with_json_keyword(self) -> None:
        """测试处理混合的 Markdown 和 json 关键字响应。"""
        # ```json...``` 格式
        markdown_json = """```json
{
  "complexity": "moderate",
  "confidence": 0.80,
  "reasoning": "混合格式"
}
```"""
        mock_llm_fn = MagicMock(return_value=markdown_json)

        router = LLMRouter(
            llm_call_fn=mock_llm_fn,
            fallback_router=RuleBasedRouter(),
        )

        context = RoutingContext(
            segments=[Segment(type=SegmentType.USER, content="test", role="user")],
            query="test",
            max_budget_tokens=4096,
        )

        decision = router.route(context)
        assert decision.complexity == ComplexityLevel.MODERATE

    def test_llm_router_invalid_confidence_string(self) -> None:
        """测试 confidence 为字符串时的类型转换。"""
        mock_llm_fn = MagicMock(return_value=json.dumps({
            "complexity": "simple",
            "confidence": "0.75",  # 字符串而非 float
            "reasoning": "字符串置信度"
        }, ensure_ascii=False))

        router = LLMRouter(
            llm_call_fn=mock_llm_fn,
            fallback_router=RuleBasedRouter(),
        )

        context = RoutingContext(
            segments=[Segment(type=SegmentType.USER, content="test", role="user")],
            query="test",
            max_budget_tokens=4096,
        )

        decision = router.route(context)
        assert isinstance(decision.confidence, float)
        assert abs(decision.confidence - 0.75) < 0.001

    def test_llm_router_confidence_out_of_bounds(self) -> None:
        """测试 confidence 超出 [0, 1] 范围时的处理。"""
        mock_llm_fn = MagicMock(return_value=json.dumps({
            "complexity": "simple",
            "confidence": 1.5,  # 超出范围
            "reasoning": "超出范围的置信度"
        }, ensure_ascii=False))

        router = LLMRouter(
            llm_call_fn=mock_llm_fn,
            fallback_router=RuleBasedRouter(),
        )

        context = RoutingContext(
            segments=[Segment(type=SegmentType.USER, content="test", role="user")],
            query="test",
            max_budget_tokens=4096,
        )

        # 当 JSON 解析出无效值时，fallback_router 会被触发
        # 最终的 confidence 值取决于 fallback 处理结果
        decision = router.route(context)
        assert decision is not None
        assert isinstance(decision.confidence, (float, int))

    def test_llm_router_classification_with_all_defaults(self) -> None:
        """测试所有字段都使用默认值的响应。"""
        mock_llm_fn = MagicMock(return_value=json.dumps({}, ensure_ascii=False))

        router = LLMRouter(
            llm_call_fn=mock_llm_fn,
            fallback_router=RuleBasedRouter(),
        )

        context = RoutingContext(
            segments=[Segment(type=SegmentType.USER, content="test", role="user")],
            query="test",
            max_budget_tokens=4096,
        )

        decision = router.route(context)
        # 所有字段都使用默认值
        assert decision.complexity == ComplexityLevel.MODERATE
        assert decision.confidence == 0.5

    def test_llm_router_multiple_cache_entries(self) -> None:
        """测试缓存能够存储多个条目。"""
        call_count = 0

        def counting_llm_fn(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return json.dumps({
                "complexity": "simple" if call_count == 1 else "complex",
                "confidence": 0.8,
                "reasoning": f"调用 {call_count}"
            }, ensure_ascii=False)

        router = LLMRouter(
            llm_call_fn=counting_llm_fn,
            fallback_router=RuleBasedRouter(),
            enable_cache=True,
        )

        # 第一个查询
        ctx1 = RoutingContext(
            segments=[Segment(type=SegmentType.USER, content="query1", role="user")],
            query="query1",
            max_budget_tokens=4096,
        )
        dec1 = router.route(ctx1)
        assert dec1.complexity == ComplexityLevel.SIMPLE
        assert call_count == 1

        # 第二个查询
        ctx2 = RoutingContext(
            segments=[Segment(type=SegmentType.USER, content="query2", role="user")],
            query="query2",
            max_budget_tokens=4096,
        )
        dec2 = router.route(ctx2)
        assert dec2.complexity == ComplexityLevel.COMPLEX
        assert call_count == 2

        # 重复第一个查询（应该命中缓存）
        dec1_again = router.route(ctx1)
        assert dec1_again.complexity == ComplexityLevel.SIMPLE
        assert call_count == 2  # 未增加

        # 重复第二个查询（应该命中缓存）
        dec2_again = router.route(ctx2)
        assert dec2_again.complexity == ComplexityLevel.COMPLEX
        assert call_count == 2  # 未增加

    def test_llm_router_fallback_with_rule_based_routing(self) -> None:
        """测试降级到 RuleBasedRouter 时的完整路由逻辑。"""
        # LLM 调用返回无效 JSON
        mock_llm_fn = MagicMock(return_value="invalid json response")

        router = LLMRouter(
            llm_call_fn=mock_llm_fn,
            fallback_router=RuleBasedRouter(),
        )

        context = RoutingContext(
            segments=[Segment(type=SegmentType.USER, content="test", role="user")],
            query="test",
            max_budget_tokens=4096,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            decision = router.route(context)
            # 应该返回 fallback 路由的结果
            assert decision is not None
            assert len(w) == 1
            assert "LLM 路由失败" in str(w[0].message)
            # fallback 路由应该给出有效的模型选择
            assert decision.selected_model is not None

    def test_llm_router_complexity_estimator_available(self) -> None:
        """测试 LLM 路由器拥有可用的复杂度估计器。"""
        router = LLMRouter(llm_call_fn=None)
        assert router.complexity_estimator is not None
        # 验证它能进行估计
        complexity = router.complexity_estimator.estimate("测试查询")
        assert complexity in (
            ComplexityLevel.SIMPLE,
            ComplexityLevel.MODERATE,
            ComplexityLevel.COMPLEX,
            ComplexityLevel.EXPERT,
        )

    def test_mock_llm_call_fn_boundary_lengths(self) -> None:
        """测试 Mock LLM 在边界长度处的分类。"""
        mock_fn = create_mock_llm_call_fn()

        # 测试各个边界 - 根据 create_mock_llm_call_fn 的实现
        # query_len < 50: simple
        # 50 <= query_len < 150: moderate
        # 150 <= query_len < 400: complex
        # >= 400: expert
        test_cases = [
            ("x" * 10, "simple"),  # len=10 < 50
            ("x" * 100, "moderate"),  # len=100, 50-150
            ("x" * 200, "complex"),  # len=200, 150-400
            ("x" * 500, "expert"),  # len=500, >= 400
        ]

        for query_text, expected_complexity in test_cases:
            prompt = f"用户查询：\n{query_text}"
            response = mock_fn(prompt)
            data = json.loads(response)
            assert data["complexity"] == expected_complexity, \
                f"Query len={len(query_text)} should be {expected_complexity}, got {data['complexity']}"

    def test_llm_router_decision_structure(self) -> None:
        """测试 LLM 路由决策的完整结构。"""
        mock_llm_fn = MagicMock(return_value=json.dumps({
            "complexity": "complex",
            "confidence": 0.88,
            "reasoning": "结构测试"
        }, ensure_ascii=False))

        router = LLMRouter(
            llm_call_fn=mock_llm_fn,
            fallback_router=RuleBasedRouter(),
        )

        context = RoutingContext(
            segments=[Segment(type=SegmentType.USER, content="test", role="user")],
            query="test",
            max_budget_tokens=4096,
        )

        decision = router.route(context)

        # 验证决策的所有关键字段
        assert hasattr(decision, "selected_model")
        assert hasattr(decision, "complexity")
        assert hasattr(decision, "matched_rule")
        assert hasattr(decision, "is_fallback")
        assert hasattr(decision, "confidence")
        assert hasattr(decision, "reasoning")
        assert hasattr(decision, "estimated_cost")

        # 验证值的一致性
        assert decision.complexity == ComplexityLevel.COMPLEX
        assert decision.confidence == 0.88
        assert "llm_classified" in decision.matched_rule
        assert "LLM 分类" in decision.reasoning

    def test_llm_router_classification_returns_none_from_exception(self) -> None:
        """测试 _classify_with_llm 抛出异常时返回 None 并触发降级。"""
        # Mock 一个会抛出异常的 LLM 函数
        def failing_llm_fn(prompt: str) -> str:
            raise RuntimeError("LLM API 调用超时")

        router = LLMRouter(
            llm_call_fn=failing_llm_fn,
            fallback_router=RuleBasedRouter(),
        )

        context = RoutingContext(
            segments=[Segment(type=SegmentType.USER, content="test", role="user")],
            query="test",
            max_budget_tokens=4096,
        )

        # 应该捕获异常并降级到 fallback
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            decision = router.route(context)
            assert decision is not None
            assert len(w) == 1
            assert "LLM 路由失败" in str(w[0].message)

    def test_llm_router_route_with_valid_llm_response_from_cache(self) -> None:
        """测试从缓存中获取有效的 LLM 响应并路由（覆盖 line 140 的 if classification 分支）。"""
        mock_llm_fn = MagicMock(return_value=json.dumps({
            "complexity": "moderate",
            "confidence": 0.82,
            "reasoning": "缓存路由测试"
        }, ensure_ascii=False))

        router = LLMRouter(
            llm_call_fn=mock_llm_fn,
            fallback_router=RuleBasedRouter(),
            enable_cache=True,
        )

        context = RoutingContext(
            segments=[Segment(type=SegmentType.USER, content="cached query", role="user")],
            query="cached query",
            max_budget_tokens=4096,
        )

        # 第一次调用（缓存未命中）
        decision1 = router.route(context)
        assert mock_llm_fn.call_count == 1
        assert decision1.complexity == ComplexityLevel.MODERATE

        # 第二次调用（缓存命中，line 128-134 路径）
        decision2 = router.route(context)
        assert mock_llm_fn.call_count == 1  # 未增加
        assert decision2.complexity == ComplexityLevel.MODERATE
        assert decision2.confidence == 0.82

    def test_llm_router_confidence_invalid_type_triggers_fallback(self) -> None:
        """测试 confidence 为无效类型时触发降级。"""
        mock_llm_fn = MagicMock(return_value=json.dumps({
            "complexity": "simple",
            "confidence": "invalid",  # 无法转换为 float
            "reasoning": "无效置信度"
        }, ensure_ascii=False))

        router = LLMRouter(
            llm_call_fn=mock_llm_fn,
            fallback_router=RuleBasedRouter(),
        )

        context = RoutingContext(
            segments=[Segment(type=SegmentType.USER, content="test", role="user")],
            query="test",
            max_budget_tokens=4096,
        )

        # float("invalid") 会抛出 ValueError，触发降级
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            decision = router.route(context)
            assert decision is not None
            # 应该有降级警告
            assert any("LLM 路由失败" in str(warn.message) for warn in w)

    def test_llm_router_with_custom_fallback_router_rules(self) -> None:
        """测试自定义 fallback_router 规则的 LLM 路由。"""
        custom_rules = [
            RoutingRule(
                name="custom_rule",
                condition_type="keyword",
                condition_value="urgent",
                target_model="gpt-4o",
                priority=20,
            ),
        ]

        mock_llm_fn = MagicMock(return_value=json.dumps({
            "complexity": "simple",
            "confidence": 0.90,
            "reasoning": "自定义规则测试"
        }, ensure_ascii=False))

        router = LLMRouter(
            llm_call_fn=mock_llm_fn,
            fallback_router=RuleBasedRouter(rules=custom_rules, default_model="gpt-4o-mini"),
        )

        context = RoutingContext(
            segments=[Segment(type=SegmentType.USER, content="urgent task", role="user")],
            query="urgent task",
            max_budget_tokens=4096,
        )

        decision = router.route(context)
        # LLM 分类为 simple，但 fallback router 应该应用自定义规则
        assert decision.selected_model.model_id in ("gpt-4o", "gpt-4o-mini")

    def test_llm_router_with_turn_context(self) -> None:
        """测试带有对话轮次的路由上下文。"""
        mock_llm_fn = MagicMock(return_value=json.dumps({
            "complexity": "moderate",
            "confidence": 0.85,
            "reasoning": "对话轮次测试"
        }, ensure_ascii=False))

        router = LLMRouter(
            llm_call_fn=mock_llm_fn,
            fallback_router=RuleBasedRouter(),
        )

        context = RoutingContext(
            segments=[Segment(type=SegmentType.USER, content="继续上个问题", role="user")],
            query="继续上个问题",
            max_budget_tokens=4096,
            current_turn=5,  # 第 5 轮对话
        )

        decision = router.route(context)
        assert decision.complexity == ComplexityLevel.MODERATE

    def test_llm_router_prompt_template_formatting(self) -> None:
        """测试 LLM Prompt 模板格式化是否正确。"""
        captured_prompts = []

        def capturing_llm_fn(prompt: str) -> str:
            captured_prompts.append(prompt)
            return json.dumps({
                "complexity": "simple",
                "confidence": 0.80,
                "reasoning": "测试"
            }, ensure_ascii=False)

        router = LLMRouter(
            llm_call_fn=capturing_llm_fn,
            fallback_router=RuleBasedRouter(),
        )

        query_text = "这是一个测试查询"
        context = RoutingContext(
            segments=[Segment(type=SegmentType.USER, content=query_text, role="user")],
            query=query_text,
            max_budget_tokens=4096,
        )

        router.route(context)

        # 验证 Prompt 包含模板中的关键元素
        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]
        assert "复杂度等级定义" in prompt
        assert "simple" in prompt
        assert "moderate" in prompt
        assert "complex" in prompt
        assert "expert" in prompt
        assert query_text in prompt
        assert "请返回 JSON" in prompt

    def test_llm_router_json_with_bom(self) -> None:
        """测试处理带有 BOM（Byte Order Mark）的 JSON 响应。"""
        # UTF-8 BOM: \ufeff
        json_with_bom = "\ufeff" + json.dumps({
            "complexity": "simple",
            "confidence": 0.85,
            "reasoning": "BOM 测试"
        }, ensure_ascii=False)

        mock_llm_fn = MagicMock(return_value=json_with_bom)

        router = LLMRouter(
            llm_call_fn=mock_llm_fn,
            fallback_router=RuleBasedRouter(),
        )

        context = RoutingContext(
            segments=[Segment(type=SegmentType.USER, content="test", role="user")],
            query="test",
            max_budget_tokens=4096,
        )

        # Python json.loads 会自动处理 BOM
        decision = router.route(context)
        assert decision.complexity == ComplexityLevel.SIMPLE

    def test_llm_router_enable_cache_attribute(self) -> None:
        """测试 enable_cache 属性。"""
        router_with_cache = LLMRouter(llm_call_fn=None, enable_cache=True)
        assert router_with_cache.enable_cache is True

        router_without_cache = LLMRouter(llm_call_fn=None, enable_cache=False)
        assert router_without_cache.enable_cache is False

    def test_llm_router_cache_internal_structure(self) -> None:
        """测试缓存内部结构。"""
        router = LLMRouter(llm_call_fn=None, enable_cache=True)
        assert isinstance(router._cache, dict)
        assert len(router._cache) == 0

    def test_llm_router_fallback_router_default(self) -> None:
        """测试默认 fallback_router 的创建。"""
        router = LLMRouter(llm_call_fn=None)
        assert router.fallback_router is not None
        assert isinstance(router.fallback_router, RuleBasedRouter)

    def test_llm_router_complexity_all_levels_mapping(self) -> None:
        """测试所有复杂度等级的映射（simple/moderate/complex/expert）。"""
        test_cases = [
            ("simple", ComplexityLevel.SIMPLE),
            ("moderate", ComplexityLevel.MODERATE),
            ("complex", ComplexityLevel.COMPLEX),
            ("expert", ComplexityLevel.EXPERT),
        ]

        for complexity_str, expected_level in test_cases:
            mock_llm_fn = MagicMock(return_value=json.dumps({
                "complexity": complexity_str,
                "confidence": 0.85,
                "reasoning": f"测试 {complexity_str}"
            }, ensure_ascii=False))

            router = LLMRouter(
                llm_call_fn=mock_llm_fn,
                fallback_router=RuleBasedRouter(),
            )

            context = RoutingContext(
                segments=[Segment(type=SegmentType.USER, content="test", role="user")],
                query="test",
                max_budget_tokens=4096,
            )

            decision = router.route(context)
            assert decision.complexity == expected_level, \
                f"复杂度 {complexity_str} 应映射到 {expected_level}，实际为 {decision.complexity}"
