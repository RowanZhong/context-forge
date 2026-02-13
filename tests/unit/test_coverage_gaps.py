"""
覆盖率缺口补充测试。

针对以下模块中未覆盖的代码路径编写测试，
目标是将总体覆盖率从 84.26% 提升到 85%+。

目标模块:
- pipeline/assemble.py (69% → 85%+)
- routing/rule_based.py (67% → 85%+)
- routing/context_bus.py (63% → 80%+)
- observability/diff.py (67% → 80%+)
- observability/golden_set.py (67% → 80%+)
- observability/snapshot.py (76% → 85%+)
- tokenizer/registry.py (79% → 90%+)
"""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from context_forge.models.budget import BudgetAllocation
from context_forge.models.context_package import ContextPackage
from context_forge.models.control import ControlFlags, Visibility
from context_forge.models.routing import ComplexityLevel, RoutingRule
from context_forge.models.segment import Priority, Segment, SegmentType
from context_forge.pipeline.base import PipelineContext


# ============================================================
# 1. pipeline/assemble.py — 补充 merge_adjacent + debug 路径
# ============================================================


class TestAssembleStageMergeAdjacent:
    """测试 AssembleStage 的合并相邻同角色消息功能。"""

    @pytest.mark.asyncio
    async def test_merge_adjacent_same_role(self) -> None:
        """测试合并相邻同角色消息。"""
        from context_forge.pipeline.assemble import AssembleStage

        stage = AssembleStage(merge_adjacent=True)
        assert stage.name == "assemble"

        # 两条连续 USER 消息
        segments = [
            Segment(type=SegmentType.SYSTEM, content="系统提示", role="system",
                    control=ControlFlags(must_keep=True, lock_position=True)).with_token_count(10),
            Segment(type=SegmentType.USER, content="第一条消息", role="user").with_token_count(5),
            Segment(type=SegmentType.USER, content="第二条消息", role="user").with_token_count(5),
        ]

        ctx = PipelineContext(model="gpt-4o", current_turn=1)
        result = await stage.process(segments, ctx)

        # SYSTEM 保留，两条 USER 合并为一条
        assert len(result) == 2
        merged_content = result[1].content
        assert "第一条消息" in merged_content
        assert "第二条消息" in merged_content

    @pytest.mark.asyncio
    async def test_merge_adjacent_different_roles_no_merge(self) -> None:
        """测试不同角色消息不合并。"""
        from context_forge.pipeline.assemble import AssembleStage

        stage = AssembleStage(merge_adjacent=True)

        segments = [
            Segment(type=SegmentType.USER, content="用户消息", role="user").with_token_count(5),
            Segment(type=SegmentType.ASSISTANT, content="助手消息", role="assistant").with_token_count(5),
        ]

        ctx = PipelineContext(model="gpt-4o", current_turn=1)
        result = await stage.process(segments, ctx)

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_merge_adjacent_empty_list(self) -> None:
        """测试空列表不报错。"""
        from context_forge.pipeline.assemble import AssembleStage

        stage = AssembleStage(merge_adjacent=True)
        ctx = PipelineContext(model="gpt-4o")
        result = await stage.process([], ctx)
        assert result == []

    @pytest.mark.asyncio
    async def test_merge_adjacent_lock_position_prevents_merge(self) -> None:
        """测试 lock_position 阻止合并。"""
        from context_forge.pipeline.assemble import AssembleStage

        stage = AssembleStage(merge_adjacent=True)

        segments = [
            Segment(type=SegmentType.USER, content="消息一", role="user",
                    control=ControlFlags(lock_position=True)).with_token_count(5),
            Segment(type=SegmentType.USER, content="消息二", role="user").with_token_count(5),
        ]

        ctx = PipelineContext(model="gpt-4o", current_turn=1)
        result = await stage.process(segments, ctx)

        # lock_position 的 segment 不会被合并
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_debug_mode_logging(self) -> None:
        """测试 debug 模式输出日志。"""
        from context_forge.pipeline.assemble import AssembleStage

        stage = AssembleStage()

        segments = [
            Segment(type=SegmentType.SYSTEM, content="系统", role="system",
                    control=ControlFlags(lock_position=True)).with_token_count(10),
            Segment(type=SegmentType.USER, content="用户", role="user").with_token_count(20),
        ]

        ctx = PipelineContext(model="gpt-4o", debug=True)
        result = await stage.process(segments, ctx)

        # debug 模式不影响结果
        assert len(result) == 2
        # 审计日志应该被记录
        assert len(ctx.audit_log) == 2


# ============================================================
# 2. routing/rule_based.py — 补充 token_count / segment_type / fallback
# ============================================================


class TestRuleBasedRouterTokenCount:
    """测试 RuleBasedRouter 的 Token 数量路由。"""

    def _make_context(self, query: str, segments: list[Segment] | None = None) -> Any:
        from context_forge.routing.base import RoutingContext
        return RoutingContext(
            segments=segments or [],
            query=query,
            max_budget_tokens=128000,
        )

    def test_token_count_greater_than(self) -> None:
        """测试 ">1000" 条件。"""
        from context_forge.routing.rule_based import RuleBasedRouter

        rules = [
            RoutingRule(
                name="large_to_opus",
                condition_type="token_count",
                condition_value=">1000",
                target_model="gpt-4o",
                priority=10,
            ),
        ]

        router = RuleBasedRouter(rules=rules, default_model="gpt-4o-mini")

        # 超过 1000 tokens 的 segment
        segments = [
            Segment(type=SegmentType.USER, content="x" * 100, role="user").with_token_count(1500),
        ]
        ctx = self._make_context("测试", segments)
        decision = router.route(ctx)
        assert decision.matched_rule == "large_to_opus"

    def test_token_count_less_than(self) -> None:
        """测试 "<500" 条件。"""
        from context_forge.routing.rule_based import RuleBasedRouter

        rules = [
            RoutingRule(
                name="small_to_mini",
                condition_type="token_count",
                condition_value="<500",
                target_model="gpt-4o-mini",
                priority=10,
            ),
        ]

        router = RuleBasedRouter(rules=rules, default_model="gpt-4o")

        segments = [
            Segment(type=SegmentType.USER, content="你好", role="user").with_token_count(100),
        ]
        ctx = self._make_context("你好", segments)
        decision = router.route(ctx)
        assert decision.matched_rule == "small_to_mini"

    def test_token_count_range(self) -> None:
        """测试 "500-2000" 范围条件。"""
        from context_forge.routing.rule_based import RuleBasedRouter

        rules = [
            RoutingRule(
                name="mid_range",
                condition_type="token_count",
                condition_value="500-2000",
                target_model="gpt-4o",
                priority=10,
            ),
        ]

        router = RuleBasedRouter(rules=rules, default_model="gpt-4o-mini")

        segments = [
            Segment(type=SegmentType.USER, content="测试", role="user").with_token_count(1000),
        ]
        ctx = self._make_context("测试", segments)
        decision = router.route(ctx)
        assert decision.matched_rule == "mid_range"

    def test_token_count_exact_match(self) -> None:
        """测试精确匹配。"""
        from context_forge.routing.rule_based import RuleBasedRouter

        rules = [
            RoutingRule(
                name="exact",
                condition_type="token_count",
                condition_value="100",
                target_model="gpt-4o",
                priority=10,
            ),
        ]

        router = RuleBasedRouter(rules=rules, default_model="gpt-4o-mini")

        segments = [
            Segment(type=SegmentType.USER, content="测试", role="user").with_token_count(100),
        ]
        ctx = self._make_context("测试", segments)
        decision = router.route(ctx)
        assert decision.matched_rule == "exact"

    def test_token_count_invalid_expr(self) -> None:
        """测试无效的范围表达式。"""
        from context_forge.routing.rule_based import RuleBasedRouter

        rules = [
            RoutingRule(
                name="invalid",
                condition_type="token_count",
                condition_value="abc",
                target_model="gpt-4o",
                priority=10,
            ),
        ]

        router = RuleBasedRouter(rules=rules, default_model="gpt-4o-mini")
        ctx = self._make_context("测试", [])
        decision = router.route(ctx)
        # 无效表达式不匹配，使用默认模型
        assert decision.matched_rule == "default"

    def test_segment_type_present_routing(self) -> None:
        """测试 segment_type_present 条件。"""
        from context_forge.routing.rule_based import RuleBasedRouter

        rules = [
            RoutingRule(
                name="has_rag",
                condition_type="segment_type_present",
                condition_value="rag",
                target_model="gpt-4o",
                priority=10,
            ),
        ]

        router = RuleBasedRouter(rules=rules, default_model="gpt-4o-mini")

        segments = [
            Segment(type=SegmentType.RAG, content="检索内容", role="user").with_token_count(50),
        ]
        ctx = self._make_context("查找信息", segments)
        decision = router.route(ctx)
        assert decision.matched_rule == "has_rag"

    def test_unknown_condition_type_no_match(self) -> None:
        """测试未知条件类型不匹配。"""
        from context_forge.routing.rule_based import RuleBasedRouter

        rules = [
            RoutingRule(
                name="unknown",
                condition_type="unknown_type",
                condition_value="anything",
                target_model="gpt-4o",
                priority=10,
            ),
        ]

        router = RuleBasedRouter(rules=rules, default_model="gpt-4o-mini")
        ctx = self._make_context("测试", [])
        decision = router.route(ctx)
        assert decision.matched_rule == "default"


class TestRuleBasedRouterFallback:
    """测试 RuleBasedRouter 的降级路径。"""

    def _make_context(self, query: str = "测试") -> Any:
        from context_forge.routing.base import RoutingContext
        return RoutingContext(segments=[], query=query, max_budget_tokens=128000)

    def test_fallback_to_fallback_model(self) -> None:
        """测试目标模型不存在时使用 fallback_model。"""
        from context_forge.errors.exceptions import RoutingError
        from context_forge.routing.rule_based import RuleBasedRouter

        rules = [
            RoutingRule(
                name="always_match",
                condition_type="complexity",
                condition_value="simple",
                target_model="nonexistent-model-xyz",
                priority=10,
                fallback_model="gpt-4o",
            ),
        ]

        router = RuleBasedRouter(rules=rules, default_model="gpt-4o-mini")

        # Mock _resolve_model 让不存在的模型抛出 RoutingError（而非 ModelNotFoundError）
        original_resolve = router._resolve_model

        def patched_resolve(model_id: str) -> Any:
            if model_id == "nonexistent-model-xyz":
                raise RoutingError(what=f"模型 '{model_id}' 不存在")
            return original_resolve(model_id)

        router._resolve_model = patched_resolve  # type: ignore
        ctx = self._make_context()
        decision = router.route(ctx)
        # 应该使用 fallback 模型 gpt-4o
        assert decision.selected_model.model_id == "gpt-4o"

    def test_fallback_both_fail_use_default(self) -> None:
        """测试目标和 fallback 都不存在时使用默认模型。"""
        from context_forge.errors.exceptions import RoutingError
        from context_forge.routing.rule_based import RuleBasedRouter

        rules = [
            RoutingRule(
                name="always_match",
                condition_type="complexity",
                condition_value="simple",
                target_model="nonexistent-model-1",
                priority=10,
                fallback_model="nonexistent-model-2",
            ),
        ]

        router = RuleBasedRouter(rules=rules, default_model="gpt-4o-mini")

        original_resolve = router._resolve_model

        def patched_resolve(model_id: str) -> Any:
            if model_id.startswith("nonexistent"):
                raise RoutingError(what=f"模型 '{model_id}' 不存在")
            return original_resolve(model_id)

        router._resolve_model = patched_resolve  # type: ignore
        ctx = self._make_context()
        decision = router.route(ctx)
        assert decision.selected_model.model_id == "gpt-4o-mini"

    def test_fallback_disabled(self) -> None:
        """测试 enable_fallback=False 且目标不存在时使用默认模型。"""
        from context_forge.errors.exceptions import RoutingError
        from context_forge.routing.rule_based import RuleBasedRouter

        rules = [
            RoutingRule(
                name="always_match",
                condition_type="complexity",
                condition_value="simple",
                target_model="nonexistent-model-xyz",
                priority=10,
                fallback_model="gpt-4o",
            ),
        ]

        router = RuleBasedRouter(
            rules=rules, default_model="gpt-4o-mini", enable_fallback=False,
        )

        original_resolve = router._resolve_model

        def patched_resolve(model_id: str) -> Any:
            if model_id == "nonexistent-model-xyz":
                raise RoutingError(what=f"模型 '{model_id}' 不存在")
            return original_resolve(model_id)

        router._resolve_model = patched_resolve  # type: ignore
        ctx = self._make_context()
        decision = router.route(ctx)
        # fallback 关闭后应直接使用 default
        assert decision.selected_model.model_id == "gpt-4o-mini"

    def test_build_reasoning_with_features(self) -> None:
        """测试 _build_reasoning 包含各种特征标签。"""
        from context_forge.routing.rule_based import RuleBasedRouter

        rules = [
            RoutingRule(
                name="complex_match",
                condition_type="complexity",
                condition_value="complex",
                target_model="gpt-4o",
                priority=10,
            ),
        ]
        router = RuleBasedRouter(rules=rules, default_model="gpt-4o-mini")
        # 包含对比词和复杂任务词的查询
        ctx = self._make_context("请分析并比较两种算法的性能，设计一个高可用架构")
        decision = router.route(ctx)
        # reasoning 应该包含关键特征描述
        assert "匹配规则" in decision.reasoning or "默认模型" in decision.reasoning


class TestCreateDefaultComplexityRules:
    """测试 create_default_complexity_rules 辅助函数。"""

    def test_default_rules(self) -> None:
        from context_forge.routing.rule_based import create_default_complexity_rules
        rules = create_default_complexity_rules()
        assert len(rules) == 4
        assert all(isinstance(r, RoutingRule) for r in rules)
        # 规则应包含 expert/complex/moderate/simple
        names = {r.name for r in rules}
        assert "expert_to_opus" in names
        assert "simple_to_mini" in names


# ============================================================
# 3. routing/context_bus.py — 补充多 Agent 协调场景
# ============================================================


class TestContextBusRegistration:
    """测试 ContextBus Agent 注册和注销。"""

    def test_register_duplicate_agent_warns(self) -> None:
        """测试重复注册 Agent 发出警告。"""
        from context_forge.routing.base import AgentContext
        from context_forge.routing.context_bus import ContextBus

        bus = ContextBus()
        agent = AgentContext(agent_id="agent1", namespace="ns1", role="worker")

        bus.register_agent(agent)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bus.register_agent(agent)
            assert len(w) == 1
            assert "已存在" in str(w[0].message)

    def test_unregister_agent(self) -> None:
        """测试注销 Agent。"""
        from context_forge.routing.base import AgentContext
        from context_forge.routing.context_bus import ContextBus

        bus = ContextBus()
        agent = AgentContext(agent_id="agent1", namespace="ns1", role="worker")

        bus.register_agent(agent)
        bus.unregister_agent("agent1")
        # 再次注册不应该有警告
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bus.register_agent(agent)
            assert len(w) == 0

    def test_unregister_nonexistent_agent(self) -> None:
        """测试注销不存在的 Agent 不报错。"""
        from context_forge.routing.context_bus import ContextBus

        bus = ContextBus()
        bus.unregister_agent("nonexistent")  # 不应报错


class TestContextBusPublishAndVisibility:
    """测试 ContextBus 发布和可见性。"""

    def test_publish_segment_with_default_namespace(self) -> None:
        """测试发布 Segment 时 namespace 从 Agent 继承。"""
        from context_forge.routing.base import AgentContext
        from context_forge.routing.context_bus import ContextBus

        bus = ContextBus()
        agent = AgentContext(agent_id="a1", namespace="planning", role="planner")
        bus.register_agent(agent)

        seg = Segment(type=SegmentType.USER, content="测试", role="user")
        bus.publish_segment(agent, seg)

        visible = bus.get_visible_segments(agent)
        assert len(visible) == 1

    def test_visibility_agent_only(self) -> None:
        """测试 AGENT_ONLY 可见性。"""
        from context_forge.routing.base import AgentContext
        from context_forge.routing.context_bus import ContextBus

        bus = ContextBus()
        a1 = AgentContext(agent_id="a1", namespace="ns1", role="worker")
        a2 = AgentContext(agent_id="a2", namespace="ns2", role="worker")
        bus.register_agent(a1)
        bus.register_agent(a2)

        # 发布 AGENT_ONLY segment 到 ns1
        seg = Segment(
            type=SegmentType.USER, content="秘密", role="user",
            control=ControlFlags(visibility=Visibility.AGENT_ONLY, namespace="ns1"),
        )
        bus.publish_segment(a1, seg)

        # a1 可以看到（它在自己的 namespace 中）
        visible_a1 = bus.get_visible_segments(a1)
        assert len(visible_a1) == 1

        # a2 不在 ns1，看不到 AGENT_ONLY segment
        visible_a2 = bus.get_visible_segments(a2)
        # a2 从其他 namespace 只能看到 AGENT_ONLY 且 namespace == a2.namespace 的
        agent_only_visible = [s for s in visible_a2 if s.control.visibility == Visibility.AGENT_ONLY]
        # ns1 的 AGENT_ONLY segment namespace 是 ns1，不等于 ns2，所以不可见
        assert len(agent_only_visible) == 0

    def test_visibility_internal_hidden(self) -> None:
        """测试 INTERNAL 可见性。"""
        from context_forge.routing.base import AgentContext
        from context_forge.routing.context_bus import ContextBus

        bus = ContextBus()
        a1 = AgentContext(agent_id="a1", namespace="ns1", role="worker")
        bus.register_agent(a1)

        seg = Segment(
            type=SegmentType.USER, content="内部", role="user",
            control=ControlFlags(visibility=Visibility.INTERNAL, namespace="ns1"),
        )
        bus.publish_segment(a1, seg)

        # INTERNAL 的 segment 即使在自己的 namespace 也不可见（除了 namespace 自身列表中已过滤）
        visible = bus.get_visible_segments(a1)
        assert len(visible) == 0

    def test_get_visible_segments_include_default(self) -> None:
        """测试 default namespace 的公共 Segment 可见性。"""
        from context_forge.routing.base import AgentContext
        from context_forge.routing.context_bus import ContextBus

        bus = ContextBus()
        default_agent = AgentContext(agent_id="sys", namespace="default", role="system")
        other_agent = AgentContext(agent_id="a1", namespace="ns1", role="worker")
        bus.register_agent(default_agent)
        bus.register_agent(other_agent)

        # 在 default namespace 发布公共 Segment
        seg = Segment(
            type=SegmentType.SYSTEM, content="公共系统提示", role="system",
            control=ControlFlags(visibility=Visibility.ALL, namespace="default"),
        )
        bus.publish_segment(default_agent, seg)

        # other_agent 可以看到 default namespace 的公共 Segment
        visible = bus.get_visible_segments(other_agent, include_default=True)
        assert any(s.content == "公共系统提示" for s in visible)

        # include_default=False 则看不到
        visible_no_default = bus.get_visible_segments(other_agent, include_default=False)
        assert not any(s.content == "公共系统提示" for s in visible_no_default)


class TestContextBusHandoff:
    """测试 ContextBus 上下文移交。"""

    def test_handoff_all_segments(self) -> None:
        """测试移交所有非 INTERNAL Segment。"""
        from context_forge.routing.base import AgentContext
        from context_forge.routing.context_bus import ContextBus, HandoffRequest

        bus = ContextBus()
        a1 = AgentContext(agent_id="planner", namespace="planning", role="planner")
        a2 = AgentContext(agent_id="executor", namespace="execution", role="executor")
        bus.register_agent(a1)
        bus.register_agent(a2)

        # planner 发布 segment
        seg = Segment(type=SegmentType.USER, content="任务计划", role="user").with_token_count(50)
        bus.publish_segment(a1, seg)

        # handoff 到 executor
        request = HandoffRequest(
            from_agent_id="planner",
            to_agent_id="executor",
            reason="规划完成",
        )
        bus.handoff(request)

        # executor 应能看到移交的 Segment
        visible = bus.get_visible_segments(a2)
        assert len(visible) >= 1

    def test_handoff_specific_segment_ids(self) -> None:
        """测试移交指定 ID 的 Segment。"""
        from context_forge.routing.base import AgentContext
        from context_forge.routing.context_bus import ContextBus, HandoffRequest

        bus = ContextBus()
        a1 = AgentContext(agent_id="a1", namespace="ns1", role="worker")
        a2 = AgentContext(agent_id="a2", namespace="ns2", role="worker")
        bus.register_agent(a1)
        bus.register_agent(a2)

        seg1 = Segment(type=SegmentType.USER, content="内容一", role="user")
        seg2 = Segment(type=SegmentType.USER, content="内容二", role="user")
        bus.publish_segment(a1, seg1)
        bus.publish_segment(a1, seg2)

        # 仅移交 seg1
        request = HandoffRequest(
            from_agent_id="a1",
            to_agent_id="a2",
            segment_ids=[seg1.id],
            reason="部分移交",
        )
        bus.handoff(request)

        visible_a2 = bus.get_visible_segments(a2)
        assert len(visible_a2) >= 1

    def test_handoff_invalid_source_agent(self) -> None:
        """测试源 Agent 不存在时报错。"""
        from context_forge.routing.base import AgentContext
        from context_forge.routing.context_bus import ContextBus, HandoffRequest

        bus = ContextBus()
        a2 = AgentContext(agent_id="a2", namespace="ns2", role="worker")
        bus.register_agent(a2)

        request = HandoffRequest(
            from_agent_id="nonexistent",
            to_agent_id="a2",
        )
        with pytest.raises(ValueError, match="不存在"):
            bus.handoff(request)

    def test_handoff_invalid_target_agent(self) -> None:
        """测试目标 Agent 不存在时报错。"""
        from context_forge.routing.base import AgentContext
        from context_forge.routing.context_bus import ContextBus, HandoffRequest

        bus = ContextBus()
        a1 = AgentContext(agent_id="a1", namespace="ns1", role="worker")
        bus.register_agent(a1)

        request = HandoffRequest(
            from_agent_id="a1",
            to_agent_id="nonexistent",
        )
        with pytest.raises(ValueError, match="不存在"):
            bus.handoff(request)


class TestContextBusEvents:
    """测试 ContextBus 事件系统。"""

    def test_subscribe_and_publish_event(self) -> None:
        """测试事件订阅和发布。"""
        from context_forge.routing.context_bus import ContextBus, ContextEvent

        bus = ContextBus()
        bus.subscribe("agent1", "task_completed")

        event = ContextEvent(
            event_type="task_completed",
            publisher_id="agent2",
            data={"task": "分析"},
        )
        bus.publish_event(event)

        events = bus.get_recent_events(event_type="task_completed")
        assert len(events) == 1
        assert events[0].event_type == "task_completed"

    def test_unsubscribe(self) -> None:
        """测试取消订阅。"""
        from context_forge.routing.context_bus import ContextBus

        bus = ContextBus()
        bus.subscribe("agent1", "evt")
        bus.unsubscribe("agent1", "evt")
        # 确认取消订阅不报错
        bus.unsubscribe("agent1", "evt")  # 已经取消了，不报错

    def test_get_recent_events_all_types(self) -> None:
        """测试获取所有类型的最近事件。"""
        from context_forge.routing.context_bus import ContextBus, ContextEvent

        bus = ContextBus()
        for i in range(5):
            bus.publish_event(ContextEvent(
                event_type=f"type_{i % 2}",
                publisher_id="agent",
                data=i,
            ))

        all_events = bus.get_recent_events(limit=10)
        assert len(all_events) == 5

        filtered = bus.get_recent_events(event_type="type_0", limit=10)
        assert len(filtered) == 3  # i=0,2,4

    def test_event_history_overflow(self) -> None:
        """测试事件历史溢出截断。"""
        from context_forge.routing.context_bus import ContextBus, ContextEvent

        bus = ContextBus()
        bus._max_history_size = 5

        for i in range(10):
            bus.publish_event(ContextEvent(
                event_type="test",
                publisher_id="agent",
                data=i,
            ))

        assert len(bus._event_history) == 5


class TestContextBusNamespace:
    """测试 ContextBus namespace 管理。"""

    def test_clear_namespace(self) -> None:
        """测试清空 namespace。"""
        from context_forge.routing.base import AgentContext
        from context_forge.routing.context_bus import ContextBus

        bus = ContextBus()
        agent = AgentContext(agent_id="a1", namespace="ns1", role="worker")
        bus.register_agent(agent)

        bus.publish_segment(agent, Segment(
            type=SegmentType.USER, content="测试", role="user",
        ))

        assert len(bus.get_visible_segments(agent)) >= 1

        bus.clear_namespace("ns1")
        assert len(bus.get_visible_segments(agent)) == 0

    def test_get_namespace_stats(self) -> None:
        """测试获取 namespace 统计信息。"""
        from context_forge.routing.base import AgentContext
        from context_forge.routing.context_bus import ContextBus

        bus = ContextBus()
        agent = AgentContext(agent_id="a1", namespace="ns1", role="worker")
        bus.register_agent(agent)

        bus.publish_segment(agent, Segment(
            type=SegmentType.USER, content="用户消息", role="user",
        ).with_token_count(100))
        bus.publish_segment(agent, Segment(
            type=SegmentType.RAG, content="检索结果", role="user",
        ).with_token_count(200))

        stats = bus.get_namespace_stats("ns1")
        assert stats["namespace"] == "ns1"
        assert stats["segment_count"] == 2
        assert stats["total_tokens"] == 300
        assert "user" in stats["type_distribution"]
        assert "rag" in stats["type_distribution"]

    def test_get_namespace_stats_empty(self) -> None:
        """测试空 namespace 的统计。"""
        from context_forge.routing.context_bus import ContextBus

        bus = ContextBus()
        stats = bus.get_namespace_stats("empty_ns")
        assert stats["segment_count"] == 0
        assert stats["total_tokens"] == 0


# ============================================================
# 4. observability/diff.py — 补充 format_rich / budget diff / reorder
# ============================================================


class TestDiffEngineAdvanced:
    """测试 DiffEngine 的高级功能。"""

    def _make_package(
        self,
        segments: list[Segment] | None = None,
        model: str = "gpt-4o",
        policy_version: str = "1.0",
        budget: BudgetAllocation | None = None,
    ) -> ContextPackage:
        segs = segments or [
            Segment(type=SegmentType.SYSTEM, content="系统", role="system"),
        ]
        return ContextPackage(
            segments=segs,
            model=model,
            policy_version=policy_version,
            budget_allocation=budget,
        )

    @pytest.mark.asyncio
    async def test_diff_segments_modified(self) -> None:
        """测试 Segment 内容修改的检测。"""
        from context_forge.observability.diff import DiffEngine

        seg_id = "fixed_id"
        seg_old = Segment(id=seg_id, type=SegmentType.USER, content="旧内容", role="user")
        seg_new = Segment(id=seg_id, type=SegmentType.USER, content="新内容", role="user")

        old_pkg = self._make_package(segments=[seg_old])
        new_pkg = self._make_package(segments=[seg_new])

        engine = DiffEngine()
        diff = await engine.diff(old_pkg, new_pkg)

        modified_entries = [e for e in diff.entries if e.diff_type.value == "modified"]
        assert len(modified_entries) >= 1

    @pytest.mark.asyncio
    async def test_diff_segments_reordered(self) -> None:
        """测试 Segment 位置变化的检测。"""
        from context_forge.observability.diff import DiffEngine

        seg_a = Segment(id="seg_a", type=SegmentType.USER, content="A", role="user")
        seg_b = Segment(id="seg_b", type=SegmentType.USER, content="B", role="user")

        old_pkg = self._make_package(segments=[seg_a, seg_b])
        new_pkg = self._make_package(segments=[seg_b, seg_a])

        engine = DiffEngine()
        diff = await engine.diff(old_pkg, new_pkg)

        reordered = [e for e in diff.entries if e.diff_type.value == "reordered"]
        assert len(reordered) >= 1

    @pytest.mark.asyncio
    async def test_diff_budget_changes(self) -> None:
        """测试预算分配变化的检测。"""
        from context_forge.observability.diff import DiffEngine

        old_budget = BudgetAllocation(
            total_budget=100000,
            content_budget=90000,
            total_used=50000,
            rigid_used=20000,
            output_reserved=10000,
        )
        new_budget = BudgetAllocation(
            total_budget=128000,
            content_budget=120000,
            total_used=60000,
            rigid_used=25000,
            output_reserved=8000,
        )

        old_pkg = self._make_package(budget=old_budget)
        new_pkg = self._make_package(budget=new_budget)

        engine = DiffEngine()
        diff = await engine.diff(old_pkg, new_pkg)

        budget_entries = [e for e in diff.entries if e.diff_type.value == "budget_changed"]
        # total_budget 不同 + rigid_used 不同 = 2 entries
        assert len(budget_entries) >= 1

    @pytest.mark.asyncio
    async def test_diff_audit_log_dropped(self) -> None:
        """测试审计日志丢弃数量变化。"""
        from context_forge.models.audit import AuditEntry, DecisionType, ReasonCode
        from context_forge.observability.diff import DiffEngine

        # dropped_segments 是基于 audit_log 中 decision==DROP 的条目
        drop_entry = AuditEntry(
            segment_id="seg_dropped",
            pipeline_stage="allocate",
            decision=DecisionType.DROP,
            reason_code=ReasonCode.BUDGET_EXCEEDED,
            reason_detail="预算不足",
        )

        old_pkg = self._make_package()
        # 创建带 DROP 审计记录的新 package
        new_pkg = ContextPackage(
            segments=[Segment(type=SegmentType.SYSTEM, content="系统", role="system")],
            model="gpt-4o",
            audit_log=[drop_entry],
        )

        engine = DiffEngine()
        diff = await engine.diff(old_pkg, new_pkg)

        meta_entries = [
            e for e in diff.entries
            if "dropped" in e.path
        ]
        assert len(meta_entries) >= 1

    @pytest.mark.asyncio
    async def test_format_text_with_overflow(self) -> None:
        """测试 format_text 超过最大条目数时的截断。"""
        from context_forge.observability.diff import DiffEngine

        segments_old = []
        segments_new = []
        for i in range(60):
            segments_new.append(
                Segment(type=SegmentType.USER, content=f"新 Segment {i}", role="user")
            )

        old_pkg = self._make_package(segments=segments_old)
        new_pkg = self._make_package(segments=segments_new)

        engine = DiffEngine()
        diff = await engine.diff(old_pkg, new_pkg)

        text = engine.format_text(diff, max_entries=5)
        assert "还有" in text  # 应显示截断提示

    @pytest.mark.asyncio
    async def test_format_rich(self) -> None:
        """测试 Rich 格式化输出。"""
        from context_forge.observability.diff import DiffEngine

        seg = Segment(type=SegmentType.USER, content="新增", role="user")
        old_pkg = self._make_package(segments=[])
        new_pkg = self._make_package(segments=[seg])

        engine = DiffEngine()
        diff = await engine.diff(old_pkg, new_pkg)

        rich_text = engine.format_rich(diff)
        assert "Context Diff" in rich_text
        assert "[bold" in rich_text  # Rich 标记

    @pytest.mark.asyncio
    async def test_format_json(self) -> None:
        """测试 JSON 格式化输出。"""
        from context_forge.observability.diff import DiffEngine

        old_pkg = self._make_package()
        new_pkg = self._make_package(model="gpt-4o-mini")

        engine = DiffEngine()
        diff = await engine.diff(old_pkg, new_pkg)

        json_data = engine.format_json(diff)
        assert "summary" in json_data
        assert "entries" in json_data
        assert json_data["old_package"]["model"] == "gpt-4o"
        assert json_data["new_package"]["model"] == "gpt-4o-mini"

    def test_get_diff_color(self) -> None:
        """测试颜色映射。"""
        from context_forge.observability.diff import DiffEngine

        engine = DiffEngine()
        assert engine._get_diff_color("added") == "green"
        assert engine._get_diff_color("removed") == "red"
        assert engine._get_diff_color("modified") == "yellow"
        assert engine._get_diff_color("unknown_type") == "white"


# ============================================================
# 5. observability/golden_set.py — 补充批量/过滤/摘要/自定义断言
# ============================================================


class TestGoldenSetRunnerAdvanced:
    """测试 GoldenSetRunner 的高级功能。"""

    def test_add_cases_batch(self) -> None:
        """测试批量添加 Golden Case。"""
        from context_forge.observability.golden_set import (
            GoldenCase,
            GoldenSetRunner,
            GoldenTolerance,
        )

        runner = GoldenSetRunner()
        cases = [
            GoldenCase(
                name=f"case_{i}",
                description=f"测试用例 {i}",
                build_inputs={"system_prompt": "test"},
                expected_outputs={"segment_count": 1},
            )
            for i in range(3)
        ]
        runner.add_cases(cases)
        assert len(runner.cases) == 3

    @pytest.mark.asyncio
    async def test_run_with_filter_tags(self) -> None:
        """测试带标签过滤的运行。"""
        from context_forge.observability.golden_set import (
            GoldenCase,
            GoldenSetRunner,
        )

        runner = GoldenSetRunner()
        runner.add_case(GoldenCase(
            name="rag_case",
            description="RAG 场景",
            build_inputs={"system_prompt": "test"},
            expected_outputs={"segment_count": 1},
            tags={"scenario": "rag"},
        ))
        runner.add_case(GoldenCase(
            name="chat_case",
            description="对话场景",
            build_inputs={"system_prompt": "test"},
            expected_outputs={"segment_count": 1},
            tags={"scenario": "chat"},
        ))

        async def mock_build(**kwargs: Any) -> ContextPackage:
            return ContextPackage(
                segments=[Segment(type=SegmentType.SYSTEM, content="test", role="system")],
                model="gpt-4o",
            )

        # 只运行 rag 场景
        results = await runner.run(mock_build, filter_tags={"scenario": "rag"})
        assert len(results) == 1
        assert results[0].case.name == "rag_case"

    @pytest.mark.asyncio
    async def test_run_build_failure(self) -> None:
        """测试构建失败时的处理。"""
        from context_forge.observability.golden_set import GoldenCase, GoldenSetRunner

        runner = GoldenSetRunner()
        runner.add_case(GoldenCase(
            name="fail_case",
            description="会失败的用例",
            build_inputs={"system_prompt": "test"},
            expected_outputs={"total_tokens": 100},
        ))

        async def failing_build(**kwargs: Any) -> ContextPackage:
            raise RuntimeError("构建失败")

        results = await runner.run(failing_build)
        assert len(results) == 1
        assert not results[0].passed
        assert "构建失败" in results[0].error

    @pytest.mark.asyncio
    async def test_summary_output(self) -> None:
        """测试人类可读摘要生成。"""
        from context_forge.observability.golden_set import GoldenCase, GoldenSetRunner

        runner = GoldenSetRunner()
        runner.add_case(GoldenCase(
            name="pass_case",
            description="通过的用例",
            build_inputs={"system_prompt": "test"},
            expected_outputs={"segment_count": 1},
        ))
        runner.add_case(GoldenCase(
            name="fail_case",
            description="失败的用例",
            build_inputs={"system_prompt": "test"},
            expected_outputs={"segment_count": 99},
        ))

        async def mock_build(**kwargs: Any) -> ContextPackage:
            return ContextPackage(
                segments=[Segment(type=SegmentType.SYSTEM, content="test", role="system")],
                model="gpt-4o",
            )

        results = await runner.run(mock_build)

        summary_text = runner.summary(results)
        assert "通过: 1" in summary_text
        assert "失败: 1" in summary_text
        assert "fail_case" in summary_text

    @pytest.mark.asyncio
    async def test_passed_count_and_failed_cases(self) -> None:
        """测试 passed_count 和 failed_cases 方法。"""
        from context_forge.observability.golden_set import GoldenCase, GoldenSetRunner

        runner = GoldenSetRunner()
        for i in range(3):
            runner.add_case(GoldenCase(
                name=f"case_{i}",
                description=f"用例 {i}",
                build_inputs={"system_prompt": "test"},
                expected_outputs={"segment_count": 1 if i < 2 else 99},
            ))

        async def mock_build(**kwargs: Any) -> ContextPackage:
            return ContextPackage(
                segments=[Segment(type=SegmentType.SYSTEM, content="test", role="system")],
                model="gpt-4o",
            )

        results = await runner.run(mock_build)
        assert runner.passed_count(results) == 2
        assert len(runner.failed_cases(results)) == 1

    @pytest.mark.asyncio
    async def test_custom_assertions(self) -> None:
        """测试自定义断言函数。"""
        from context_forge.observability.golden_set import (
            GoldenCase,
            GoldenSetRunner,
            GoldenTolerance,
        )

        runner = GoldenSetRunner()

        # 自定义断言：Segment 数量必须 >= 1
        def check_has_segments(pkg: ContextPackage) -> bool:
            return len(pkg.segments) >= 1

        # 自定义断言：会抛异常的
        def check_raises(pkg: ContextPackage) -> bool:
            raise ValueError("自定义断言出错了")

        runner.add_case(GoldenCase(
            name="custom_test",
            description="自定义断言测试",
            build_inputs={"system_prompt": "test"},
            expected_outputs={},
            tolerance=GoldenTolerance(
                custom_assertions=[check_has_segments, check_raises],
            ),
        ))

        async def mock_build(**kwargs: Any) -> ContextPackage:
            return ContextPackage(
                segments=[Segment(type=SegmentType.SYSTEM, content="test", role="system")],
                model="gpt-4o",
            )

        results = await runner.run(mock_build)
        assert len(results) == 1
        # 第一个自定义断言通过，第二个失败
        assertions = results[0].assertions
        assert len(assertions) == 2
        assert assertions[0].passed is True
        assert assertions[1].passed is False
        assert "出错" in assertions[1].message

    @pytest.mark.asyncio
    async def test_segment_types_assertion(self) -> None:
        """测试 segment_types 断言。"""
        from context_forge.observability.golden_set import GoldenCase, GoldenSetRunner

        runner = GoldenSetRunner()
        runner.add_case(GoldenCase(
            name="type_check",
            description="类型检查",
            build_inputs={"system_prompt": "test"},
            expected_outputs={"segment_types": {"system": 1}},
        ))

        async def mock_build(**kwargs: Any) -> ContextPackage:
            return ContextPackage(
                segments=[Segment(type=SegmentType.SYSTEM, content="test", role="system")],
                model="gpt-4o",
            )

        results = await runner.run(mock_build)
        assert results[0].passed

    def test_compare_with_tolerance_zero_expected(self) -> None:
        """测试容差比较当期望值为 0 时。"""
        from context_forge.observability.golden_set import GoldenSetRunner

        runner = GoldenSetRunner()
        passed, msg = runner._compare_with_tolerance(0, 0, 0.05)
        assert passed
        assert "符合" in msg

    def test_match_tags(self) -> None:
        """测试标签匹配。"""
        from context_forge.observability.golden_set import GoldenSetRunner

        runner = GoldenSetRunner()
        assert runner._match_tags({"a": "1", "b": "2"}, {"a": "1"})
        assert not runner._match_tags({"a": "1"}, {"a": "2"})
        assert not runner._match_tags({}, {"a": "1"})


# ============================================================
# 6. observability/snapshot.py — 补充搜索/删除/自动清理
# ============================================================


class TestSnapshotManagerAdvanced:
    """测试 SnapshotManager 的高级功能。"""

    def _make_package(self, model: str = "gpt-4o") -> ContextPackage:
        return ContextPackage(
            segments=[Segment(type=SegmentType.SYSTEM, content="test", role="system")],
            model=model,
        )

    @pytest.mark.asyncio
    async def test_search_by_tags(self, tmp_path: Path) -> None:
        """测试按标签搜索 Snapshot。"""
        from context_forge.observability.snapshot import SnapshotManager

        manager = SnapshotManager(storage_dir=tmp_path)

        await manager.save(self._make_package(), tags={"env": "prod"})
        await manager.save(self._make_package(), tags={"env": "dev"})

        prod_results = await manager.search(tags={"env": "prod"})
        assert len(prod_results) == 1
        assert prod_results[0].tags["env"] == "prod"

    @pytest.mark.asyncio
    async def test_search_by_model(self, tmp_path: Path) -> None:
        """测试按模型搜索 Snapshot。"""
        from context_forge.observability.snapshot import SnapshotManager

        manager = SnapshotManager(storage_dir=tmp_path)

        await manager.save(self._make_package("gpt-4o"))
        await manager.save(self._make_package("gpt-4o-mini"))

        results = await manager.search(model="gpt-4o-mini")
        assert len(results) == 1
        assert results[0].model == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_delete_snapshot(self, tmp_path: Path) -> None:
        """测试删除 Snapshot。"""
        from context_forge.observability.snapshot import SnapshotManager

        manager = SnapshotManager(storage_dir=tmp_path)
        sid = await manager.save(self._make_package())

        assert await manager.delete(sid) is True
        assert await manager.delete(sid) is False  # 已删除

    @pytest.mark.asyncio
    async def test_delete_nonexistent_snapshot(self, tmp_path: Path) -> None:
        """测试删除不存在的 Snapshot。"""
        from context_forge.observability.snapshot import SnapshotManager

        manager = SnapshotManager(storage_dir=tmp_path)
        assert await manager.delete("nonexistent_snap") is False

    @pytest.mark.asyncio
    async def test_load_nonexistent_snapshot(self, tmp_path: Path) -> None:
        """测试加载不存在的 Snapshot 报错。"""
        from context_forge.errors.exceptions import SerializationError
        from context_forge.observability.snapshot import SnapshotManager

        manager = SnapshotManager(storage_dir=tmp_path)
        with pytest.raises(SerializationError, match="不存在"):
            await manager.load("nonexistent_snap")

    @pytest.mark.asyncio
    async def test_auto_cleanup_old_snapshots(self, tmp_path: Path) -> None:
        """测试自动清理旧 Snapshot。"""
        from context_forge.observability.snapshot import SnapshotManager

        manager = SnapshotManager(storage_dir=tmp_path, auto_cleanup_days=1)

        sid = await manager.save(self._make_package())

        # 手动设置文件时间为 2 天前
        snap_file = tmp_path / f"{sid}.json"
        import os
        old_time = (datetime.now(timezone.utc) - timedelta(days=2)).timestamp()
        os.utime(snap_file, (old_time, old_time))

        # 再保存一个，触发自动清理
        await manager.save(self._make_package())

        # 旧的应该被清理
        assert not snap_file.exists()

    @pytest.mark.asyncio
    async def test_list_all(self, tmp_path: Path) -> None:
        """测试列出所有 Snapshot。"""
        from context_forge.observability.snapshot import SnapshotManager

        manager = SnapshotManager(storage_dir=tmp_path)

        for i in range(3):
            await manager.save(self._make_package())

        all_snapshots = await manager.list_all()
        assert len(all_snapshots) == 3

    @pytest.mark.asyncio
    async def test_search_with_corrupted_file(self, tmp_path: Path) -> None:
        """测试搜索时遇到损坏的文件不崩溃。"""
        from context_forge.observability.snapshot import SnapshotManager

        manager = SnapshotManager(storage_dir=tmp_path)

        # 保存正常的
        await manager.save(self._make_package())

        # 创建一个损坏的文件
        corrupted = tmp_path / "snap_corrupted_12345678.json"
        corrupted.write_text("this is not json", encoding="utf-8")

        # 搜索时跳过损坏的文件，不崩溃
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            results = await manager.search()
            # 至少有 1 个正常的
            assert len(results) >= 1


# ============================================================
# 7. tokenizer/registry.py — 补充注册/缓存/降级
# ============================================================


class TestTokenizerRegistry:
    """测试 Tokenizer 注册表。"""

    def test_register_custom_tokenizer(self) -> None:
        """测试注册自定义 Tokenizer。"""
        from context_forge.tokenizer.registry import (
            _custom_counters,
            clear_cache,
            get_tokenizer,
            register_tokenizer,
        )

        class MockCounter:
            def count(self, text: str) -> int:
                return len(text)

            def count_messages(self, messages: list[dict[str, str]]) -> int:
                return sum(len(m.get("content", "")) for m in messages)

            @property
            def name(self) -> str:
                return "mock_counter"

        try:
            register_tokenizer("test-model-xyz", MockCounter())
            counter = get_tokenizer("test-model-xyz")
            assert counter.name == "mock_counter"
            assert counter.count("hello") == 5
        finally:
            # 清理
            _custom_counters.pop("test-model-xyz", None)
            clear_cache()

    def test_clear_cache(self) -> None:
        """测试清除缓存。"""
        from context_forge.tokenizer.registry import _counter_cache, clear_cache, get_tokenizer

        # 先触发一次缓存
        get_tokenizer("gpt-4o")
        assert "gpt-4o" in _counter_cache

        clear_cache()
        assert "gpt-4o" not in _counter_cache

    def test_fallback_to_char_based(self) -> None:
        """测试未知模型回退到字符计数器。"""
        from context_forge.tokenizer.fallback import CharBasedCounter
        from context_forge.tokenizer.registry import _counter_cache, clear_cache, get_tokenizer

        try:
            counter = get_tokenizer("completely-unknown-model-xyz-2026")
            assert isinstance(counter, CharBasedCounter)
        finally:
            _counter_cache.pop("completely-unknown-model-xyz-2026", None)
            clear_cache()

    def test_tiktoken_creation_failure_fallback(self) -> None:
        """测试 tiktoken 创建失败时回退到字符计数器。"""
        from context_forge.tokenizer.fallback import CharBasedCounter
        from context_forge.tokenizer.registry import _counter_cache, clear_cache, get_tokenizer

        try:
            # 模拟 TiktokenCounter 创建失败
            with patch(
                "context_forge.tokenizer.registry.TiktokenCounter",
                side_effect=RuntimeError("tiktoken 不可用"),
            ):
                counter = get_tokenizer("gpt-4o-test-fail")
                assert isinstance(counter, CharBasedCounter)
        finally:
            _counter_cache.pop("gpt-4o-test-fail", None)
            clear_cache()

    def test_register_invalid_tokenizer_raises(self) -> None:
        """测试注册不符合协议的对象报错。"""
        from context_forge.tokenizer.registry import register_tokenizer

        with pytest.raises(TypeError, match="TokenCounter"):
            register_tokenizer("bad-model", "not a counter")  # type: ignore
