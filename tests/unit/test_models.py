"""
数据模型单元测试 — 测试所有 8 个模型文件。

覆盖范围:
- models/segment.py: Segment, SegmentType, Priority, DEFAULT_PRIORITY_MAP
- models/provenance.py: Provenance, SourceType
- models/control.py: ControlFlags, Visibility
- models/metadata.py: SegmentMetadata
- models/context_package.py: ContextPackage, TokenUsage
- models/budget.py: BudgetPolicy, BudgetAllocation, SpendType
- models/routing.py: ModelConfig, RoutingRule, RoutingDecision, ComplexityLevel
- models/audit.py: AuditEntry, DecisionType, ReasonCode
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from context_forge.models.audit import AuditEntry, DecisionType, ReasonCode
from context_forge.models.budget import BudgetAllocation, BudgetPolicy, SpendType
from context_forge.models.context_package import ContextPackage, TokenUsage
from context_forge.models.control import ControlFlags, Visibility
from context_forge.models.metadata import SegmentMetadata
from context_forge.models.provenance import Provenance, SourceType
from context_forge.models.routing import ComplexityLevel, ModelConfig, RoutingDecision, RoutingRule
from context_forge.models.segment import (
    DEFAULT_PRIORITY_MAP,
    Priority,
    Segment,
    SegmentType,
)


# === Segment 测试（~15 tests）===


class TestSegment:
    """Segment 模型测试。"""

    def test_create_basic_segment(self) -> None:
        """测试创建基础 Segment。"""
        seg = Segment(
            type=SegmentType.USER,
            content="测试内容",
            role="user",
        )
        assert seg.content == "测试内容"
        assert seg.type == SegmentType.USER
        assert seg.role == "user"
        assert seg.id.startswith("seg_")
        assert len(seg.id) == 16  # seg_ + 12 hex chars

    def test_auto_generated_id_unique(self) -> None:
        """测试 ID 自动生成且唯一。"""
        seg1 = Segment(type=SegmentType.USER, content="A", role="user")
        seg2 = Segment(type=SegmentType.USER, content="B", role="user")
        assert seg1.id != seg2.id

    def test_default_priority_assignment(self) -> None:
        """测试默认优先级自动分配。"""
        # SYSTEM → CRITICAL
        sys_seg = Segment(type=SegmentType.SYSTEM, content="", role="system")
        assert sys_seg.effective_priority == Priority.CRITICAL

        # USER → HIGH
        user_seg = Segment(type=SegmentType.USER, content="", role="user")
        assert user_seg.effective_priority == Priority.HIGH

        # RAG → MEDIUM
        rag_seg = Segment(type=SegmentType.RAG, content="", role="user")
        assert rag_seg.effective_priority == Priority.MEDIUM

    def test_explicit_priority_override(self) -> None:
        """测试显式指定优先级覆盖默认值。"""
        seg = Segment(
            type=SegmentType.RAG,
            content="",
            role="user",
            priority=Priority.HIGH,  # 覆盖默认的 MEDIUM
        )
        assert seg.effective_priority == Priority.HIGH

    def test_immutability(self) -> None:
        """测试 Segment 不可变性。"""
        seg = Segment(type=SegmentType.USER, content="原始", role="user")
        with pytest.raises((ValidationError, AttributeError)):
            seg.content = "尝试修改"  # type: ignore

    def test_with_content(self) -> None:
        """测试 with_content() 返回新对象。"""
        seg = Segment(type=SegmentType.USER, content="原始", role="user")
        new_seg = seg.with_content("新内容")

        assert seg.content == "原始"  # 原对象不变
        assert new_seg.content == "新内容"
        assert seg is not new_seg  # 不同的对象实例

    def test_with_token_count(self) -> None:
        """测试 with_token_count() 填充 token 计数。"""
        seg = Segment(type=SegmentType.USER, content="test", role="user")
        assert seg.token_count is None

        counted = seg.with_token_count(42)
        assert counted.token_count == 42
        assert seg.token_count is None  # 原对象不变

    def test_with_priority(self) -> None:
        """测试 with_priority() 更新优先级。"""
        seg = Segment(type=SegmentType.RAG, content="", role="user")
        assert seg.effective_priority == Priority.MEDIUM

        high_seg = seg.with_priority(Priority.HIGH)
        assert high_seg.effective_priority == Priority.HIGH
        assert seg.effective_priority == Priority.MEDIUM  # 原对象不变

    def test_to_message(self) -> None:
        """测试转换为 LLM API 消息格式。"""
        seg = Segment(
            type=SegmentType.USER,
            content="你好",
            role="user",
        )
        msg = seg.to_message()
        assert msg == {"role": "user", "content": "你好"}

    def test_control_flags_auto_created(self) -> None:
        """测试 ControlFlags 自动创建。"""
        seg = Segment(type=SegmentType.USER, content="", role="user")
        assert seg.control is not None
        assert isinstance(seg.control, ControlFlags)

    def test_metadata_auto_created(self) -> None:
        """测试 SegmentMetadata 自动创建。"""
        seg = Segment(type=SegmentType.USER, content="", role="user")
        assert seg.metadata is not None
        assert isinstance(seg.metadata, SegmentMetadata)

    def test_created_at_auto_filled(self) -> None:
        """测试 created_at 自动填充为当前时间。"""
        before = datetime.now(timezone.utc)
        seg = Segment(type=SegmentType.USER, content="", role="user")
        after = datetime.now(timezone.utc)

        assert before <= seg.created_at <= after

    def test_all_segment_types_have_priority(self) -> None:
        """测试所有 SegmentType 都有默认优先级。"""
        for seg_type in SegmentType:
            seg = Segment(type=seg_type, content="", role="user")
            assert seg.effective_priority in [
                Priority.CRITICAL,
                Priority.HIGH,
                Priority.MEDIUM,
                Priority.LOW,
            ]

    def test_segment_with_custom_provenance(self) -> None:
        """测试带自定义 Provenance 的 Segment。"""
        prov = Provenance(
            source_id="doc_001",
            source_type=SourceType.RAG_RETRIEVAL,
            retrieval_score=0.95,
        )
        seg = Segment(
            type=SegmentType.RAG,
            content="",
            role="user",
            provenance=prov,
        )
        assert seg.provenance == prov
        assert seg.provenance.retrieval_score == 0.95

    def test_segment_with_custom_control_flags(self) -> None:
        """测试带自定义 ControlFlags 的 Segment。"""
        flags = ControlFlags(must_keep=True, lock_position=True)
        seg = Segment(
            type=SegmentType.SYSTEM,
            content="",
            role="system",
            control=flags,
        )
        assert seg.control.must_keep is True
        assert seg.control.lock_position is True


class TestSegmentType:
    """SegmentType 枚举测试。"""

    def test_all_types_defined(self) -> None:
        """测试所有预期的类型都已定义。"""
        expected = {
            "SYSTEM",
            "USER",
            "ASSISTANT",
            "RAG",
            "TOOL_CALL",
            "TOOL_RESULT",
            "FEW_SHOT",
            "SUMMARY",
            "STATE",
            "SCHEMA",
            "TOOL_DEFINITION",
        }
        actual = {t.name for t in SegmentType}
        assert actual == expected

    def test_type_values_are_lowercase(self) -> None:
        """测试 SegmentType 的值是小写字符串。"""
        for seg_type in SegmentType:
            assert seg_type.value.islower() or seg_type.value.replace("_", "").islower()


class TestPriority:
    """Priority 枚举测试。"""

    def test_all_priorities_defined(self) -> None:
        """测试所有优先级都已定义。"""
        expected = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
        actual = {p.name for p in Priority}
        assert actual == expected

    def test_default_priority_map_coverage(self) -> None:
        """测试 DEFAULT_PRIORITY_MAP 覆盖所有 SegmentType。"""
        for seg_type in SegmentType:
            assert seg_type in DEFAULT_PRIORITY_MAP


# === Provenance 测试（~5 tests）===


class TestProvenance:
    """Provenance 模型测试。"""

    def test_create_basic_provenance(self) -> None:
        """测试创建基础 Provenance。"""
        prov = Provenance(
            source_id="test_001",
            source_type=SourceType.USER_INPUT,
        )
        assert prov.source_id == "test_001"
        assert prov.source_type == SourceType.USER_INPUT

    def test_provenance_with_retrieval_score(self) -> None:
        """测试带检索分数的 Provenance。"""
        prov = Provenance(
            source_id="doc_001",
            source_type=SourceType.RAG_RETRIEVAL,
            retrieval_score=0.92,
            uri="https://example.com/doc",
        )
        assert prov.retrieval_score == 0.92
        assert prov.uri == "https://example.com/doc"

    def test_provenance_immutable(self) -> None:
        """测试 Provenance 不可变性。"""
        prov = Provenance(source_id="test", source_type=SourceType.SYSTEM_CONFIG)
        with pytest.raises((ValidationError, AttributeError)):
            prov.source_id = "modified"  # type: ignore

    def test_provenance_with_custom_fields(self) -> None:
        """测试带自定义扩展字段的 Provenance。"""
        prov = Provenance(
            source_id="test",
            source_type=SourceType.SYSTEM_CONFIG,
            custom={"key": "value"},
        )
        assert prov.custom == {"key": "value"}

    def test_provenance_compression_provenance(self) -> None:
        """测试压缩溯源的 Provenance。"""
        prov = Provenance(
            source_id="summary_001",
            source_type=SourceType.COMPRESSION,
            parent_segment_ids=["seg_a1b2c3", "seg_d4e5f6"],
            compression_method="abstractive_summary",
        )
        assert prov.parent_segment_ids == ["seg_a1b2c3", "seg_d4e5f6"]
        assert prov.compression_method == "abstractive_summary"
        assert prov.is_derived is True

    def test_provenance_is_trusted_source(self) -> None:
        """测试 is_trusted_source 属性。"""
        system_prov = Provenance(source_id="sys", source_type=SourceType.SYSTEM_CONFIG)
        assert system_prov.is_trusted_source is True

        user_prov = Provenance(source_id="usr", source_type=SourceType.USER_INPUT)
        assert user_prov.is_trusted_source is False

    def test_source_type_enum(self) -> None:
        """测试 SourceType 枚举值。"""
        expected = {
            "USER_INPUT",
            "RAG_RETRIEVAL",
            "TOOL_RESPONSE",
            "SYSTEM_CONFIG",
            "COMPRESSION",
            "AGENT_HANDOFF",
            "MANUAL_INJECTION",
        }
        actual = {t.name for t in SourceType}
        assert actual == expected


# === ControlFlags 测试（~6 tests）===


class TestControlFlags:
    """ControlFlags 模型测试。"""

    def test_create_default_control_flags(self) -> None:
        """测试创建默认 ControlFlags。"""
        flags = ControlFlags()
        assert flags.must_keep is False
        assert flags.lock_position is False
        assert flags.compressible is True
        assert flags.ttl is None
        assert flags.namespace == "default"
        assert flags.visibility == Visibility.ALL

    def test_control_flags_with_ttl(self) -> None:
        """测试带 TTL 的 ControlFlags（轮次数）。"""
        flags = ControlFlags(ttl=5)
        assert flags.ttl == 5

    def test_control_flags_must_keep(self) -> None:
        """测试 must_keep 标志。"""
        flags = ControlFlags(must_keep=True)
        assert flags.must_keep is True

    def test_control_flags_lock_position(self) -> None:
        """测试 lock_position 标志。"""
        flags = ControlFlags(lock_position=True)
        assert flags.lock_position is True

    def test_control_flags_namespace(self) -> None:
        """测试 namespace 字段。"""
        flags = ControlFlags(namespace="tools")
        assert flags.namespace == "tools"

    def test_control_flags_with_methods(self) -> None:
        """测试 with_xxx 不可变更新方法。"""
        flags = ControlFlags()
        updated = flags.with_must_keep(True)
        assert updated.must_keep is True
        assert flags.must_keep is False  # 原对象不变

        ns_updated = flags.with_namespace("rag")
        assert ns_updated.namespace == "rag"
        assert flags.namespace == "default"

        ttl_updated = flags.with_ttl(10)
        assert ttl_updated.ttl == 10
        assert flags.ttl is None

    def test_control_flags_is_ephemeral(self) -> None:
        """测试 is_ephemeral 属性。"""
        normal = ControlFlags()
        assert normal.is_ephemeral is False

        turn_scoped = ControlFlags(turn_scoped=True)
        assert turn_scoped.is_ephemeral is True

        current_turn = ControlFlags(visibility=Visibility.CURRENT_TURN)
        assert current_turn.is_ephemeral is True

    def test_control_flags_is_protected(self) -> None:
        """测试 is_protected 属性。"""
        normal = ControlFlags()
        assert normal.is_protected is False

        must_keep = ControlFlags(must_keep=True)
        assert must_keep.is_protected is True

        not_compressible = ControlFlags(compressible=False)
        assert not_compressible.is_protected is True

    def test_control_flags_is_expired(self) -> None:
        """测试 is_expired 方法。"""
        flags = ControlFlags(ttl=5)
        assert flags.is_expired(current_turn=3, created_turn=0) is False
        assert flags.is_expired(current_turn=5, created_turn=0) is True
        assert flags.is_expired(current_turn=10, created_turn=0) is True

        no_ttl = ControlFlags()
        assert no_ttl.is_expired(current_turn=100, created_turn=0) is False

    def test_visibility_enum(self) -> None:
        """测试 Visibility 枚举。"""
        expected = {"ALL", "CURRENT_TURN", "AGENT_ONLY", "INTERNAL", "NAMESPACE", "DOWNSTREAM", "GLOBAL"}
        actual = {v.name for v in Visibility}
        assert actual == expected


# === SegmentMetadata 测试（~4 tests）===


class TestSegmentMetadata:
    """SegmentMetadata 模型测试。"""

    def test_create_empty_metadata(self) -> None:
        """测试创建空元数据。"""
        meta = SegmentMetadata()
        assert meta.retrieval_score is None
        assert meta.turn_number is None
        assert meta.rerank_score is None
        assert meta.debug_labels == {}

    def test_metadata_with_scores(self) -> None:
        """测试带分数的元数据。"""
        meta = SegmentMetadata(
            retrieval_score=0.95,
            rerank_score=0.88,
        )
        assert meta.retrieval_score == 0.95
        assert meta.rerank_score == 0.88

    def test_metadata_with_compression_info(self) -> None:
        """测试带压缩信息的元数据。"""
        meta = SegmentMetadata(
            compression_ratio=0.3,
            token_count_before=1000,
        )
        assert meta.compression_ratio == 0.3
        assert meta.token_count_before == 1000

    def test_metadata_with_debug_labels(self) -> None:
        """测试带调试标签的元数据。"""
        meta = SegmentMetadata(
            debug_labels={"category": "tech", "priority": "high"}
        )
        assert meta.debug_labels["category"] == "tech"
        assert meta.debug_labels["priority"] == "high"

    def test_metadata_with_methods(self) -> None:
        """测试 with_xxx 不可变更新方法。"""
        meta = SegmentMetadata()

        with_score = meta.with_rerank_score(0.9)
        assert with_score.rerank_score == 0.9
        assert meta.rerank_score is None

        with_budget = meta.with_budget(1024)
        assert with_budget.budget_allocated == 1024
        assert meta.budget_allocated is None

        with_compression = meta.with_compression(0.5, 2000)
        assert with_compression.compression_ratio == 0.5
        assert with_compression.token_count_before == 2000

        with_label = meta.with_debug_label("env", "test")
        assert with_label.debug_labels["env"] == "test"
        assert "env" not in meta.debug_labels


# === BudgetPolicy 测试（~8 tests）===


class TestBudgetPolicy:
    """BudgetPolicy 模型测试。"""

    def test_create_basic_budget_policy(self) -> None:
        """测试创建基础预算策略。"""
        policy = BudgetPolicy(
            max_context_tokens=8192,
            output_reserved_tokens=1024,
        )
        assert policy.max_context_tokens == 8192
        assert policy.output_reserved_tokens == 1024

    def test_available_for_content_calculation(self) -> None:
        """测试 available_for_content 属性计算。"""
        policy = BudgetPolicy(
            max_context_tokens=10000,
            output_reserved_tokens=1000,
            thinking_reserved_tokens=500,
        )
        # 10000 - 1000 - 500 = 8500
        assert policy.available_for_content == 8500

    def test_budget_policy_with_elastic_ratios(self) -> None:
        """测试带弹性区间比例的策略。"""
        policy = BudgetPolicy(
            max_context_tokens=8192,
            elastic_ratios={
                SegmentType.USER: 0.3,
                SegmentType.RAG: 0.25,
                SegmentType.ASSISTANT: 0.2,
            },
        )
        assert policy.elastic_ratios[SegmentType.RAG] == 0.25
        assert policy.elastic_ratios[SegmentType.USER] == 0.3

    def test_budget_policy_immutable(self) -> None:
        """测试预算策略不可变性。"""
        policy = BudgetPolicy(max_context_tokens=8192)
        with pytest.raises((ValidationError, AttributeError)):
            policy.max_context_tokens = 16384  # type: ignore

    def test_budget_policy_min_elastic_tokens(self) -> None:
        """测试最小弹性 Token 数设置。"""
        policy = BudgetPolicy(
            max_context_tokens=8192,
            min_elastic_tokens=512,
        )
        assert policy.min_elastic_tokens == 512

    def test_budget_policy_with_saturation_config(self) -> None:
        """测试带饱和度配置的预算策略。"""
        policy = BudgetPolicy(
            max_context_tokens=8192,
            saturation_threshold=0.9,
            overflow_strategy="compress",
        )
        assert policy.saturation_threshold == 0.9
        assert policy.overflow_strategy == "compress"

    def test_budget_policy_default_values(self) -> None:
        """测试预算策略的默认值。"""
        policy = BudgetPolicy()
        assert policy.max_context_tokens > 0
        assert policy.output_reserved_tokens >= 0
        assert policy.thinking_reserved_tokens >= 0
        assert isinstance(policy.rigid_segment_types, list)
        assert isinstance(policy.elastic_ratios, dict)

    def test_budget_policy_validation(self) -> None:
        """测试预算策略的验证逻辑。"""
        # 负数应该被拒绝
        with pytest.raises(ValidationError):
            BudgetPolicy(max_context_tokens=-1000)

    def test_budget_policy_elastic_budget_for(self) -> None:
        """测试 elastic_budget_for 方法。"""
        policy = BudgetPolicy(
            max_context_tokens=10000,
            elastic_ratios={
                SegmentType.RAG: 0.4,
                SegmentType.USER: 0.2,
            },
            min_elastic_tokens=100,
        )
        # available=5000, RAG ratio=0.4 → 2000
        assert policy.elastic_budget_for(SegmentType.RAG, 5000) == 2000
        # available=5000, USER ratio=0.2 → 1000
        assert policy.elastic_budget_for(SegmentType.USER, 5000) == 1000
        # 不在 ratios 中的类型返回 0
        assert policy.elastic_budget_for(SegmentType.SUMMARY, 5000) == 0

    def test_budget_policy_saturation_trigger_tokens(self) -> None:
        """测试 saturation_trigger_tokens 属性。"""
        policy = BudgetPolicy(
            max_context_tokens=10000,
            output_reserved_tokens=2000,
            thinking_reserved_tokens=0,
            saturation_threshold=0.85,
        )
        # available_for_content = 10000 - 2000 = 8000
        # trigger = 8000 * 0.85 = 6800
        assert policy.saturation_trigger_tokens == 6800


class TestBudgetAllocation:
    """BudgetAllocation 模型测试。"""

    def test_create_budget_allocation(self) -> None:
        """测试创建预算分配记录。"""
        alloc = BudgetAllocation(
            total_budget=8192,
            content_budget=7168,
            total_used=5000,
        )
        assert alloc.total_budget == 8192
        assert alloc.content_budget == 7168
        assert alloc.total_used == 5000

    def test_budget_allocation_with_breakdown(self) -> None:
        """测试带分类的预算分配。"""
        alloc = BudgetAllocation(
            total_budget=8192,
            content_budget=7168,
            total_used=5000,
            rigid_used=500,
            elastic_used={"rag": 2000, "user": 1500, "assistant": 500},
            output_reserved=1024,
            thinking_reserved=0,
        )
        assert alloc.rigid_used == 500
        assert alloc.elastic_used["rag"] == 2000
        assert alloc.elastic_used["user"] == 1500
        assert alloc.output_reserved == 1024

    def test_budget_allocation_saturation_rate(self) -> None:
        """测试饱和度计算。"""
        alloc = BudgetAllocation(
            total_budget=10000,
            content_budget=8000,
            total_used=6400,
        )
        assert alloc.saturation_rate == pytest.approx(0.8)

    def test_budget_allocation_remaining(self) -> None:
        """测试剩余预算计算。"""
        alloc = BudgetAllocation(
            total_budget=10000,
            content_budget=8000,
            total_used=5000,
        )
        assert alloc.remaining == 3000

    def test_budget_allocation_is_over_budget(self) -> None:
        """测试是否超出预算。"""
        normal = BudgetAllocation(
            total_budget=10000,
            content_budget=8000,
            total_used=5000,
        )
        assert normal.is_over_budget is False

        over = BudgetAllocation(
            total_budget=10000,
            content_budget=8000,
            total_used=9000,
        )
        assert over.is_over_budget is True

    def test_budget_allocation_summary(self) -> None:
        """测试预算摘要字符串。"""
        alloc = BudgetAllocation(
            total_budget=8192,
            content_budget=7168,
            total_used=5000,
            rigid_used=500,
            elastic_used={"rag": 2000},
            output_reserved=1024,
        )
        summary = alloc.summary()
        assert isinstance(summary, str)
        assert "8,192" in summary


class TestSpendType:
    """SpendType 枚举测试。"""

    def test_spend_type_values(self) -> None:
        """测试 SpendType 枚举值。"""
        expected = {"RIGID", "ELASTIC", "RESERVED"}
        actual = {t.name for t in SpendType}
        assert actual == expected


# === ContextPackage 和 TokenUsage 测试（~6 tests）===


class TestTokenUsage:
    """TokenUsage 模型测试。"""

    def test_create_token_usage(self) -> None:
        """测试创建 TokenUsage。"""
        usage = TokenUsage(
            total_tokens=5000,
            by_role={"system": 400, "user": 1500, "assistant": 1000},
            by_type={"system": 400, "user": 1500, "rag": 2000},
            segment_count=10,
        )
        assert usage.total_tokens == 5000
        assert usage.by_role["system"] == 400
        assert usage.segment_count == 10

    def test_token_usage_minimal(self) -> None:
        """测试 TokenUsage 最小创建（只需 total_tokens）。"""
        usage = TokenUsage(total_tokens=1000)
        assert usage.total_tokens == 1000
        assert usage.by_role == {}
        assert usage.by_type == {}
        assert usage.segment_count == 0


class TestContextPackage:
    """ContextPackage 模型测试。"""

    @pytest.fixture
    def _budget_allocation(self) -> BudgetAllocation:
        """用于 ContextPackage 测试的 BudgetAllocation fixture。"""
        return BudgetAllocation(
            total_budget=8192,
            content_budget=7168,
            total_used=5000,
            rigid_used=500,
            elastic_used={"rag": 2000, "user": 1500},
            output_reserved=1024,
        )

    def test_create_context_package(
        self,
        sample_segment: Segment,
        _budget_allocation: BudgetAllocation,
    ) -> None:
        """测试创建 ContextPackage。"""
        package = ContextPackage(
            segments=[sample_segment],
            audit_log=[],
            budget_allocation=_budget_allocation,
            model="gpt-4o",
            policy_version="1.0.0",
        )
        assert len(package.segments) == 1
        assert package.model == "gpt-4o"
        assert package.policy_version == "1.0.0"

    def test_context_package_to_messages(
        self,
        system_segment: Segment,
        conversation_segments: list[Segment],
        _budget_allocation: BudgetAllocation,
    ) -> None:
        """测试 to_messages() 转换。"""
        package = ContextPackage(
            segments=[system_segment] + conversation_segments,
            audit_log=[],
            budget_allocation=_budget_allocation,
            model="gpt-4o",
        )
        messages = package.to_messages()

        assert isinstance(messages, list)
        assert len(messages) == len(package.segments)
        assert all("role" in m and "content" in m for m in messages)

    def test_context_package_summary(
        self,
        system_segment: Segment,
        conversation_segments: list[Segment],
        _budget_allocation: BudgetAllocation,
    ) -> None:
        """测试 summary() 方法。"""
        package = ContextPackage(
            segments=[system_segment] + conversation_segments,
            audit_log=[],
            budget_allocation=_budget_allocation,
            model="gpt-4o",
            policy_version="1.0.0",
            assembly_duration_ms=45.2,
        )
        summary = package.summary()
        assert isinstance(summary, str)
        # summary() 包含 "Segment" 和 "Token" 关键词
        assert "Segment" in summary
        assert "Token" in summary

    def test_context_package_token_usage_auto_calculated(
        self,
        _budget_allocation: BudgetAllocation,
    ) -> None:
        """测试 token_usage 自动计算。"""
        segments = [
            Segment(
                type=SegmentType.SYSTEM,
                content="test",
                role="system",
            ).with_token_count(100),
            Segment(
                type=SegmentType.USER,
                content="test",
                role="user",
            ).with_token_count(200),
        ]
        package = ContextPackage(
            segments=segments,
            audit_log=[],
            budget_allocation=_budget_allocation,
            model="gpt-4o",
        )
        assert package.token_usage.total_tokens == 300
        assert package.token_usage.by_role["system"] == 100
        assert package.token_usage.by_role["user"] == 200

    def test_context_package_to_snapshot(
        self,
        system_segment: Segment,
        conversation_segments: list[Segment],
        _budget_allocation: BudgetAllocation,
    ) -> None:
        """测试 to_snapshot() 序列化。"""
        package = ContextPackage(
            segments=[system_segment] + conversation_segments,
            audit_log=[],
            budget_allocation=_budget_allocation,
            model="gpt-4o",
            policy_version="1.0.0",
        )
        snapshot = package.to_snapshot()
        assert isinstance(snapshot, dict)
        assert "segments" in snapshot
        assert "budget" in snapshot
        assert "model" in snapshot

    def test_context_package_dropped_segments(self) -> None:
        """测试 dropped_segments 属性。"""
        drop_entry = AuditEntry(
            segment_id="seg_abc",
            decision=DecisionType.DROP,
            reason_code=ReasonCode.BUDGET_EXCEEDED,
            reason_detail="预算不足",
            pipeline_stage="allocate",
        )
        keep_entry = AuditEntry(
            segment_id="seg_def",
            decision=DecisionType.KEEP,
            reason_code=ReasonCode.RIGID_GUARANTEED,
            pipeline_stage="allocate",
        )
        package = ContextPackage(
            segments=[],
            audit_log=[drop_entry, keep_entry],
            model="gpt-4o",
        )
        assert len(package.dropped_segments) == 1
        assert package.dropped_segments[0].segment_id == "seg_abc"
        assert package.has_drops is True

    def test_context_package_to_text(self) -> None:
        """测试 to_text() 方法。"""
        seg1 = Segment(type=SegmentType.USER, content="Hello", role="user")
        seg2 = Segment(type=SegmentType.ASSISTANT, content="World", role="assistant")
        package = ContextPackage(segments=[seg1, seg2], model="gpt-4o")
        text = package.to_text()
        assert "Hello" in text
        assert "World" in text


# === ModelConfig 和 Routing 测试（~8 tests）===


class TestModelConfig:
    """ModelConfig 模型测试。"""

    def test_create_model_config(self) -> None:
        """测试创建模型配置。"""
        config = ModelConfig(
            model_id="gpt-4o",
            provider="openai",
            max_context_tokens=128000,
            max_output_tokens=16384,
        )
        assert config.model_id == "gpt-4o"
        assert config.provider == "openai"
        assert config.max_context_tokens == 128000

    def test_model_config_with_cost(self) -> None:
        """测试带成本信息的模型配置。"""
        config = ModelConfig(
            model_id="gpt-4o",
            provider="openai",
            max_context_tokens=128000,
            cost_per_million_input=2.5,
            cost_per_million_output=10.0,
        )
        assert config.cost_per_million_input == 2.5
        assert config.cost_per_million_output == 10.0

    def test_model_config_capabilities(self) -> None:
        """测试模型能力标志。"""
        config = ModelConfig(
            model_id="gpt-4o",
            provider="openai",
            max_context_tokens=128000,
            supports_thinking=False,
            supports_vision=True,
            supports_tool_use=True,
        )
        assert config.supports_thinking is False
        assert config.supports_vision is True
        assert config.supports_tool_use is True

    def test_model_config_estimate_cost(self) -> None:
        """测试成本估算方法。"""
        config = ModelConfig(
            model_id="gpt-4o",
            provider="openai",
            max_context_tokens=128000,
            cost_per_million_input=2.5,
            cost_per_million_output=10.0,
        )
        cost = config.estimate_cost(input_tokens=1_000_000, output_tokens=100_000)
        # input: 2.5, output: 1.0
        assert cost == pytest.approx(3.5)


class TestRoutingDecision:
    """RoutingDecision 模型测试。"""

    def _make_model_config(self, model_id: str = "gpt-4o-mini") -> ModelConfig:
        """辅助方法：创建 ModelConfig。"""
        return ModelConfig(
            model_id=model_id,
            provider="openai",
            max_context_tokens=128000,
        )

    def test_create_routing_decision(self) -> None:
        """测试创建路由决策。"""
        model = self._make_model_config("gpt-4o-mini")
        decision = RoutingDecision(
            selected_model=model,
            complexity=ComplexityLevel.SIMPLE,
            estimated_cost=0.05,
            confidence=0.92,
            reasoning="简单任务使用小模型",
        )
        assert decision.selected_model.model_id == "gpt-4o-mini"
        assert decision.complexity == ComplexityLevel.SIMPLE
        assert decision.estimated_cost == 0.05
        assert decision.confidence == 0.92

    def test_routing_decision_defaults(self) -> None:
        """测试路由决策的默认值。"""
        model = self._make_model_config("gpt-4o")
        decision = RoutingDecision(
            selected_model=model,
            complexity=ComplexityLevel.COMPLEX,
        )
        assert decision.matched_rule == "default"
        assert decision.is_fallback is False
        assert decision.confidence == 1.0
        assert decision.reasoning == ""
        assert decision.estimated_cost == 0.0


class TestRoutingRule:
    """RoutingRule 模型测试。"""

    def test_create_routing_rule(self) -> None:
        """测试创建路由规则。"""
        rule = RoutingRule(
            name="simple_to_mini",
            condition_type="complexity",
            condition_value="simple",
            target_model="gpt-4o-mini",
            priority=10,
        )
        assert rule.name == "simple_to_mini"
        assert rule.condition_type == "complexity"
        assert rule.condition_value == "simple"
        assert rule.target_model == "gpt-4o-mini"
        assert rule.priority == 10

    def test_routing_rule_with_fallback(self) -> None:
        """测试带降级模型的路由规则。"""
        rule = RoutingRule(
            name="code_to_sonnet",
            condition_type="keyword",
            condition_value="代码",
            target_model="claude-sonnet-4-5",
            priority=5,
            fallback_model="gpt-4o",
        )
        assert rule.fallback_model == "gpt-4o"


class TestComplexityLevel:
    """ComplexityLevel 枚举测试。"""

    def test_complexity_levels(self) -> None:
        """测试复杂度级别枚举。"""
        expected = {"SIMPLE", "MODERATE", "COMPLEX", "EXPERT"}
        actual = {c.name for c in ComplexityLevel}
        assert actual == expected


# === AuditEntry 测试（~5 tests）===


class TestAuditEntry:
    """AuditEntry 模型测试。"""

    def test_create_audit_entry(self) -> None:
        """测试创建审计条目。"""
        entry = AuditEntry(
            segment_id="seg_abc123",
            pipeline_stage="allocate",
            decision=DecisionType.KEEP,
            reason_code=ReasonCode.RIGID_GUARANTEED,
            reason_detail="刚性支出，全额保障",
        )
        assert entry.segment_id == "seg_abc123"
        assert entry.pipeline_stage == "allocate"
        assert entry.decision == DecisionType.KEEP

    def test_audit_entry_with_token_impact(self) -> None:
        """测试带 Token 影响量的审计条目。"""
        entry = AuditEntry(
            segment_id="seg_123",
            pipeline_stage="compress",
            decision=DecisionType.COMPRESS,
            reason_code=ReasonCode.COMPRESS_SUMMARY_REPLACED,
            token_impact=-700,
        )
        assert entry.token_impact == -700

    def test_audit_entry_timestamp_auto_filled(self) -> None:
        """测试时间戳自动填充。"""
        before = datetime.now(timezone.utc)
        entry = AuditEntry(
            segment_id="seg_123",
            pipeline_stage="test",
            decision=DecisionType.KEEP,
            reason_code=ReasonCode.RIGID_GUARANTEED,
        )
        after = datetime.now(timezone.utc)

        assert before <= entry.timestamp <= after

    def test_audit_entry_is_destructive(self) -> None:
        """测试 is_destructive 属性。"""
        keep = AuditEntry(
            segment_id="seg_1",
            decision=DecisionType.KEEP,
            reason_code=ReasonCode.RIGID_GUARANTEED,
        )
        assert keep.is_destructive is False

        drop = AuditEntry(
            segment_id="seg_2",
            decision=DecisionType.DROP,
            reason_code=ReasonCode.BUDGET_EXCEEDED,
        )
        assert drop.is_destructive is True

        truncate = AuditEntry(
            segment_id="seg_3",
            decision=DecisionType.TRUNCATE,
            reason_code=ReasonCode.BUDGET_QUOTA_LIMIT,
        )
        assert truncate.is_destructive is True

    def test_audit_entry_summary(self) -> None:
        """测试 summary 属性。"""
        entry = AuditEntry(
            segment_id="seg_abc",
            pipeline_stage="allocate",
            decision=DecisionType.DROP,
            reason_code=ReasonCode.BUDGET_EXCEEDED,
            token_impact=-3200,
        )
        summary = entry.summary
        assert "allocate" in summary
        assert "drop" in summary
        assert "seg_abc" in summary
        assert "budget_exceeded" in summary
        assert "-3200" in summary


class TestDecisionType:
    """DecisionType 枚举测试。"""

    def test_decision_types(self) -> None:
        """测试决策类型枚举。"""
        expected = {
            "KEEP",
            "DROP",
            "TRUNCATE",
            "COMPRESS",
            "REORDER",
            "SANITIZE",
            "OVERRIDE",
            "MERGE",
            "CACHE_HIT",
        }
        actual = {d.name for d in DecisionType}
        assert actual == expected


class TestReasonCode:
    """ReasonCode 枚举测试。"""

    def test_reason_codes_exist(self) -> None:
        """测试原因码枚举存在且有值。"""
        # 至少应该有这些常用的原因码
        expected = {
            "BUDGET_EXCEEDED",
            "BUDGET_LOW_PRIORITY",
            "BUDGET_QUOTA_LIMIT",
            "RIGID_GUARANTEED",
            "ELASTIC_ALLOCATED",
            "SANITIZE_PII_DETECTED",
            "SANITIZE_INJECTION_DETECTED",
            "SELECT_EXPIRED",
            "COMPRESS_SUMMARY_REPLACED",
        }
        actual = {r.name for r in ReasonCode}
        assert expected.issubset(actual)
