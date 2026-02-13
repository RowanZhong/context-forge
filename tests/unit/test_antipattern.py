"""
反模式检测器单元测试。

→ 6.7 反模式检测与诊断

测试所有 10 个检测规则和 AntiPatternDetector 编排器。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from context_forge.antipattern.base import AntiPatternSeverity, DetectionContext
from context_forge.antipattern.detector import AntiPatternDetector
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
from context_forge.models.audit import AuditEntry, DecisionType, ReasonCode
from context_forge.models.budget import BudgetAllocation, BudgetPolicy
from context_forge.models.control import ControlFlags
from context_forge.models.metadata import SegmentMetadata
from context_forge.models.provenance import Provenance, SourceType
from context_forge.models.segment import Priority, Segment, SegmentType


# ============================================================
# 辅助函数
# ============================================================


def make_segment(
    id: str = "seg_1",
    type: SegmentType = SegmentType.USER,
    content: str = "Test content",
    role: str = "user",
    priority: Priority = Priority.MEDIUM,
    token_count: int | None = 10,
    namespace: str | None = None,
    created_at: datetime | None = None,
    source_id: str | None = None,
) -> Segment:
    """创建测试用 Segment。"""
    control = None
    if namespace is not None:
        control = ControlFlags(namespace=namespace)

    metadata = None
    if created_at is not None:
        metadata = SegmentMetadata(injected_at=created_at)

    provenance = None
    if source_id is not None:
        provenance = Provenance(source_type=SourceType.USER_INPUT, source_id=source_id)

    return Segment(
        id=id,
        type=type,
        content=content,
        role=role,
        priority=priority,
        token_count=token_count,
        control=control,
        metadata=metadata,
        provenance=provenance,
    )


# ============================================================
# CRITICAL 级别规则测试
# ============================================================


def test_missing_token_count_rule():
    """测试 MissingTokenCountRule。"""
    rule = MissingTokenCountRule()

    # 有问题的情况：部分 Segment 缺失 token_count
    segments = [
        make_segment("seg_1", token_count=100),
        make_segment("seg_2", token_count=None),  # 缺失
        make_segment("seg_3", token_count=0),     # 缺失（0 视为缺失）
    ]

    context = DetectionContext(segments=segments, audit_log=[], config={})
    results = rule.detect(context)

    assert len(results) == 1
    result = results[0]
    assert result.severity == AntiPatternSeverity.CRITICAL
    assert result.rule_name == "MissingTokenCountRule"
    assert len(result.segment_ids) == 2  # seg_2 和 seg_3
    assert "seg_2" in result.segment_ids
    assert "seg_3" in result.segment_ids

    # 正常情况：所有 Segment 都有 token_count
    normal_segments = [
        make_segment("seg_1", token_count=100),
        make_segment("seg_2", token_count=50),
    ]
    normal_context = DetectionContext(segments=normal_segments, audit_log=[], config={})
    normal_results = rule.detect(normal_context)
    assert len(normal_results) == 0


def test_namespace_leakage_rule():
    """测试 NamespaceLeakageRule。"""
    rule = NamespaceLeakageRule()

    # 有问题的情况：部分 Segment 属于其他命名空间
    segments = [
        make_segment("seg_1", namespace="agent_a"),  # 正确的命名空间
        make_segment("seg_2", namespace="agent_b"),  # 泄漏
        make_segment("seg_3", namespace="agent_a"),  # 正确的命名空间
        make_segment("seg_4", namespace="global"),   # global 命名空间允许
    ]

    context = DetectionContext(
        segments=segments,
        audit_log=[],
        config={"target_namespace": "agent_a"},
    )
    results = rule.detect(context)

    assert len(results) == 1
    result = results[0]
    assert result.severity == AntiPatternSeverity.CRITICAL
    assert result.rule_name == "NamespaceLeakageRule"
    assert len(result.segment_ids) == 1  # 只有 seg_2
    assert "seg_2" in result.segment_ids

    # 正常情况：所有 Segment 都属于正确的命名空间
    normal_segments = [
        make_segment("seg_1", namespace="agent_a"),
        make_segment("seg_2", namespace="agent_a"),
    ]
    normal_context = DetectionContext(
        segments=normal_segments,
        audit_log=[],
        config={"target_namespace": "agent_a"},
    )
    normal_results = rule.detect(normal_context)
    assert len(normal_results) == 0


def test_circular_dependency_rule():
    """测试 CircularDependencyRule。"""
    rule = CircularDependencyRule()

    # 注意：当前 ControlFlags 没有 depends_on 字段，此规则不会生效
    # 这是一个占位测试，等待未来扩展
    segments = [
        make_segment("seg_1"),
        make_segment("seg_2"),
    ]

    context = DetectionContext(segments=segments, audit_log=[], config={})
    results = rule.detect(context)

    # 当前应该不检测到循环依赖（因为没有依赖关系）
    assert len(results) == 0


# ============================================================
# WARNING 级别规则测试
# ============================================================


def test_overuse_critical_rule():
    """测试 OveruseCriticalRule。"""
    rule = OveruseCriticalRule()

    # 有问题的情况：超过 50% 的 Segment 为 CRITICAL
    segments = [
        make_segment("seg_1", priority=Priority.CRITICAL),
        make_segment("seg_2", priority=Priority.CRITICAL),
        make_segment("seg_3", priority=Priority.CRITICAL),
        make_segment("seg_4", priority=Priority.MEDIUM),
    ]

    context = DetectionContext(segments=segments, audit_log=[], config={})
    results = rule.detect(context)

    assert len(results) == 1
    result = results[0]
    assert result.severity == AntiPatternSeverity.WARNING
    assert result.rule_name == "OveruseCriticalRule"
    assert result.metadata["critical_count"] == 3
    assert result.metadata["total_count"] == 4

    # 正常情况：CRITICAL 占比 <= 50%
    normal_segments = [
        make_segment("seg_1", priority=Priority.CRITICAL),
        make_segment("seg_2", priority=Priority.MEDIUM),
        make_segment("seg_3", priority=Priority.MEDIUM),
    ]
    normal_context = DetectionContext(segments=normal_segments, audit_log=[], config={})
    normal_results = rule.detect(normal_context)
    assert len(normal_results) == 0


def test_rigid_budget_too_large_rule():
    """测试 RigidBudgetTooLargeRule。"""
    rule = RigidBudgetTooLargeRule()

    # 有问题的情况：刚性预算占比 > 70%
    budget_allocation = BudgetAllocation(
        total_budget=1200,
        content_budget=1000,
        rigid_used=800,  # 80% 刚性
        output_reserved=100,
        thinking_reserved=100,
    )

    budget_policy = BudgetPolicy(
        rigid_ratio=0.7,
        elastic_ratio=0.2,
        reserved_ratio=0.1,
    )

    context = DetectionContext(
        segments=[],
        audit_log=[],
        config={},
        budget_allocation=budget_allocation,
        budget_policy=budget_policy,
    )
    results = rule.detect(context)

    assert len(results) == 1
    result = results[0]
    assert result.severity == AntiPatternSeverity.WARNING
    assert result.rule_name == "RigidBudgetTooLargeRule"
    assert result.metadata["rigid_used"] == 800
    assert result.metadata["content_budget"] == 1000

    # 正常情况：刚性预算占比 <= 70%
    normal_allocation = BudgetAllocation(
        total_budget=1200,
        content_budget=1000,
        rigid_used=600,  # 60% 刚性
        output_reserved=100,
        thinking_reserved=100,
    )

    normal_context = DetectionContext(
        segments=[],
        audit_log=[],
        config={},
        budget_allocation=normal_allocation,
        budget_policy=budget_policy,
    )
    normal_results = rule.detect(normal_context)
    assert len(normal_results) == 0


def test_expired_data_rule():
    """测试 ExpiredDataRule。"""
    rule = ExpiredDataRule()

    now = datetime.now(timezone.utc)
    old_date = now - timedelta(days=35)  # 超过 30 天

    # 有问题的情况：存在过期数据
    segments = [
        make_segment("seg_1", created_at=now),  # 新数据
        make_segment("seg_2", created_at=old_date),  # 过期数据
    ]

    context = DetectionContext(segments=segments, audit_log=[], config={})
    results = rule.detect(context)

    assert len(results) == 1
    result = results[0]
    assert result.severity == AntiPatternSeverity.WARNING
    assert result.rule_name == "ExpiredDataRule"
    assert "seg_2" in result.segment_ids

    # 正常情况：所有数据都是新鲜的
    normal_segments = [
        make_segment("seg_1", created_at=now),
        make_segment("seg_2", created_at=now - timedelta(days=5)),
    ]
    normal_context = DetectionContext(segments=normal_segments, audit_log=[], config={})
    normal_results = rule.detect(normal_context)
    assert len(normal_results) == 0


def test_over_compression_rule():
    """测试 OverCompressionRule。"""
    rule = OverCompressionRule()

    # 有问题的情况：压缩率 < 10%
    audit_log = [
        AuditEntry(
            segment_id="seg_1",
            decision=DecisionType.COMPRESS,
            reason_code=ReasonCode.COMPRESS_WINDOW_SATURATION,
            reason_detail="压缩以节省空间",
            metadata={
                "original_tokens": 1000,
                "compressed_tokens": 50,  # 5% 压缩率
            },
        ),
    ]

    context = DetectionContext(segments=[], audit_log=audit_log, config={})
    results = rule.detect(context)

    assert len(results) == 1
    result = results[0]
    assert result.severity == AntiPatternSeverity.WARNING
    assert result.rule_name == "OverCompressionRule"

    # 正常情况：压缩率 >= 10%
    normal_audit_log = [
        AuditEntry(
            segment_id="seg_1",
            decision=DecisionType.COMPRESS,
            reason_code=ReasonCode.COMPRESS_WINDOW_SATURATION,
            reason_detail="压缩以节省空间",
            metadata={
                "original_tokens": 1000,
                "compressed_tokens": 150,  # 15% 压缩率
            },
        ),
    ]

    normal_context = DetectionContext(segments=[], audit_log=normal_audit_log, config={})
    normal_results = rule.detect(normal_context)
    assert len(normal_results) == 0


# ============================================================
# INFO 级别规则测试
# ============================================================


def test_ineffective_routing_rule():
    """测试 IneffectiveRoutingRule。"""
    rule = IneffectiveRoutingRule()

    # 有问题的情况：路由前后窗口差异 < 10%
    context = DetectionContext(
        segments=[],
        audit_log=[],
        config={
            "routing_enabled": True,
            "routing_decision": {"model": "gpt-4o-mini"},
            "original_window_size": 128000,
            "selected_window_size": 130000,  # 差异仅 ~1.5%
        },
    )
    results = rule.detect(context)

    assert len(results) == 1
    result = results[0]
    assert result.severity == AntiPatternSeverity.INFO
    assert result.rule_name == "IneffectiveRoutingRule"

    # 正常情况：路由带来明显差异
    normal_context = DetectionContext(
        segments=[],
        audit_log=[],
        config={
            "routing_enabled": True,
            "routing_decision": {"model": "gpt-4o-mini"},
            "original_window_size": 128000,
            "selected_window_size": 8000,  # 差异 ~94%
        },
    )
    normal_results = rule.detect(normal_context)
    assert len(normal_results) == 0


def test_cache_key_collision_rule():
    """测试 CacheKeyCollisionRule。"""
    rule = CacheKeyCollisionRule()

    # 有问题的情况：多个 Segment 共享同一个 source_id
    segments = [
        make_segment("seg_1", source_id="doc_123"),
        make_segment("seg_2", source_id="doc_123"),  # 冲突
        make_segment("seg_3", source_id="doc_456"),
    ]

    context = DetectionContext(segments=segments, audit_log=[], config={})
    results = rule.detect(context)

    assert len(results) == 1
    result = results[0]
    assert result.severity == AntiPatternSeverity.INFO
    assert result.rule_name == "CacheKeyCollisionRule"
    assert result.metadata["collision_count"] == 1  # 一个冲突的 source_id
    assert result.metadata["total_segments_affected"] == 2  # 两个 Segment

    # 正常情况：所有 source_id 唯一
    normal_segments = [
        make_segment("seg_1", source_id="doc_123"),
        make_segment("seg_2", source_id="doc_456"),
        make_segment("seg_3", source_id="doc_789"),
    ]
    normal_context = DetectionContext(segments=normal_segments, audit_log=[], config={})
    normal_results = rule.detect(normal_context)
    assert len(normal_results) == 0


def test_unused_sanitizer_rule():
    """测试 UnusedSanitizerRule。"""
    rule = UnusedSanitizerRule()

    # 有问题的情况：存在需要清洗的 Segment，但无 SANITIZE 决策
    segments = [
        make_segment("seg_1", type=SegmentType.USER),
        make_segment("seg_2", type=SegmentType.RAG),
    ]

    context = DetectionContext(segments=segments, audit_log=[], config={})
    results = rule.detect(context)

    assert len(results) == 1
    result = results[0]
    assert result.severity == AntiPatternSeverity.INFO
    assert result.rule_name == "UnusedSanitizerRule"

    # 正常情况：有 SANITIZE 决策
    normal_audit_log = [
        AuditEntry(
            segment_id="seg_1",
            decision=DecisionType.SANITIZE,
            reason_code=ReasonCode.SANITIZE_HTML_STRIPPED,
            reason_detail="移除 HTML 标签",
        ),
    ]

    normal_context = DetectionContext(
        segments=segments,
        audit_log=normal_audit_log,
        config={},
    )
    normal_results = rule.detect(normal_context)
    assert len(normal_results) == 0


# ============================================================
# AntiPatternDetector 编排测试
# ============================================================


def test_detector_detect_all():
    """测试 AntiPatternDetector.detect() 编排多个规则。"""
    from context_forge.antipattern import create_default_detector

    detector = create_default_detector()

    # 构建包含多个问题的上下文
    segments = [
        make_segment("seg_1", priority=Priority.CRITICAL, token_count=None),  # 缺失 token_count
        make_segment("seg_2", priority=Priority.CRITICAL, token_count=100),
        make_segment("seg_3", priority=Priority.CRITICAL, token_count=100),
        make_segment("seg_4", priority=Priority.MEDIUM, token_count=100),
    ]

    context = DetectionContext(
        segments=segments,
        audit_log=[],
        config={},
    )

    results = detector.detect(context)

    # 应该检测到至少 2 个问题
    assert len(results) >= 2

    # 检查是否包含预期的问题
    rule_names = {result.rule_name for result in results}
    assert "MissingTokenCountRule" in rule_names  # 缺失 token_count
    assert "OveruseCriticalRule" in rule_names  # CRITICAL 滥用


def test_detector_format_report():
    """测试 AntiPatternDetector.format_report() 格式化报告。"""
    from context_forge.antipattern import create_default_detector

    detector = create_default_detector()

    segments = [
        make_segment("seg_1", token_count=None),
    ]

    context = DetectionContext(
        segments=segments,
        audit_log=[],
        config={},
    )

    results = detector.detect(context)

    # 格式化为 Rich 格式
    rich_report = detector.format_report(results, format="rich")
    assert rich_report is not None
    # Rich 格式返回 Panel 对象，这里只验证不会抛异常

    # 格式化为纯文本
    text_report = detector.format_report(results, format="text")
    assert isinstance(text_report, str)
    assert "MissingTokenCountRule" in text_report

    # 格式化为 JSON
    json_report = detector.format_report(results, format="json")
    assert isinstance(json_report, str)
    assert "MissingTokenCountRule" in json_report


def test_detector_filter_by_severity():
    """测试按严重性过滤结果。"""
    from context_forge.antipattern import create_default_detector

    detector = create_default_detector()

    segments = [
        make_segment("seg_1", priority=Priority.CRITICAL, token_count=None),  # CRITICAL
        make_segment("seg_2", priority=Priority.CRITICAL, token_count=100),
        make_segment("seg_3", priority=Priority.CRITICAL, token_count=100),
    ]

    context = DetectionContext(
        segments=segments,
        audit_log=[],
        config={},
    )

    all_results = detector.detect(context)

    # 过滤 CRITICAL 级别
    critical_results = [r for r in all_results if r.severity == AntiPatternSeverity.CRITICAL]
    assert len(critical_results) >= 1

    # 过滤 WARNING 级别
    warning_results = [r for r in all_results if r.severity == AntiPatternSeverity.WARNING]
    assert len(warning_results) >= 1  # OveruseCriticalRule
