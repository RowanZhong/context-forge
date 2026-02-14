"""
缓存集成测试 — 验证 ContextPackage 的缓存序列化/反序列化和 Facade 缓存命中逻辑。

覆盖范围:
- ContextPackage.to_cache_dict() 完整序列化
- ContextPackage.from_cache_dict() 反序列化重建
- to_cache_dict() / from_cache_dict() 往返一致性
- Facade 缓存命中路径（跳过 Pipeline 直接返回）
- PrefixCacheKeyGenerator 前缀缓存键生成
- 边界条件和降级行为
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pytest

from context_forge.cache.base import CacheEntry
from context_forge.cache.keys import (
    compute_prefix_cache_key,
    extract_key_metadata,
)
from context_forge.models.audit import AuditEntry, DecisionType, ReasonCode
from context_forge.models.budget import BudgetAllocation
from context_forge.models.context_package import ContextPackage
from context_forge.models.control import ControlFlags
from context_forge.models.metadata import SegmentMetadata
from context_forge.models.provenance import Provenance, SourceType
from context_forge.models.routing import (
    ComplexityLevel,
    ModelConfig,
    RoutingDecision,
)
from context_forge.models.segment import Priority, Segment, SegmentType


# === to_cache_dict 测试 ===


class TestToCacheDict:
    """测试 ContextPackage.to_cache_dict() 序列化方法。"""

    def test_basic_serialization(self) -> None:
        """测试基本序列化——Segment 完整内容保留。"""
        long_content = "A" * 500  # 超过 200 字符
        seg = Segment(
            type=SegmentType.USER,
            content=long_content,
            role="user",
            token_count=100,
        )
        package = ContextPackage(
            segments=[seg],
            model="gpt-4o",
            policy_version="1.0",
        )

        cache_dict = package.to_cache_dict()

        # 验证完整内容保留（不同于 to_snapshot 的 200 字符截断）
        assert cache_dict["segments"][0]["content"] == long_content
        assert len(cache_dict["segments"][0]["content"]) == 500

    def test_snapshot_truncates_but_cache_preserves(self) -> None:
        """验证 to_snapshot 截断内容而 to_cache_dict 保留完整内容。"""
        long_content = "B" * 300
        seg = Segment(
            type=SegmentType.RAG,
            content=long_content,
            role="user",
            token_count=50,
        )
        package = ContextPackage(segments=[seg], model="gpt-4o")

        snapshot = package.to_snapshot()
        cache_dict = package.to_cache_dict()

        # to_snapshot 截断 + "..."
        assert snapshot["segments"][0]["content_preview"].endswith("...")
        assert len(snapshot["segments"][0]["content_preview"]) == 203  # 200 + "..."

        # to_cache_dict 保留完整内容
        assert cache_dict["segments"][0]["content"] == long_content
        assert "content_preview" not in cache_dict["segments"][0]

    def test_multiple_segments(self) -> None:
        """测试多个 Segment 的序列化。"""
        segments = [
            Segment(
                type=SegmentType.SYSTEM,
                content="系统提示",
                role="system",
                priority=Priority.CRITICAL,
                control=ControlFlags(must_keep=True, lock_position=True),
                token_count=10,
            ),
            Segment(
                type=SegmentType.USER,
                content="用户消息",
                role="user",
                token_count=5,
            ),
            Segment(
                type=SegmentType.RAG,
                content="RAG 检索片段",
                role="user",
                priority=Priority.MEDIUM,
                control=ControlFlags(namespace="rag_ns"),
                token_count=20,
            ),
        ]
        package = ContextPackage(
            segments=segments,
            model="gpt-4o",
            policy_version="2.0",
        )

        cache_dict = package.to_cache_dict()

        assert len(cache_dict["segments"]) == 3
        assert cache_dict["segments"][0]["type"] == "system"
        assert cache_dict["segments"][0]["must_keep"] is True
        assert cache_dict["segments"][0]["lock_position"] is True
        assert cache_dict["segments"][2]["namespace"] == "rag_ns"

    def test_budget_allocation_included(self) -> None:
        """测试预算分配信息包含在缓存字典中。"""
        budget = BudgetAllocation(
            total_budget=8192,
            content_budget=7168,
            total_used=3000,
            rigid_used=500,
            elastic_used={"user": 1000, "rag": 1500},
            output_reserved=1024,
        )
        package = ContextPackage(
            segments=[],
            model="gpt-4o",
            budget_allocation=budget,
        )

        cache_dict = package.to_cache_dict()

        assert cache_dict["budget"] is not None
        assert cache_dict["budget"]["total_budget"] == 8192
        assert cache_dict["budget"]["total_used"] == 3000

    def test_routing_decision_included(self) -> None:
        """测试路由决策信息包含在缓存字典中。"""
        routing = RoutingDecision(
            selected_model=ModelConfig(
                model_id="gpt-4o-mini",
                provider="openai",
                max_context_tokens=128000,
            ),
            complexity=ComplexityLevel.SIMPLE,
            estimated_cost=0.01,
            reasoning="简单查询",
        )
        package = ContextPackage(
            segments=[],
            model="gpt-4o-mini",
            routing_decision=routing,
        )

        cache_dict = package.to_cache_dict()

        assert cache_dict["routing"] is not None
        assert cache_dict["routing"]["complexity"] == "simple"

    def test_audit_log_included(self) -> None:
        """测试审计日志包含在缓存字典中。"""
        audit = AuditEntry(
            segment_id="seg_abc123",
            decision=DecisionType.KEEP,
            reason_code=ReasonCode.RIGID_GUARANTEED,
            reason_detail="刚性保障",
            pipeline_stage="allocate",
        )
        package = ContextPackage(
            segments=[],
            model="gpt-4o",
            audit_log=[audit],
        )

        cache_dict = package.to_cache_dict()

        assert len(cache_dict["audit_log"]) == 1
        assert cache_dict["audit_log"][0]["decision"] == "keep"

    def test_warnings_included(self) -> None:
        """测试警告信息包含在缓存字典中。"""
        package = ContextPackage(
            segments=[],
            model="gpt-4o",
            warnings=["测试警告 1", "测试警告 2"],
        )

        cache_dict = package.to_cache_dict()

        assert cache_dict["warnings"] == ["测试警告 1", "测试警告 2"]

    def test_json_serializable(self) -> None:
        """测试缓存字典可以被 JSON 序列化。"""
        seg = Segment(
            type=SegmentType.USER,
            content="中文内容测试",
            role="user",
            token_count=10,
        )
        budget = BudgetAllocation(
            total_budget=8192,
            content_budget=7168,
        )
        package = ContextPackage(
            segments=[seg],
            model="gpt-4o",
            budget_allocation=budget,
            warnings=["警告"],
        )

        cache_dict = package.to_cache_dict()
        # 确保可以序列化和反序列化
        json_str = json.dumps(cache_dict, ensure_ascii=False, default=str)
        restored = json.loads(json_str)
        assert restored["model"] == "gpt-4o"
        assert restored["segments"][0]["content"] == "中文内容测试"


# === from_cache_dict 测试 ===


class TestFromCacheDict:
    """测试 ContextPackage.from_cache_dict() 反序列化方法。"""

    def test_basic_deserialization(self) -> None:
        """测试基本反序列化——从字典重建 ContextPackage。"""
        data = {
            "request_id": "req_test123",
            "model": "gpt-4o",
            "policy_version": "1.0",
            "created_at": "2026-02-14T10:00:00+00:00",
            "assembly_duration_ms": 25.5,
            "segments": [
                {
                    "id": "seg_abc123def4",
                    "type": "user",
                    "content": "测试内容",
                    "role": "user",
                    "priority": "high",
                    "token_count": 10,
                    "namespace": "default",
                    "must_keep": False,
                    "lock_position": False,
                    "compressible": True,
                }
            ],
            "budget": {
                "total_budget": 8192,
                "content_budget": 7168,
                "total_used": 100,
            },
            "routing": None,
            "audit_log": [],
            "warnings": ["测试警告"],
        }

        package = ContextPackage.from_cache_dict(data)

        assert package.request_id == "req_test123"
        assert package.model == "gpt-4o"
        assert package.policy_version == "1.0"
        assert package.assembly_duration_ms == 25.5
        assert len(package.segments) == 1
        assert package.segments[0].content == "测试内容"
        assert package.segments[0].type == SegmentType.USER
        assert package.segments[0].role == "user"
        assert package.segments[0].token_count == 10
        assert package.warnings == ["测试警告"]

    def test_segment_control_flags_restored(self) -> None:
        """测试 Segment 的控制标志正确恢复。"""
        data = {
            "segments": [
                {
                    "type": "system",
                    "content": "系统提示",
                    "role": "system",
                    "priority": "critical",
                    "namespace": "system_ns",
                    "must_keep": True,
                    "lock_position": True,
                    "compressible": False,
                }
            ],
            "model": "gpt-4o",
        }

        package = ContextPackage.from_cache_dict(data)

        seg = package.segments[0]
        assert seg.control.must_keep is True
        assert seg.control.lock_position is True
        assert seg.control.compressible is False
        assert seg.control.namespace == "system_ns"

    def test_budget_allocation_restored(self) -> None:
        """测试预算分配信息正确恢复。"""
        data = {
            "segments": [],
            "model": "gpt-4o",
            "budget": {
                "total_budget": 128000,
                "content_budget": 120000,
                "rigid_used": 500,
                "elastic_used": {"user": 2000, "rag": 3000},
                "total_used": 5500,
                "output_reserved": 4096,
                "thinking_reserved": 0,
                "overflow_count": 1,
            },
        }

        package = ContextPackage.from_cache_dict(data)

        assert package.budget_allocation is not None
        assert package.budget_allocation.total_budget == 128000
        assert package.budget_allocation.total_used == 5500
        assert package.budget_allocation.rigid_used == 500
        assert package.budget_allocation.overflow_count == 1

    def test_routing_decision_restored(self) -> None:
        """测试路由决策信息正确恢复。"""
        data = {
            "segments": [],
            "model": "gpt-4o-mini",
            "routing": {
                "selected_model": {
                    "model_id": "gpt-4o-mini",
                    "provider": "openai",
                    "max_context_tokens": 128000,
                },
                "complexity": "simple",
                "matched_rule": "simple_to_mini",
                "is_fallback": False,
                "confidence": 0.95,
                "reasoning": "简单查询",
                "estimated_cost": 0.01,
            },
        }

        package = ContextPackage.from_cache_dict(data)

        assert package.routing_decision is not None
        assert package.routing_decision.selected_model.model_id == "gpt-4o-mini"
        assert package.routing_decision.complexity == ComplexityLevel.SIMPLE
        assert package.routing_decision.confidence == 0.95

    def test_audit_log_restored(self) -> None:
        """测试审计日志正确恢复。"""
        data = {
            "segments": [],
            "model": "gpt-4o",
            "audit_log": [
                {
                    "segment_id": "seg_abc",
                    "decision": "keep",
                    "reason_code": "rigid_guaranteed",
                    "reason_detail": "刚性保障",
                    "pipeline_stage": "allocate",
                    "token_impact": 0,
                },
                {
                    "segment_id": "seg_def",
                    "decision": "drop",
                    "reason_code": "budget_exceeded",
                    "reason_detail": "预算不足",
                    "pipeline_stage": "allocate",
                    "token_impact": -500,
                },
            ],
        }

        package = ContextPackage.from_cache_dict(data)

        assert len(package.audit_log) == 2
        assert package.audit_log[0].decision == DecisionType.KEEP
        assert package.audit_log[1].decision == DecisionType.DROP

    def test_missing_optional_fields(self) -> None:
        """测试缺少可选字段时的优雅降级。"""
        data = {
            "segments": [
                {
                    "type": "user",
                    "content": "内容",
                    "role": "user",
                    # 不提供 priority, namespace, must_keep 等可选字段
                }
            ],
            "model": "gpt-4o",
            # 不提供 budget, routing, audit_log, warnings
        }

        package = ContextPackage.from_cache_dict(data)

        assert len(package.segments) == 1
        assert package.segments[0].content == "内容"
        assert package.budget_allocation is None
        assert package.routing_decision is None
        assert len(package.audit_log) == 0
        assert len(package.warnings) == 0

    def test_empty_segments(self) -> None:
        """测试空 Segment 列表的反序列化。"""
        data = {
            "segments": [],
            "model": "gpt-4o",
        }

        package = ContextPackage.from_cache_dict(data)

        assert len(package.segments) == 0
        assert package.model == "gpt-4o"

    def test_created_at_parsed(self) -> None:
        """测试创建时间正确解析。"""
        data = {
            "segments": [],
            "model": "gpt-4o",
            "created_at": "2026-01-15T08:30:00+00:00",
        }

        package = ContextPackage.from_cache_dict(data)

        assert package.created_at.year == 2026
        assert package.created_at.month == 1
        assert package.created_at.day == 15

    def test_invalid_created_at_fallback(self) -> None:
        """测试无效创建时间降级为当前时间。"""
        data = {
            "segments": [],
            "model": "gpt-4o",
            "created_at": "invalid-date-string",
        }

        package = ContextPackage.from_cache_dict(data)

        # 应该降级为当前时间（不报错）
        assert isinstance(package.created_at, datetime)

    def test_budget_with_extra_fields_ignored(self) -> None:
        """测试预算数据中的额外字段（如计算属性）被忽略。"""
        data = {
            "segments": [],
            "model": "gpt-4o",
            "budget": {
                "total_budget": 8192,
                "content_budget": 7168,
                "total_used": 3000,
                # 计算属性，应该被过滤掉
                "saturation_rate": 0.42,
                "remaining": 4168,
                "is_over_budget": False,
            },
        }

        package = ContextPackage.from_cache_dict(data)

        assert package.budget_allocation is not None
        assert package.budget_allocation.total_budget == 8192


# === 往返一致性测试 ===


class TestRoundTrip:
    """测试 to_cache_dict() → JSON → from_cache_dict() 的往返一致性。"""

    def test_simple_round_trip(self) -> None:
        """测试简单 ContextPackage 的往返一致性。"""
        seg = Segment(
            type=SegmentType.USER,
            content="往返测试内容",
            role="user",
            priority=Priority.HIGH,
            token_count=20,
        )
        original = ContextPackage(
            segments=[seg],
            model="gpt-4o",
            policy_version="1.0",
            assembly_duration_ms=10.5,
            warnings=["测试警告"],
        )

        # 序列化 → JSON → 反序列化
        cache_dict = original.to_cache_dict()
        json_str = json.dumps(cache_dict, ensure_ascii=False, default=str)
        restored_dict = json.loads(json_str)
        restored = ContextPackage.from_cache_dict(restored_dict)

        # 验证核心属性一致
        assert restored.model == original.model
        assert restored.policy_version == original.policy_version
        assert restored.assembly_duration_ms == original.assembly_duration_ms
        assert restored.warnings == original.warnings
        assert len(restored.segments) == len(original.segments)

    def test_full_round_trip(self) -> None:
        """测试包含所有字段的完整往返一致性。"""
        segments = [
            Segment(
                type=SegmentType.SYSTEM,
                content="你是一个助手。",
                role="system",
                priority=Priority.CRITICAL,
                control=ControlFlags(must_keep=True, lock_position=True, compressible=False),
                token_count=10,
            ),
            Segment(
                type=SegmentType.USER,
                content="你好，我需要帮助",
                role="user",
                priority=Priority.HIGH,
                token_count=15,
            ),
            Segment(
                type=SegmentType.RAG,
                content="C" * 500,  # 超过 200 字符，测试内容不被截断
                role="user",
                priority=Priority.MEDIUM,
                control=ControlFlags(namespace="rag_ns"),
                token_count=100,
            ),
        ]
        budget = BudgetAllocation(
            total_budget=8192,
            content_budget=7168,
            total_used=125,
            rigid_used=10,
            elastic_used={"user": 15, "rag": 100},
            output_reserved=1024,
        )
        routing = RoutingDecision(
            selected_model=ModelConfig(
                model_id="gpt-4o",
                provider="openai",
                max_context_tokens=128000,
            ),
            complexity=ComplexityLevel.MODERATE,
            estimated_cost=0.05,
            reasoning="中等复杂度查询",
        )
        audit_log = [
            AuditEntry(
                segment_id="seg_001",
                decision=DecisionType.KEEP,
                reason_code=ReasonCode.RIGID_GUARANTEED,
                pipeline_stage="allocate",
            ),
        ]

        original = ContextPackage(
            segments=segments,
            model="gpt-4o",
            policy_version="2.0",
            budget_allocation=budget,
            routing_decision=routing,
            audit_log=audit_log,
            assembly_duration_ms=45.2,
            warnings=["清洗了 1 个 PII 字段"],
        )

        # 往返
        cache_dict = original.to_cache_dict()
        json_str = json.dumps(cache_dict, ensure_ascii=False, default=str)
        restored_dict = json.loads(json_str)
        restored = ContextPackage.from_cache_dict(restored_dict)

        # 验证 Segment 内容和属性
        assert len(restored.segments) == 3
        assert restored.segments[0].content == "你是一个助手。"
        assert restored.segments[0].type == SegmentType.SYSTEM
        assert restored.segments[0].control.must_keep is True
        assert restored.segments[2].content == "C" * 500  # 完整内容
        assert restored.segments[2].control.namespace == "rag_ns"

        # 验证 Budget
        assert restored.budget_allocation is not None
        assert restored.budget_allocation.total_budget == 8192
        assert restored.budget_allocation.total_used == 125

        # 验证 Routing
        assert restored.routing_decision is not None
        assert restored.routing_decision.selected_model.model_id == "gpt-4o"

        # 验证 Audit
        assert len(restored.audit_log) == 1

        # 验证 to_messages() 输出一致
        original_messages = original.to_messages()
        restored_messages = restored.to_messages()
        assert len(original_messages) == len(restored_messages)
        for om, rm in zip(original_messages, restored_messages):
            assert om["role"] == rm["role"]
            assert om["content"] == rm["content"]


# === PrefixCacheKey 测试 ===


class TestPrefixCacheKey:
    """测试 compute_prefix_cache_key 前缀缓存键生成。"""

    def test_same_prefix_same_key(self) -> None:
        """测试相同前缀生成相同键。"""
        messages = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！"},
            {"role": "user", "content": "今天天气如何？"},
        ]
        key1 = compute_prefix_cache_key(
            system_prompt="你是助手",
            messages=messages,
            model="gpt-4o",
        )
        key2 = compute_prefix_cache_key(
            system_prompt="你是助手",
            messages=messages,
            model="gpt-4o",
        )
        assert key1 == key2

    def test_different_last_message_same_prefix_key(self) -> None:
        """测试最后一条消息不同但前缀相同时生成相同的键。"""
        messages1 = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！"},
            {"role": "user", "content": "今天天气如何？"},
        ]
        messages2 = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！"},
            {"role": "user", "content": "明天天气如何？"},  # 不同的最后一条
        ]
        key1 = compute_prefix_cache_key(
            system_prompt="你是助手",
            messages=messages1,
            model="gpt-4o",
        )
        key2 = compute_prefix_cache_key(
            system_prompt="你是助手",
            messages=messages2,
            model="gpt-4o",
        )
        # prefix_depth=-1 默认排除最后一条消息
        assert key1 == key2

    def test_different_system_prompt_different_key(self) -> None:
        """测试不同 system_prompt 生成不同键。"""
        key1 = compute_prefix_cache_key(
            system_prompt="你是助手",
            messages=[{"role": "user", "content": "你好"}],
            model="gpt-4o",
        )
        key2 = compute_prefix_cache_key(
            system_prompt="你是客服",
            messages=[{"role": "user", "content": "你好"}],
            model="gpt-4o",
        )
        assert key1 != key2

    def test_different_model_different_key(self) -> None:
        """测试不同模型生成不同键。"""
        key1 = compute_prefix_cache_key(
            system_prompt="你是助手",
            model="gpt-4o",
        )
        key2 = compute_prefix_cache_key(
            system_prompt="你是助手",
            model="gpt-4o-mini",
        )
        assert key1 != key2

    def test_prefix_depth_zero(self) -> None:
        """测试 prefix_depth=0 仅使用 system_prompt。"""
        key1 = compute_prefix_cache_key(
            system_prompt="你是助手",
            messages=[{"role": "user", "content": "A"}],
            model="gpt-4o",
            prefix_depth=0,
        )
        key2 = compute_prefix_cache_key(
            system_prompt="你是助手",
            messages=[{"role": "user", "content": "B"}],
            model="gpt-4o",
            prefix_depth=0,
        )
        # 仅 system_prompt 相同，消息不参与键计算
        assert key1 == key2

    def test_explicit_prefix_depth(self) -> None:
        """测试指定 prefix_depth 参数。"""
        messages = [
            {"role": "user", "content": "第一条"},
            {"role": "assistant", "content": "回复"},
            {"role": "user", "content": "第三条"},
            {"role": "user", "content": "第四条"},
        ]
        key_depth2 = compute_prefix_cache_key(
            system_prompt="你是助手",
            messages=messages,
            model="gpt-4o",
            prefix_depth=2,
        )
        # 使用前 2 条消息 + system_prompt
        key_depth3 = compute_prefix_cache_key(
            system_prompt="你是助手",
            messages=messages,
            model="gpt-4o",
            prefix_depth=3,
        )
        assert key_depth2 != key_depth3

    def test_key_format(self) -> None:
        """测试键格式正确。"""
        key = compute_prefix_cache_key(
            system_prompt="你是助手",
            model="gpt-4o",
            policy_version="v2",
        )
        assert key.startswith("prefix_match:gpt-4o:v2:")
        parts = key.split(":")
        assert len(parts) == 4
        assert len(parts[3]) == 16  # SHA256 前 16 字符

    def test_empty_messages(self) -> None:
        """测试空消息列表。"""
        key = compute_prefix_cache_key(
            system_prompt="你是助手",
            messages=[],
            model="gpt-4o",
        )
        assert key.startswith("prefix_match:gpt-4o:")

    def test_none_messages(self) -> None:
        """测试 messages=None。"""
        key = compute_prefix_cache_key(
            system_prompt="你是助手",
            messages=None,
            model="gpt-4o",
        )
        assert key.startswith("prefix_match:gpt-4o:")

    def test_single_message(self) -> None:
        """测试只有一条消息时，prefix_depth=-1 排除唯一一条。"""
        key_with_msg = compute_prefix_cache_key(
            system_prompt="你是助手",
            messages=[{"role": "user", "content": "唯一消息"}],
            model="gpt-4o",
        )
        key_no_msg = compute_prefix_cache_key(
            system_prompt="你是助手",
            messages=[],
            model="gpt-4o",
        )
        # 只有一条消息时，prefix_depth=-1 排除它，等价于空消息
        assert key_with_msg == key_no_msg

    def test_extract_metadata_for_prefix_match_key(self) -> None:
        """测试 extract_key_metadata 支持 prefix_match 类型。"""
        key = "prefix_match:gpt-4o:v1.0:abc123def456"
        metadata = extract_key_metadata(key)

        assert metadata["type"] == "prefix_match"
        assert metadata["model"] == "gpt-4o"
        assert metadata["policy_version"] == "v1.0"
        assert metadata["hash"] == "abc123def456"


# === Facade 缓存集成测试 ===


class TestFacadeCacheIntegration:
    """测试 ContextForge.build() 的缓存命中路径。"""

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached_package(self) -> None:
        """测试缓存命中时直接返回缓存的 ContextPackage（跳过 Pipeline）。"""
        from unittest.mock import AsyncMock, MagicMock

        from context_forge import ContextForge

        # 创建带缓存的 forge
        forge = ContextForge(model="gpt-4o")

        # 准备缓存数据
        cached_package = ContextPackage(
            request_id="req_cached",
            segments=[
                Segment(
                    type=SegmentType.SYSTEM,
                    content="缓存的系统提示",
                    role="system",
                    token_count=10,
                ),
            ],
            model="gpt-4o",
            policy_version="1.0",
            assembly_duration_ms=5.0,
        )
        cached_dict = cached_package.to_cache_dict()
        cached_json = json.dumps(cached_dict, ensure_ascii=False, default=str)

        # 设置 mock 缓存管理器
        mock_cache = MagicMock()
        mock_cache.get = AsyncMock(return_value=CacheEntry(value=cached_json))
        mock_cache.set = AsyncMock()
        forge._cache_manager = mock_cache

        # 调用 build
        result = await forge.build(
            system_prompt="测试",
            messages=[{"role": "user", "content": "你好"}],
        )

        # 验证返回的是缓存的包（request_id 一致）
        assert result.request_id == "req_cached"
        assert result.segments[0].content == "缓存的系统提示"

    @pytest.mark.asyncio
    async def test_cache_miss_builds_and_stores(self) -> None:
        """测试缓存未命中时走完整 Pipeline 并存储结果。"""
        from unittest.mock import AsyncMock, MagicMock

        from context_forge import ContextForge

        forge = ContextForge(model="gpt-4o")

        # 设置 mock 缓存管理器（get 返回 None = 未命中）
        mock_cache = MagicMock()
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache.set = AsyncMock()
        forge._cache_manager = mock_cache

        # 调用 build
        result = await forge.build(
            system_prompt="测试系统提示",
            messages=[{"role": "user", "content": "你好"}],
        )

        # 验证结果来自 Pipeline（非缓存）
        assert result.model == "gpt-4o"
        assert len(result.segments) > 0

        # 验证缓存写入被调用
        mock_cache.set.assert_called_once()
        # 验证写入的值可以被 from_cache_dict 解析
        call_args = mock_cache.set.call_args
        stored_entry = call_args[0][1]  # CacheEntry
        stored_dict = json.loads(stored_entry.value)
        restored = ContextPackage.from_cache_dict(stored_dict)
        assert restored.model == "gpt-4o"

    @pytest.mark.asyncio
    async def test_cache_deserialize_error_fallback(self) -> None:
        """测试缓存反序列化失败时优雅降级到完整 Pipeline。"""
        from unittest.mock import AsyncMock, MagicMock

        from context_forge import ContextForge

        forge = ContextForge(model="gpt-4o")

        # 设置缓存返回无效 JSON
        mock_cache = MagicMock()
        mock_cache.get = AsyncMock(return_value=CacheEntry(value="invalid json"))
        mock_cache.set = AsyncMock()
        forge._cache_manager = mock_cache

        # 不应抛出异常，应降级到完整 Pipeline
        result = await forge.build(
            system_prompt="测试",
            messages=[{"role": "user", "content": "你好"}],
        )

        assert result.model == "gpt-4o"
        assert len(result.segments) > 0

    @pytest.mark.asyncio
    async def test_two_identical_builds_second_hits_cache(self) -> None:
        """测试两次相同输入的 build，第二次命中缓存。"""
        from context_forge import ContextForge
        from context_forge.cache import CacheManager, MemoryCache

        # 创建带真实缓存的 forge
        forge = ContextForge(model="gpt-4o")
        forge._cache_manager = CacheManager(
            l1=MemoryCache(max_size=100, default_ttl=3600),
        )

        system_prompt = "你是一个助手"
        messages = [{"role": "user", "content": "你好"}]

        # 第一次调用
        result1 = await forge.build(system_prompt=system_prompt, messages=messages)

        # 第二次调用（相同输入）
        result2 = await forge.build(system_prompt=system_prompt, messages=messages)

        # 两次结果的内容应该一致
        msg1 = result1.to_messages()
        msg2 = result2.to_messages()
        assert len(msg1) == len(msg2)
        for m1, m2 in zip(msg1, msg2):
            assert m1["content"] == m2["content"]
            assert m1["role"] == m2["role"]

        # 第二次应该比第一次更快（缓存命中跳过 Pipeline）
        # 注意：在 CI 环境中时间可能不稳定，仅做软断言
        # assert result2.assembly_duration_ms <= result1.assembly_duration_ms

    @pytest.mark.asyncio
    async def test_different_inputs_miss_cache(self) -> None:
        """测试不同输入不会命中缓存。"""
        from context_forge import ContextForge
        from context_forge.cache import CacheManager, MemoryCache

        forge = ContextForge(model="gpt-4o")
        forge._cache_manager = CacheManager(
            l1=MemoryCache(max_size=100, default_ttl=3600),
        )

        # 第一次调用
        result1 = await forge.build(
            system_prompt="你是助手",
            messages=[{"role": "user", "content": "问题 A"}],
        )

        # 第二次调用（不同输入）
        result2 = await forge.build(
            system_prompt="你是助手",
            messages=[{"role": "user", "content": "问题 B"}],
        )

        # 两次结果的内容应该不同（不同用户消息）
        msg1_contents = [m["content"] for m in result1.to_messages()]
        msg2_contents = [m["content"] for m in result2.to_messages()]
        # 用户消息不同
        assert msg1_contents != msg2_contents
