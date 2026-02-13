"""
测试套件共享 Fixtures 和配置。

本文件定义了所有测试中可复用的 fixtures、mock 对象和辅助函数。
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from context_forge import ContextForge
from context_forge.config.schema import PolicyConfig
from context_forge.models.audit import AuditEntry, DecisionType, ReasonCode
from context_forge.models.budget import BudgetAllocation, BudgetPolicy
from context_forge.models.context_package import ContextPackage, TokenUsage
from context_forge.models.control import ControlFlags
from context_forge.models.metadata import SegmentMetadata
from context_forge.models.provenance import Provenance, SourceType
from context_forge.models.routing import ComplexityLevel, ModelConfig, RoutingDecision, RoutingRule
from context_forge.models.segment import Priority, Segment, SegmentType
from context_forge.pipeline.base import Pipeline, PipelineContext


# === 数据模型 Fixtures ===


@pytest.fixture
def sample_segment() -> Segment:
    """基础 Segment 示例。"""
    return Segment(
        type=SegmentType.USER,
        content="测试内容",
        role="user",
    )


@pytest.fixture
def system_segment() -> Segment:
    """System Segment 示例（CRITICAL 优先级）。"""
    return Segment(
        type=SegmentType.SYSTEM,
        content="你是一个有用的助手。",
        role="system",
        priority=Priority.CRITICAL,
        control=ControlFlags(
            must_keep=True,
            lock_position=True,
            compressible=False,
        ),
    )


@pytest.fixture
def rag_segments() -> list[Segment]:
    """RAG Segment 列表示例（带检索分数）。"""
    return [
        Segment(
            type=SegmentType.RAG,
            content="Python 3.13 移除了 GIL",
            role="user",
            provenance=Provenance(
                source_id="doc_001",
                source_type=SourceType.RAG_RETRIEVAL,
                retrieval_score=0.95,
            ),
            metadata=SegmentMetadata(retrieval_score=0.95),
        ),
        Segment(
            type=SegmentType.RAG,
            content="异步编程提高并发性能",
            role="user",
            provenance=Provenance(
                source_id="doc_002",
                source_type=SourceType.RAG_RETRIEVAL,
                retrieval_score=0.87,
            ),
            metadata=SegmentMetadata(retrieval_score=0.87),
        ),
        Segment(
            type=SegmentType.RAG,
            content="类型提示改善代码可读性",
            role="user",
            provenance=Provenance(
                source_id="doc_003",
                source_type=SourceType.RAG_RETRIEVAL,
                retrieval_score=0.72,
            ),
            metadata=SegmentMetadata(retrieval_score=0.72),
        ),
    ]


@pytest.fixture
def conversation_segments() -> list[Segment]:
    """对话历史示例（多轮）。"""
    return [
        Segment(
            type=SegmentType.USER,
            content="你好",
            role="user",
            metadata=SegmentMetadata(turn_number=0),
        ),
        Segment(
            type=SegmentType.ASSISTANT,
            content="你好！有什么可以帮助你的吗？",
            role="assistant",
            metadata=SegmentMetadata(turn_number=0),
        ),
        Segment(
            type=SegmentType.USER,
            content="介绍一下 Python 的 GIL",
            role="user",
            metadata=SegmentMetadata(turn_number=1),
        ),
        Segment(
            type=SegmentType.ASSISTANT,
            content="GIL（全局解释器锁）是 Python 解释器中的一个机制...",
            role="assistant",
            metadata=SegmentMetadata(turn_number=1),
        ),
    ]


@pytest.fixture
def expired_segment() -> Segment:
    """已过期的 Segment（TTL 已超时）。"""
    return Segment(
        type=SegmentType.ASSISTANT,
        content="旧的对话内容",
        role="assistant",
        control=ControlFlags(
            ttl=2,  # 2 轮 TTL
        ),
        created_at=datetime.now(timezone.utc) - timedelta(seconds=120),  # 2 分钟前创建
    )


@pytest.fixture
def budget_policy() -> BudgetPolicy:
    """标准预算策略。"""
    return BudgetPolicy(
        max_context_tokens=8192,
        output_reserved_tokens=1024,
        thinking_reserved_tokens=0,
        rigid_segment_types=[
            SegmentType.SYSTEM,
            SegmentType.SCHEMA,
        ],
        elastic_ratios={
            SegmentType.USER: 0.25,
            SegmentType.ASSISTANT: 0.20,
            SegmentType.RAG: 0.30,
        },
    )


@pytest.fixture
def budget_allocation() -> BudgetAllocation:
    """预算分配记录示例。"""
    return BudgetAllocation(
        total_budget=8192,
        content_budget=7168,  # 8192 - 1024
        total_used=5000,
        rigid_used=500,
        elastic_used={"system": 400, "user": 1500, "rag": 2000},
        output_reserved=1024,
    )


@pytest.fixture
def provenance() -> Provenance:
    """来源溯源示例。"""
    return Provenance(
        source_id="rag_retrieval_001",
        source_type=SourceType.RAG_RETRIEVAL,
        uri="https://docs.python.org/3.13/whatsnew/",
        retrieval_score=0.92,
        custom={"created_by": "vector_search_engine"},
    )


@pytest.fixture
def control_flags() -> ControlFlags:
    """控制标志示例。"""
    return ControlFlags(
        must_keep=True,
        lock_position=False,
        compressible=True,
        ttl=5,
        namespace="rag",
    )


@pytest.fixture
def segment_metadata() -> SegmentMetadata:
    """Segment 元数据示例。"""
    return SegmentMetadata(
        retrieval_score=0.88,
        turn_number=5,
        rerank_score=0.92,
        debug_labels={"category": "technical", "language": "zh"},
    )


@pytest.fixture
def audit_entry() -> AuditEntry:
    """审计条目示例。"""
    return AuditEntry(
        segment_id="seg_abc123",
        pipeline_stage="allocate",
        decision=DecisionType.KEEP,
        reason_code=ReasonCode.RIGID_GUARANTEED,
        reason_detail="预算充足，保留所有 HIGH 优先级 Segment",
        token_impact=0,
    )


@pytest.fixture
def model_config() -> ModelConfig:
    """模型配置示例。"""
    return ModelConfig(
        model_id="gpt-4o",
        provider="openai",
        max_context_tokens=128000,
        max_output_tokens=16384,
        supports_thinking=False,
        supports_vision=True,
        cost_per_million_input=2.5,
        cost_per_million_output=10.0,
    )


@pytest.fixture
def routing_decision() -> RoutingDecision:
    """路由决策示例。"""
    return RoutingDecision(
        selected_model=ModelConfig(
            model_id="gpt-4o-mini",
            provider="openai",
            max_context_tokens=128000,
        ),
        complexity=ComplexityLevel.SIMPLE,
        estimated_cost=0.05,
        confidence=0.92,
        reasoning="简单查询任务，使用小模型降低成本",
    )


@pytest.fixture
def context_package(
    system_segment: Segment,
    conversation_segments: list[Segment],
    budget_allocation: BudgetAllocation,
) -> ContextPackage:
    """完整的 ContextPackage 示例。"""
    all_segments = [system_segment] + conversation_segments
    return ContextPackage(
        segments=all_segments,
        audit_log=[],
        budget_allocation=budget_allocation,
        model="gpt-4o",
        policy_version="1.0.0",
        assembly_duration_ms=45.2,
    )


# === Pipeline 和 Context Fixtures ===


@pytest.fixture
def pipeline_context(budget_policy: BudgetPolicy) -> PipelineContext:
    """Pipeline 上下文示例。"""
    return PipelineContext(
        model="gpt-4o",
        budget_policy=budget_policy,
        current_turn=3,
        target_namespace="default",
        debug=False,
    )


@pytest.fixture
def default_pipeline() -> Pipeline:
    """默认流水线实例。"""
    from context_forge.pipeline.base import create_default_pipeline
    return create_default_pipeline()


# === Mock 对象 Fixtures ===


@pytest.fixture
def mock_llm_response() -> dict[str, Any]:
    """Mock LLM 响应（用于压缩等场景）。"""
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "这是一段压缩后的摘要内容。",
                }
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
        },
    }


@pytest.fixture
def mock_cache_manager() -> MagicMock:
    """Mock 缓存管理器。"""
    cache = MagicMock()
    cache.get = AsyncMock(return_value=None)  # 默认缓存未命中
    cache.set = AsyncMock(return_value=None)
    cache.delete = AsyncMock(return_value=None)
    cache.clear = AsyncMock(return_value=None)
    return cache


@pytest.fixture
def mock_router() -> MagicMock:
    """Mock 路由器（同步 route 方法）。"""
    router = MagicMock()
    # route 方法应该返回同步结果，不是异步
    router.route.return_value = RoutingDecision(
        selected_model=ModelConfig(
            model_id="gpt-4o",
            provider="openai",
            max_context_tokens=128000,
        ),
        complexity=ComplexityLevel.MODERATE,
        estimated_cost=0.1,
        confidence=0.85,
        reasoning="默认模型",
    )
    return router


@pytest.fixture
def mock_metrics_collector() -> MagicMock:
    """Mock 指标收集器。"""
    collector = MagicMock()
    collector.record = MagicMock()
    collector.collect_from_package = MagicMock()
    return collector


@pytest.fixture
def mock_snapshot_manager() -> MagicMock:
    """Mock 快照管理器。"""
    manager = MagicMock()
    manager.save = AsyncMock(return_value="snapshot_abc123")
    manager.load = AsyncMock(return_value={
        "segments": [],
        "budget_allocation": {},
        "model": "gpt-4o",
    })
    return manager


# === 配置 Fixtures ===


@pytest.fixture
def default_policy() -> PolicyConfig:
    """默认策略配置。"""
    from context_forge.config.loader import load_policy
    return load_policy()


@pytest.fixture
def custom_policy_dict() -> dict[str, Any]:
    """自定义策略配置字典（用于测试 YAML 加载）。"""
    return {
        "version": "test-1.0.0",
        "name": "test-policy",
        "budget": {
            "max_context_tokens": 4096,
            "output_reserved_tokens": 512,
            "thinking_reserved_tokens": 0,
            "elastic_ratios": {
                "user": 0.4,
                "assistant": 0.3,
            },
        },
        "sanitize": {
            "strip_html": True,
            "injection_detection": True,
            "on_injection": "reject",
        },
        "rerank": {
            "enable_mmr": True,
            "mmr_lambda": 0.7,
        },
        "compress": {
            "enabled": False,
        },
        "cache": {
            "enabled": False,
        },
        "routing": {
            "enabled": False,
        },
        "observability": {
            "metrics_enabled": False,
            "snapshot_enabled": False,
        },
        "antipattern": {
            "check_on_build": False,
        },
    }


@pytest.fixture
def temp_policy_file(tmp_path: Path, custom_policy_dict: dict[str, Any]) -> Path:
    """创建临时策略 YAML 文件。"""
    import yaml

    policy_file = tmp_path / "test_policy.yaml"
    with open(policy_file, "w", encoding="utf-8") as f:
        yaml.dump(custom_policy_dict, f, allow_unicode=True)

    return policy_file


# === Facade Fixtures ===


@pytest.fixture
def context_forge() -> ContextForge:
    """基础 ContextForge 实例（无外部依赖）。"""
    return ContextForge(model="gpt-4o")


@pytest.fixture
def context_forge_with_cache(mock_cache_manager: MagicMock) -> ContextForge:
    """启用缓存的 ContextForge 实例。"""
    return ContextForge(
        model="gpt-4o",
        cache_backend=mock_cache_manager,
    )


@pytest.fixture
def sample_segments() -> list[Segment]:
    """
    提供 10+ 个多样化的 Segment 用于Batch 1 测试。

    包含所有主要类型、优先级、控制标志和元数据组合，
    确保测试覆盖各种边界情况。
    """
    base_time = datetime.now(timezone.utc) - timedelta(days=1)

    segments = [
        # 1. System Prompt (CRITICAL, must_keep)
        Segment(
            type=SegmentType.SYSTEM,
            content="你是一个专业的技术助手，专注于 Python 和 LLM 应用开发。",
            role="system",
            priority=Priority.CRITICAL,
            control=ControlFlags(must_keep=True),
            provenance=Provenance(
                source_type=SourceType.SYSTEM_CONFIG,
                source_id="system_prompt_v1",
            ),
            created_at=base_time,
        ),
        # 2. Tool Definition (HIGH)
        Segment(
            type=SegmentType.TOOL_DEFINITION,
            content='{"name": "search", "description": "搜索文档", "parameters": {"query": "string"}}',
            role="system",
            priority=Priority.HIGH,
            provenance=Provenance(
                source_type=SourceType.SYSTEM_CONFIG,
                source_id="tool_search",
            ),
            created_at=base_time + timedelta(minutes=1),
        ),
        # 3. Few-shot Example (HIGH)
        Segment(
            type=SegmentType.FEW_SHOT,
            content="Q: 什么是 GIL？\nA: GIL 是 Python 的全局解释器锁...",
            role="user",
            priority=Priority.HIGH,
            provenance=Provenance(
                source_type=SourceType.SYSTEM_CONFIG,
                source_id="few_shot_example_1",
            ),
            created_at=base_time + timedelta(minutes=2),
        ),
        # 4. User Message (HIGH, must_keep)
        Segment(
            type=SegmentType.USER,
            content="我想了解 Context Forge 的预算管理机制。",
            role="user",
            priority=Priority.HIGH,
            control=ControlFlags(must_keep=True),
            provenance=Provenance(
                source_type=SourceType.USER_INPUT,
                source_id="user_123",
            ),
            created_at=base_time + timedelta(minutes=3),
        ),
        # 5. RAG Chunk (MEDIUM, high relevance score)
        Segment(
            type=SegmentType.RAG,
            content="Context Forge 的预算管理采用刚性 + 弹性双层分配机制...",
            role="user",
            priority=Priority.MEDIUM,
            provenance=Provenance(
                source_type=SourceType.RAG_RETRIEVAL,
                source_id="doc_001",
                retrieval_score=0.92,
            ),
            metadata=SegmentMetadata(
                rerank_score=0.95,
                debug_labels={"category": "budget", "topic": "core_concept"},
            ),
            created_at=base_time + timedelta(minutes=4),
        ),
        # 6. RAG Chunk (MEDIUM, medium relevance score)
        Segment(
            type=SegmentType.RAG,
            content="预算溢出时，Context Forge 支持三种策略：截断 / 压缩 / 报错。",
            role="user",
            priority=Priority.MEDIUM,
            provenance=Provenance(
                source_type=SourceType.RAG_RETRIEVAL,
                source_id="doc_002",
                retrieval_score=0.78,
            ),
            metadata=SegmentMetadata(
                rerank_score=0.80,
                debug_labels={"category": "budget", "topic": "overflow"},
            ),
            created_at=base_time + timedelta(minutes=5),
        ),
        # 7. Assistant Message (MEDIUM)
        Segment(
            type=SegmentType.ASSISTANT,
            content="根据你的需求，我建议使用 Context Forge 的弹性预算配置...",
            role="assistant",
            priority=Priority.MEDIUM,
            provenance=Provenance(
                source_type=SourceType.MANUAL_INJECTION,
                source_id="assistant",
            ),
            created_at=base_time + timedelta(minutes=6),
        ),
        # 8. State Snapshot (HIGH, lock_position)
        Segment(
            type=SegmentType.STATE,
            content="当前状态：用户正在咨询预算管理功能，已检索 3 篇文档。",
            role="user",
            priority=Priority.HIGH,
            control=ControlFlags(lock_position=True),
            provenance=Provenance(
                source_type=SourceType.SYSTEM_CONFIG,
                source_id="state_snapshot_1",
            ),
            created_at=base_time + timedelta(minutes=7),
        ),
        # 9. Tool Result (MEDIUM)
        Segment(
            type=SegmentType.TOOL_RESULT,
            content='{"results": ["doc_001", "doc_002", "doc_003"], "total": 3}',
            role="user",
            priority=Priority.MEDIUM,
            provenance=Provenance(
                source_type=SourceType.TOOL_RESPONSE,
                source_id="tool_search_result",
            ),
            created_at=base_time + timedelta(minutes=8),
        ),
        # 10. Old Conversation (LOW, TTL expired)
        Segment(
            type=SegmentType.ASSISTANT,
            content="这是一条很久以前的对话内容...",
            role="assistant",
            priority=Priority.LOW,
            control=ControlFlags(ttl=3),  # 3 轮 TTL
            provenance=Provenance(
                source_type=SourceType.MANUAL_INJECTION,
                source_id="assistant",
            ),
            created_at=base_time - timedelta(days=2),  # 2 天前（已过期）
        ),
        # 11. Summary (MEDIUM)
        Segment(
            type=SegmentType.SUMMARY,
            content="历史对话摘要：用户咨询了 RAG 架构、预算管理、安全清洗等话题。",
            role="user",
            priority=Priority.MEDIUM,
            provenance=Provenance(
                source_type=SourceType.COMPRESSION,
                source_id="summary_compressor",
            ),
            created_at=base_time + timedelta(minutes=9),
        ),
        # 12. Schema Definition (CRITICAL, must_keep)
        Segment(
            type=SegmentType.SCHEMA,
            content='{"type": "object", "properties": {"answer": {"type": "string"}}}',
            role="system",
            priority=Priority.CRITICAL,
            control=ControlFlags(must_keep=True),
            provenance=Provenance(
                source_type=SourceType.SYSTEM_CONFIG,
                source_id="output_schema",
            ),
            created_at=base_time + timedelta(minutes=10),
        ),
    ]

    return segments


@pytest.fixture
def sample_rag_chunks() -> list[Segment]:
    """
    提供 10 个 RAG chunks 用于测试重排、去重、MMR 等功能。

    包含不同相关性分数、相似内容、时效性等特征。
    """
    base_time = datetime.now(timezone.utc) - timedelta(days=7)

    chunks = []
    for i in range(10):
        # 模拟相关性分数递减
        score = 0.95 - i * 0.05

        # 模拟相似内容（第 5、6 条内容相似）
        if i == 5:
            content = "Python 的 GIL 在 3.13 版本中被移除，多线程性能显著提升。"
        elif i == 6:
            content = "Python 3.13 移除了全局解释器锁（GIL），这对多线程应用是重大改进。"
        else:
            content = f"这是第 {i + 1} 个 RAG 检索片段，包含关于 Python 的技术内容。Score: {score:.2f}"

        chunk = Segment(
            type=SegmentType.RAG,
            content=content,
            role="user",
            priority=Priority.MEDIUM,
            provenance=Provenance(
                source_type=SourceType.RAG_RETRIEVAL,
                source_id=f"doc_{i:03d}",
                retrieval_score=score,
            ),
            metadata=SegmentMetadata(
                rerank_score=score,
                debug_labels={"chunk_id": f"chunk_{i}", "language": "python"},
            ),
            created_at=base_time + timedelta(days=i),  # 时效性递增
        )
        chunks.append(chunk)

    return chunks


@pytest.fixture
def sample_conversation() -> list[Segment]:
    """
    提供 20 轮（40 条）对话历史用于测试压缩、记忆管理等功能。
    """
    base_time = datetime.now(timezone.utc) - timedelta(hours=2)
    history = []

    topics = [
        "Python 基础",
        "异步编程",
        "类型系统",
        "性能优化",
        "测试框架",
    ]

    for turn in range(20):
        topic = topics[turn % len(topics)]

        # User message
        user = Segment(
            type=SegmentType.USER,
            content=f"我想了解 {topic} 的相关内容。",
            role="user",
            priority=Priority.HIGH if turn >= 18 else Priority.MEDIUM,  # 最近 2 轮 HIGH
            provenance=Provenance(
                source_type=SourceType.USER_INPUT,
                source_id="user_001",
            ),
            created_at=base_time + timedelta(minutes=turn * 2),
        )
        history.append(user)

        # Assistant message
        assistant = Segment(
            type=SegmentType.ASSISTANT,
            content=f"关于 {topic}，我可以从以下几个方面为你讲解：...",
            role="assistant",
            priority=Priority.MEDIUM if turn < 18 else Priority.HIGH,
            provenance=Provenance(
                source_type=SourceType.MANUAL_INJECTION,
                source_id="assistant",
            ),
            created_at=base_time + timedelta(minutes=turn * 2 + 1),
        )
        history.append(assistant)

    return history


@pytest.fixture
def minimal_forge() -> ContextForge:
    """
    提供默认配置的 ContextForge 实例。

    适用于基础功能测试，不涉及高级特性（缓存、路由等）。
    """
    return ContextForge(model="gpt-4o")


@pytest.fixture
def full_forge() -> ContextForge:
    """
    提供全功能配置的 ContextForge 实例。

    启用所有特性：缓存、压缩、路由、可观测性等。
    """
    policy = PolicyConfig(
        budget={"max_context_tokens": 128_000},
        sanitize={"pii_redaction": True, "injection_detection": True},
        rerank={"enable_mmr": True, "enable_temporal_weighting": True},
        compress={"enabled": True},
        cache={"enabled": True, "backend": "memory"},
        routing={"enabled": False},  # 路由在测试中默认关闭（避免复杂性）
        observability={"snapshot_enabled": True, "metrics_enabled": True},
    )

    # Save policy to a temp file since ContextForge takes policy_path, not policy object
    import tempfile
    import yaml
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(policy.model_dump(), f, allow_unicode=True)
        policy_path = f.name
    return ContextForge(model="gpt-4o", policy_path=policy_path)


@pytest.fixture
def mock_policy_path(tmp_path: Path) -> Path:
    """
    创建临时 YAML 策略文件用于测试。

    参数:
        tmp_path: Pytest 提供的临时目录

    返回:
        策略文件路径
    """
    policy_yaml = """
version: "1.0"
name: "test_policy"
description: "测试专用策略"

budget:
  max_context_tokens: 64000
  output_reserved_tokens: 2048
  saturation_threshold: 0.85

sanitize:
  unicode_normalize: true
  strip_html: true
  pii_redaction: true
  injection_detection: true
  on_injection: "warn_and_remove"

rerank:
  enable_mmr: true
  mmr_lambda: 0.7
  similarity_threshold: 0.85

compress:
  enabled: true
  default_compressor: "truncation"

cache:
  enabled: true
  backend: "memory"

observability:
  snapshot_enabled: true
  metrics_enabled: true
"""

    policy_file = tmp_path / "test_policy.yaml"
    policy_file.write_text(policy_yaml, encoding="utf-8")
    return policy_file


@pytest.fixture
def sample_package(sample_segments: list[Segment]) -> ContextPackage:
    """
    提供预构建的 ContextPackage 用于测试。

    使用 sample_segments 中的前 5 个 Segment。
    """
    segments = sample_segments[:5]

    # 填充 token_count（模拟 Normalize 阶段）
    segments_with_tokens = []
    for seg in segments:
        # 粗估：每个字符约 0.5 token（中文）
        token_count = int(len(seg.content) * 0.5)
        segments_with_tokens.append(seg.with_token_count(token_count))

    total_tokens = sum(seg.token_count or 0 for seg in segments_with_tokens)

    return ContextPackage(
        segments=segments_with_tokens,
        model="gpt-4o",
        budget_allocation=BudgetAllocation(
            total_budget=128_000,
            content_budget=128_000 - 4_096,
            total_used=total_tokens,
            output_reserved=4_096,
        ),
    )


@pytest.fixture
def tmp_snapshot_dir(tmp_path: Path) -> Path:
    """
    创建临时快照目录用于测试可观测性功能。

    参数:
        tmp_path: Pytest 提供的临时目录

    返回:
        快照目录路径
    """
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    return snapshot_dir


@pytest.fixture
def mock_llm_fn() -> Any:
    """
    提供异步 mock LLM 函数用于测试。

    返回一个可调用对象，模拟 LLM API 的行为。
    """

    async def _mock_llm(prompt: str, max_tokens: int = 100, **kwargs: Any) -> str:
        """
        Mock LLM 函数。

        参数:
            prompt: 输入 Prompt
            max_tokens: 最大 Token 数
            **kwargs: 其他参数（忽略）

        返回:
            Mock 响应
        """
        # 根据 prompt 内容返回不同响应
        if "摘要" in prompt or "summary" in prompt.lower():
            return "这是一段摘要内容。核心观点已保留，细节被省略。"
        elif "压缩" in prompt or "compress" in prompt.lower():
            return "压缩后的内容"
        elif "injection" in prompt.lower() or "安全" in prompt:
            return "safe"
        elif "路由" in prompt or "route" in prompt.lower():
            return "gpt-4o-mini"
        else:
            return "这是一个 Mock LLM 响应。"

    return _mock_llm


# === 辅助函数 ===


def create_segments_with_tokens(
    count: int,
    tokens_per_segment: int = 100,
    segment_type: SegmentType = SegmentType.RAG,
) -> list[Segment]:
    """创建指定数量的带 token 计数的 Segment 列表。"""
    segments = []
    for i in range(count):
        seg = Segment(
            type=segment_type,
            content=f"测试内容 {i}",
            role="user",
        ).with_token_count(tokens_per_segment)
        segments.append(seg)
    return segments


def assert_segment_immutable(segment: Segment) -> None:
    """断言 Segment 是不可变的。"""
    import pytest
    from pydantic import ValidationError

    with pytest.raises((ValidationError, AttributeError)):
        segment.content = "尝试修改"  # type: ignore


def assert_package_valid(package: ContextPackage) -> None:
    """断言 ContextPackage 满足基本不变量。"""
    assert len(package.segments) > 0, "Package 必须包含至少一个 Segment"
    assert package.token_usage.total_tokens > 0, "总 Token 数必须大于 0"
    assert package.budget_allocation.total_budget > 0, "预算必须大于 0"
    assert package.assembly_duration_ms >= 0, "组装耗时不能为负"

    # 验证 Segment 顺序
    system_segments = [s for s in package.segments if s.type == SegmentType.SYSTEM]
    if system_segments:
        assert package.segments[0].type == SegmentType.SYSTEM, "SYSTEM Segment 必须在开头"


# === Pytest 配置 ===


def pytest_configure(config: Any) -> None:
    """Pytest 配置钩子。"""
    config.addinivalue_line(
        "markers", "integration: 标记集成测试（需要多个模块协作）"
    )
    config.addinivalue_line(
        "markers", "e2e: 标记端到端测试（完整场景测试）"
    )
    config.addinivalue_line(
        "markers", "slow: 标记慢速测试（运行时间 > 1s）"
    )
