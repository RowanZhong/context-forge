"""
压缩模块集成测试 — 完整的端到端测试。

验证压缩模块在真实场景中的表现。
"""

import pytest

from context_forge.compress import (
    CompressContext,
    CompressEngine,
    DedupCompressor,
    TruncationCompressor,
    TruncationStrategy,
)
from context_forge.models.control import ControlFlags
from context_forge.models.segment import Priority, Segment, SegmentType
from context_forge.pipeline.base import PipelineContext
from context_forge.pipeline.compress_stage import CompressStage


@pytest.mark.asyncio
async def test_full_compress_pipeline():
    """测试完整的压缩流水线：去重 → 压缩 → 保护"""
    # 模拟 RAG 场景：多个重复片段 + 系统提示 + 用户消息
    segments = [
        # 系统提示（CRITICAL，不可压缩）
        Segment(
            type=SegmentType.SYSTEM,
            content="你是一个专业的 AI 助手" * 50,
            role="system",
            token_count=500,
            priority=Priority.CRITICAL,
        ),
        # 用户消息（HIGH + must_keep，受保护）
        Segment(
            type=SegmentType.USER,
            content="请解释 Python GIL",
            role="user",
            token_count=100,
            priority=Priority.HIGH,
            control=ControlFlags(must_keep=True),
        ),
        # RAG 片段（部分重复，LOW 优先级，可压缩）
        Segment(
            type=SegmentType.RAG,
            content="Python 的 GIL 在 3.13 中已被移除",
            role="user",
            token_count=100,
            priority=Priority.LOW,
        ),
        Segment(
            type=SegmentType.RAG,
            content="Python 3.13 移除了 GIL 全局解释器锁",  # 相似
            role="user",
            token_count=100,
            priority=Priority.LOW,
        ),
        Segment(
            type=SegmentType.RAG,
            content="Python 的 GIL 在 3.13 中已被移除",  # 完全重复
            role="user",
            token_count=100,
            priority=Priority.LOW,
        ),
        # 更多 RAG 片段
        *[
            Segment(
                type=SegmentType.RAG,
                content=f"其他 RAG 片段 {i}",
                role="user",
                token_count=100,
                priority=Priority.LOW,
            )
            for i in range(5)
        ],
    ]

    original_count = len(segments)
    original_tokens = sum(seg.token_count or 0 for seg in segments)

    # 创建压缩引擎（低饱和度阈值，容易触发）
    engine = CompressEngine(
        saturation_threshold=0.5,  # 50% 就触发
        preserve_must_keep=True,
    )

    # 执行压缩
    audit_log = []
    compressed = await engine.compress(
        segments=segments,
        available_tokens=800,  # 紧张的预算
        audit_log=audit_log,
    )

    compressed_count = len(compressed)
    compressed_tokens = sum(seg.token_count or 0 for seg in compressed)

    # 验证结果
    assert compressed_count < original_count, "应该删除了一些 Segment"
    assert compressed_tokens <= 800, "应该满足预算限制"
    assert compressed_tokens > 0, "不应该删除所有 Segment"

    # 验证 CRITICAL 被保护
    critical_segments = [
        seg for seg in compressed if seg.effective_priority == Priority.CRITICAL
    ]
    assert len(critical_segments) > 0, "CRITICAL Segment 应该被保留"

    # 验证 must_keep 被保护
    must_keep_segments = [
        seg for seg in compressed if seg.control and seg.control.must_keep
    ]
    assert len(must_keep_segments) > 0, "must_keep Segment 应该被保留"

    print(f"\n压缩结果：")
    print(f"  原始：{original_count} 个 Segment，{original_tokens} Token")
    print(f"  压缩后：{compressed_count} 个 Segment，{compressed_tokens} Token")
    print(f"  节省：{original_tokens - compressed_tokens} Token")


@pytest.mark.asyncio
async def test_compress_stage_in_pipeline():
    """测试 CompressStage 在 Pipeline 中的集成"""
    segments = [
        Segment(
            type=SegmentType.RAG,
            content=f"RAG 片段 {i}",
            role="user",
            token_count=100,
            priority=Priority.LOW,
        )
        for i in range(20)
    ]

    # 创建 Pipeline 上下文
    context = PipelineContext(
        metadata={
            "available_tokens": 1000,  # 预算 1000，但有 2000 token
            "model_name": "gpt-4o",
        }
    )

    # 创建压缩阶段
    stage = CompressStage(
        engine=CompressEngine(saturation_threshold=0.8)  # 80% 饱和度触发
    )

    # 执行压缩阶段
    result = await stage.process(segments, context)

    # 验证
    total_tokens = sum(seg.token_count or 0 for seg in result)
    assert total_tokens <= 1000, "应该满足预算限制"

    # 检查审计日志
    assert len(context.audit_log) > 0, "应该记录了审计日志"
    assert any(
        entry.decision.value == "compress" for entry in context.audit_log
    ), "应该有压缩决策记录"


@pytest.mark.asyncio
async def test_compression_preserves_provenance():
    """测试压缩保留 Provenance 溯源信息"""
    segments = [
        Segment(
            id=f"seg_{i}",
            type=SegmentType.RAG,
            content=f"原始片段 {i}",
            role="user",
            token_count=100,
        )
        for i in range(5)
    ]

    compressor = TruncationCompressor(strategy=TruncationStrategy.TAIL)
    context = CompressContext(
        available_tokens=1000,
        target_token_count=200,
        saturation=0.9,
    )

    result = await compressor.compress(segments, context)

    # 验证 Provenance
    for seg in result.compressed_segments:
        assert seg.provenance is not None, "应该有 Provenance"
        assert (
            seg.provenance.source_type.value == "compression"
        ), "来源类型应该是 compression"
        assert (
            seg.provenance.compression_method == "truncation_tail"
        ), "应该记录压缩方法"
        assert len(seg.provenance.parent_segment_ids) > 0, "应该记录父 Segment ID"


@pytest.mark.asyncio
async def test_dedup_with_different_priorities():
    """测试去重时的优先级处理"""
    segments = [
        Segment(
            type=SegmentType.RAG,
            content="相同的内容",
            role="user",
            token_count=100,
            priority=Priority.HIGH,  # 高优先级
        ),
        Segment(
            type=SegmentType.RAG,
            content="相同的内容",
            role="user",
            token_count=100,
            priority=Priority.LOW,  # 低优先级，应该被删除
        ),
    ]

    compressor = DedupCompressor(similarity_threshold=0.5)  # 低阈值，容易匹配
    context = CompressContext(
        available_tokens=1000,
        target_token_count=200,
        saturation=0.9,
    )

    result = await compressor.compress(segments, context)

    # 应该只保留一个
    assert len(result.compressed_segments) == 1, "应该去重到 1 个 Segment"

    # 保留的应该是高优先级的
    kept = result.compressed_segments[0]
    assert kept.effective_priority == Priority.HIGH, "应该保留高优先级的"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
