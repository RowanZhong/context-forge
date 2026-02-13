"""
压缩模块单元测试。

测试覆盖：
1. TruncationCompressor（三种策略）
2. DedupCompressor（语义去重）
3. CompressEngine（饱和度触发 + 优先级保护）
4. CompressStage（Pipeline 集成）
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
async def test_truncation_compressor_tail():
    """测试截断压缩器 - tail 策略（保留头部）"""
    segments = [
        Segment(
            type=SegmentType.USER,
            content="第一条消息",
            role="user",
            token_count=100,
        ),
        Segment(
            type=SegmentType.ASSISTANT,
            content="第二条消息",
            role="assistant",
            token_count=100,
        ),
        Segment(
            type=SegmentType.USER,
            content="第三条消息",
            role="user",
            token_count=100,
        ),
    ]

    compressor = TruncationCompressor(strategy=TruncationStrategy.TAIL)
    context = CompressContext(
        available_tokens=1000,
        target_token_count=250,  # 保留前两条完整 + 部分第三条
        saturation=0.9,
    )

    result = await compressor.compress(segments, context)

    # 应该保留前两条完整的
    assert len(result.compressed_segments) >= 2
    assert result.compressed_segments[0].content == "第一条消息"
    assert result.compressed_segments[1].content == "第二条消息"
    assert result.original_token_count == 300
    assert result.method == "truncation_tail"


@pytest.mark.asyncio
async def test_truncation_compressor_head():
    """测试截断压缩器 - head 策略（保留尾部）"""
    segments = [
        Segment(
            type=SegmentType.USER,
            content="第一条消息",
            role="user",
            token_count=100,
        ),
        Segment(
            type=SegmentType.ASSISTANT,
            content="第二条消息",
            role="assistant",
            token_count=100,
        ),
        Segment(
            type=SegmentType.USER,
            content="第三条消息",
            role="user",
            token_count=100,
        ),
    ]

    compressor = TruncationCompressor(strategy=TruncationStrategy.HEAD)
    context = CompressContext(
        available_tokens=1000,
        target_token_count=250,  # 保留后两条完整
        saturation=0.9,
    )

    result = await compressor.compress(segments, context)

    # 应该保留后两条完整的
    assert len(result.compressed_segments) >= 2
    assert result.compressed_segments[-1].content == "第三条消息"
    assert result.compressed_segments[-2].content == "第二条消息"


@pytest.mark.asyncio
async def test_truncation_compressor_middle():
    """测试截断压缩器 - middle 策略（保留头尾）"""
    segments = [
        Segment(
            type=SegmentType.USER,
            content="第一条消息",
            role="user",
            token_count=100,
        ),
        Segment(
            type=SegmentType.ASSISTANT,
            content="第二条消息",
            role="assistant",
            token_count=100,
        ),
        Segment(
            type=SegmentType.USER,
            content="第三条消息",
            role="user",
            token_count=100,
        ),
        Segment(
            type=SegmentType.ASSISTANT,
            content="第四条消息",
            role="assistant",
            token_count=100,
        ),
    ]

    compressor = TruncationCompressor(
        strategy=TruncationStrategy.MIDDLE, head_ratio=0.5
    )
    context = CompressContext(
        available_tokens=1000,
        target_token_count=200,  # 头尾各 100
        saturation=0.9,
    )

    result = await compressor.compress(segments, context)

    # 应该保留第一条和最后一条
    assert len(result.compressed_segments) >= 2
    assert result.compressed_segments[0].content == "第一条消息"


@pytest.mark.asyncio
async def test_dedup_compressor():
    """测试去重压缩器"""
    segments = [
        Segment(
            type=SegmentType.RAG,
            content="Python 的 GIL 在 3.13 中已被移除",
            role="user",
            token_count=100,
            priority=Priority.MEDIUM,
        ),
        Segment(
            type=SegmentType.RAG,
            content="Python 的 GIL 在 3.13 中已被移除",  # 完全重复
            role="user",
            token_count=100,
            priority=Priority.LOW,  # 优先级更低，应该被删除
        ),
        Segment(
            type=SegmentType.RAG,
            content="Python 3.13 移除了 GIL 全局解释器锁",  # 高度相似
            role="user",
            token_count=100,
            priority=Priority.LOW,
        ),
        Segment(
            type=SegmentType.RAG,
            content="Rust 的所有权系统是其核心特性",  # 完全不同
            role="user",
            token_count=100,
            priority=Priority.MEDIUM,
        ),
    ]

    compressor = DedupCompressor(similarity_threshold=0.85)
    context = CompressContext(
        available_tokens=1000,
        target_token_count=400,
        saturation=0.9,
    )

    result = await compressor.compress(segments, context)

    # 应该去掉至少一个重复项
    assert len(result.compressed_segments) < len(segments)
    assert result.tokens_saved > 0
    assert result.method.startswith("dedup_jaccard")


@pytest.mark.asyncio
async def test_compress_engine_no_compression_needed():
    """测试压缩引擎 - 饱和度低不触发压缩"""
    segments = [
        Segment(
            type=SegmentType.USER,
            content="测试消息",
            role="user",
            token_count=100,
        ),
    ]

    engine = CompressEngine(saturation_threshold=0.85)
    result = await engine.compress(
        segments=segments,
        available_tokens=1000,  # 充足的预算
        audit_log=[],
    )

    # 饱和度 100/1000 = 0.1，低于阈值，不压缩
    assert len(result) == len(segments)


@pytest.mark.asyncio
async def test_compress_engine_priority_protection():
    """测试压缩引擎 - 优先级保护"""
    segments = [
        Segment(
            type=SegmentType.SYSTEM,
            content="系统提示" * 100,
            role="system",
            token_count=300,
            priority=Priority.CRITICAL,
        ),
        Segment(
            type=SegmentType.RAG,
            content="RAG 片段 1",
            role="user",
            token_count=100,
            priority=Priority.LOW,
        ),
        Segment(
            type=SegmentType.RAG,
            content="RAG 片段 2",
            role="user",
            token_count=100,
            priority=Priority.LOW,
        ),
    ]

    engine = CompressEngine(saturation_threshold=0.5)
    result = await engine.compress(
        segments=segments,
        available_tokens=400,  # 需要压缩
        audit_log=[],
    )

    # CRITICAL 应该被保护，LOW 被压缩
    critical_segments = [
        seg for seg in result if seg.effective_priority == Priority.CRITICAL
    ]
    assert len(critical_segments) > 0
    assert critical_segments[0].content.startswith("系统提示")


@pytest.mark.asyncio
async def test_compress_engine_must_keep_protection():
    """测试压缩引擎 - must_keep 保护"""
    segments = [
        Segment(
            type=SegmentType.USER,
            content="必须保留的消息",
            role="user",
            token_count=100,
            priority=Priority.LOW,
            control=ControlFlags(must_keep=True),
        ),
        Segment(
            type=SegmentType.RAG,
            content="可压缩的 RAG 片段",
            role="user",
            token_count=100,
            priority=Priority.LOW,
        ),
    ]

    engine = CompressEngine(saturation_threshold=0.5, preserve_must_keep=True)
    result = await engine.compress(
        segments=segments,
        available_tokens=150,
        audit_log=[],
    )

    # must_keep 的 Segment 应该被保留
    must_keep_segments = [seg for seg in result if seg.control.must_keep]
    assert len(must_keep_segments) > 0
    assert must_keep_segments[0].content == "必须保留的消息"


@pytest.mark.asyncio
async def test_compress_stage_integration():
    """测试压缩阶段 - Pipeline 集成"""
    segments = [
        Segment(
            type=SegmentType.RAG,
            content=f"RAG 片段 {i}",
            role="user",
            token_count=100,
            priority=Priority.LOW,
        )
        for i in range(10)
    ]

    stage = CompressStage(engine=CompressEngine(saturation_threshold=0.5))
    context = PipelineContext(metadata={"available_tokens": 500})

    result = await stage.process(segments, context)

    # 应该触发压缩（1000 token 超过 500 的 50%）
    total_tokens = sum(seg.token_count or 0 for seg in result)
    assert total_tokens <= 500


@pytest.mark.asyncio
async def test_compression_result_properties():
    """测试 CompressionResult 属性"""
    from context_forge.compress.base import CompressionResult

    result = CompressionResult(
        compressed_segments=[],
        original_token_count=1000,
        compressed_token_count=300,
        method="test",
        parent_segment_ids=["seg1", "seg2"],
    )

    assert result.compression_ratio == 0.3
    assert result.tokens_saved == 700


def test_compress_context():
    """测试 CompressContext"""
    context = CompressContext(
        available_tokens=100000,
        target_token_count=80000,
        saturation=0.85,
        model_name="gpt-4o",
    )

    assert context.available_tokens == 100000
    assert context.target_token_count == 80000
    assert context.saturation == 0.85
    assert context.model_name == "gpt-4o"
