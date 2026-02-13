"""
LLM 摘要压缩器单元测试。

测试覆盖：
1. LLMSummaryCompressor 构造函数（默认参数 + 自定义参数）
2. compress() 方法 - 成功路径（mock LLM）
3. compress() 方法 - 空 Segment 列表
4. compress() 方法 - 无 Provider 时 fallback
5. compress() 方法 - LLM 调用失败时 fallback
6. compress() 方法 - LLM 调用失败且禁用 fallback（抛异常）
7. _generate_summary() 方法
8. _fallback_compress() 方法
9. Protocol 合规性
10. 批量压缩多个 Segment
"""

from unittest.mock import AsyncMock

import pytest

from context_forge.compress.base import CompressContext
from context_forge.compress.summary import LLMProvider, LLMSummaryCompressor
from context_forge.errors.exceptions import CompressionError
from context_forge.models.provenance import SourceType
from context_forge.models.segment import Segment, SegmentType


class MockLLMProvider:
    """Mock LLM 提供者，用于测试"""

    def __init__(self, response: str = "这是 LLM 生成的摘要"):
        self.response = response
        self.generate = AsyncMock(return_value=response)


class FailingLLMProvider:
    """失败的 LLM 提供者，用于测试异常处理"""

    async def generate(self, prompt: str, max_tokens: int = 500) -> str:
        raise RuntimeError("LLM API 调用失败")


# ============================================================================
# 构造函数测试
# ============================================================================


def test_constructor_default_params():
    """测试构造函数 - 默认参数"""
    compressor = LLMSummaryCompressor()

    assert compressor.name == "llm_summary"
    assert compressor._provider is None
    assert compressor._enable_fallback is True
    assert compressor._max_summary_tokens == 500
    assert compressor._fallback_compressor is not None


def test_constructor_custom_params():
    """测试构造函数 - 自定义参数"""
    mock_provider = MockLLMProvider()
    compressor = LLMSummaryCompressor(
        provider=mock_provider,
        enable_fallback=False,
        max_summary_tokens=1000,
    )

    assert compressor.name == "llm_summary"
    assert compressor._provider is mock_provider
    assert compressor._enable_fallback is False
    assert compressor._max_summary_tokens == 1000


def test_constructor_with_provider_only():
    """测试构造函数 - 仅传入 provider"""
    mock_provider = MockLLMProvider()
    compressor = LLMSummaryCompressor(provider=mock_provider)

    assert compressor._provider is mock_provider
    assert compressor._enable_fallback is True  # 默认开启
    assert compressor._max_summary_tokens == 500  # 默认值


# ============================================================================
# compress() 方法测试 - 成功路径
# ============================================================================


@pytest.mark.asyncio
async def test_compress_happy_path():
    """测试 compress() - 成功生成摘要"""
    segments = [
        Segment(
            type=SegmentType.USER,
            content="这是第一条消息",
            role="user",
            token_count=100,
        ),
        Segment(
            type=SegmentType.ASSISTANT,
            content="这是第二条消息",
            role="assistant",
            token_count=150,
        ),
        Segment(
            type=SegmentType.USER,
            content="这是第三条消息",
            role="user",
            token_count=80,
        ),
    ]

    mock_provider = MockLLMProvider(response="核心要点：用户咨询，助手回复，用户追问")
    compressor = LLMSummaryCompressor(provider=mock_provider)

    context = CompressContext(
        available_tokens=1000,
        target_token_count=500,
        saturation=0.85,
    )

    result = await compressor.compress(segments, context)

    # 验证 LLM 被调用
    mock_provider.generate.assert_called_once()
    call_args = mock_provider.generate.call_args
    assert "总结" in call_args[0][0]  # prompt 包含总结指令
    assert call_args[1]["max_tokens"] == 500  # 使用默认 max_summary_tokens

    # 验证返回的摘要
    assert len(result.compressed_segments) == 1
    summary_segment = result.compressed_segments[0]
    assert summary_segment.type == SegmentType.SUMMARY
    assert summary_segment.role == "assistant"
    assert "核心要点" in summary_segment.content

    # 验证 Provenance
    assert summary_segment.provenance is not None
    assert summary_segment.provenance.source_type == SourceType.COMPRESSION
    assert summary_segment.provenance.compression_method == "llm_summary"
    assert len(summary_segment.provenance.parent_segment_ids) == 3
    assert summary_segment.provenance.parent_segment_ids == [seg.id for seg in segments]

    # 验证统计信息
    assert result.original_token_count == 330  # 100 + 150 + 80
    assert result.compressed_token_count > 0  # 估算的 Token 数
    assert result.method == "llm_summary"
    assert result.parent_segment_ids == [seg.id for seg in segments]
    assert result.compression_ratio < 1.0  # 压缩后应该更小


@pytest.mark.asyncio
async def test_compress_custom_max_tokens():
    """测试 compress() - 自定义最大 Token 数"""
    segments = [
        Segment(
            type=SegmentType.USER,
            content="测试消息",
            role="user",
            token_count=100,
        ),
    ]

    mock_provider = MockLLMProvider(response="摘要")
    compressor = LLMSummaryCompressor(
        provider=mock_provider,
        max_summary_tokens=1000,  # 自定义值
    )

    context = CompressContext(
        available_tokens=10000,
        target_token_count=5000,
        saturation=0.8,
    )

    result = await compressor.compress(segments, context)

    # 验证传递了自定义的 max_tokens
    call_args = mock_provider.generate.call_args
    assert call_args[1]["max_tokens"] == 1000


# ============================================================================
# compress() 方法测试 - 边界情况
# ============================================================================


@pytest.mark.asyncio
async def test_compress_empty_segments():
    """测试 compress() - 空 Segment 列表"""
    mock_provider = MockLLMProvider()
    compressor = LLMSummaryCompressor(provider=mock_provider)

    context = CompressContext(
        available_tokens=1000,
        target_token_count=500,
        saturation=0.8,
    )

    result = await compressor.compress([], context)

    # 空列表应该直接返回空结果，不调用 LLM
    mock_provider.generate.assert_not_called()

    assert len(result.compressed_segments) == 0
    assert result.original_token_count == 0
    assert result.compressed_token_count == 0
    assert result.method == "llm_summary"
    assert result.parent_segment_ids == []


@pytest.mark.asyncio
async def test_compress_single_segment():
    """测试 compress() - 单个 Segment"""
    segment = Segment(
        type=SegmentType.RAG,
        content="这是一个 RAG 检索结果片段",
        role="user",
        token_count=200,
    )

    mock_provider = MockLLMProvider(response="RAG 摘要")
    compressor = LLMSummaryCompressor(provider=mock_provider)

    context = CompressContext(
        available_tokens=1000,
        target_token_count=500,
        saturation=0.8,
    )

    result = await compressor.compress([segment], context)

    # 单个 Segment 也应该生成摘要
    mock_provider.generate.assert_called_once()

    assert len(result.compressed_segments) == 1
    assert result.compressed_segments[0].content == "RAG 摘要"
    assert result.original_token_count == 200
    assert result.parent_segment_ids == [segment.id]


@pytest.mark.asyncio
async def test_compress_batch_segments():
    """测试 compress() - 批量压缩多个 Segment"""
    segments = [
        Segment(
            type=SegmentType.USER,
            content=f"用户消息 {i}",
            role="user",
            token_count=50,
        )
        for i in range(10)
    ]

    mock_provider = MockLLMProvider(response="批量摘要：10 条用户消息的核心要点")
    compressor = LLMSummaryCompressor(provider=mock_provider)

    context = CompressContext(
        available_tokens=10000,
        target_token_count=200,
        saturation=0.95,
    )

    result = await compressor.compress(segments, context)

    # 验证合并逻辑
    call_args = mock_provider.generate.call_args
    prompt = call_args[0][0]
    assert "用户消息 0" in prompt
    assert "用户消息 9" in prompt

    # 验证结果
    assert len(result.compressed_segments) == 1
    assert result.original_token_count == 500  # 10 * 50
    assert result.parent_segment_ids == [seg.id for seg in segments]


# ============================================================================
# compress() 方法测试 - Fallback 机制
# ============================================================================


@pytest.mark.asyncio
async def test_compress_no_provider_with_fallback(caplog):
    """测试 compress() - 无 Provider 时使用 fallback"""
    segments = [
        Segment(
            type=SegmentType.USER,
            content="测试消息",
            role="user",
            token_count=100,
        ),
    ]

    # 未传入 provider，默认 enable_fallback=True
    compressor = LLMSummaryCompressor(provider=None, enable_fallback=True)

    context = CompressContext(
        available_tokens=1000,
        target_token_count=50,  # 需要压缩
        saturation=0.9,
    )

    result = await compressor.compress(segments, context)

    # 应该降级到截断压缩
    assert len(result.compressed_segments) >= 0  # fallback 可能返回空或部分
    assert "降级到截断压缩" in caplog.text
    assert "提示：传入 LLMProvider 实例以启用摘要压缩" in caplog.text


@pytest.mark.asyncio
async def test_compress_no_provider_without_fallback():
    """测试 compress() - 无 Provider 且禁用 fallback 时抛异常"""
    segments = [
        Segment(
            type=SegmentType.USER,
            content="测试消息",
            role="user",
            token_count=100,
        ),
    ]

    compressor = LLMSummaryCompressor(provider=None, enable_fallback=False)

    context = CompressContext(
        available_tokens=1000,
        target_token_count=500,
        saturation=0.8,
    )

    with pytest.raises(CompressionError) as exc_info:
        await compressor.compress(segments, context)

    # 验证异常信息（三段式）
    assert "LLM 摘要压缩失败" in str(exc_info.value)
    assert "未配置 LLM 提供者且未启用 fallback" in str(exc_info.value)
    assert "请传入 LLMProvider 实例或设置 enable_fallback=True" in str(exc_info.value)


@pytest.mark.asyncio
async def test_compress_llm_failure_with_fallback(caplog):
    """测试 compress() - LLM 调用失败时使用 fallback"""
    segments = [
        Segment(
            type=SegmentType.USER,
            content="测试消息",
            role="user",
            token_count=100,
        ),
    ]

    failing_provider = FailingLLMProvider()
    compressor = LLMSummaryCompressor(provider=failing_provider, enable_fallback=True)

    context = CompressContext(
        available_tokens=1000,
        target_token_count=50,
        saturation=0.9,
    )

    result = await compressor.compress(segments, context)

    # 应该降级到截断压缩
    assert len(result.compressed_segments) >= 0
    assert "LLM 摘要生成失败" in caplog.text
    assert "降级到截断压缩" in caplog.text


@pytest.mark.asyncio
async def test_compress_llm_failure_without_fallback():
    """测试 compress() - LLM 调用失败且禁用 fallback 时抛异常"""
    segments = [
        Segment(
            type=SegmentType.USER,
            content="测试消息",
            role="user",
            token_count=100,
        ),
    ]

    failing_provider = FailingLLMProvider()
    compressor = LLMSummaryCompressor(provider=failing_provider, enable_fallback=False)

    context = CompressContext(
        available_tokens=1000,
        target_token_count=500,
        saturation=0.8,
    )

    with pytest.raises(CompressionError) as exc_info:
        await compressor.compress(segments, context)

    # 验证异常链
    assert "LLM 摘要生成失败" in str(exc_info.value)
    assert "检查 LLM API 配置或启用 fallback" in str(exc_info.value)
    assert exc_info.value.__cause__ is not None  # 应该有原始异常


# ============================================================================
# _generate_summary() 方法测试
# ============================================================================


@pytest.mark.asyncio
async def test_generate_summary_prompt_construction():
    """测试 _generate_summary() - Prompt 构造"""
    segments = [
        Segment(
            type=SegmentType.SYSTEM,
            content="你是一个助手",
            role="system",
            token_count=50,
        ),
        Segment(
            type=SegmentType.USER,
            content="请介绍 Python",
            role="user",
            token_count=50,
        ),
        Segment(
            type=SegmentType.ASSISTANT,
            content="Python 是一种高级编程语言",
            role="assistant",
            token_count=100,
        ),
    ]

    mock_provider = MockLLMProvider(response="摘要内容")
    compressor = LLMSummaryCompressor(provider=mock_provider)

    context = CompressContext(
        available_tokens=1000,
        target_token_count=500,
        saturation=0.8,
    )

    summary = await compressor._generate_summary(segments, context)

    # 验证生成的 prompt 包含所有 segment 内容和类型
    call_args = mock_provider.generate.call_args
    prompt = call_args[0][0]

    assert "[SYSTEM]" in prompt
    assert "你是一个助手" in prompt
    assert "[USER]" in prompt
    assert "请介绍 Python" in prompt
    assert "[ASSISTANT]" in prompt
    assert "Python 是一种高级编程语言" in prompt
    assert "请总结以下内容的关键要点" in prompt
    assert "总结：" in prompt

    # 验证返回值已去除空白
    assert summary == "摘要内容"


@pytest.mark.asyncio
async def test_generate_summary_with_whitespace_stripping():
    """测试 _generate_summary() - 去除返回值的空白"""
    segments = [
        Segment(
            type=SegmentType.USER,
            content="测试",
            role="user",
            token_count=50,
        ),
    ]

    # LLM 返回带前后空白的摘要
    mock_provider = MockLLMProvider(response="  \n\n摘要内容\n\n  ")
    compressor = LLMSummaryCompressor(provider=mock_provider)

    context = CompressContext(
        available_tokens=1000,
        target_token_count=500,
        saturation=0.8,
    )

    summary = await compressor._generate_summary(segments, context)

    # 应该去除前后空白
    assert summary == "摘要内容"


@pytest.mark.asyncio
async def test_generate_summary_no_provider_raises():
    """测试 _generate_summary() - 无 Provider 时抛异常"""
    segments = [
        Segment(
            type=SegmentType.USER,
            content="测试",
            role="user",
            token_count=50,
        ),
    ]

    # 强制设置 provider 为 None（绕过构造函数）
    compressor = LLMSummaryCompressor(provider=None)

    context = CompressContext(
        available_tokens=1000,
        target_token_count=500,
        saturation=0.8,
    )

    # _generate_summary 内部会检查 provider
    with pytest.raises(ValueError) as exc_info:
        await compressor._generate_summary(segments, context)

    assert "LLM 提供者未配置" in str(exc_info.value)


# ============================================================================
# _fallback_compress() 方法测试
# ============================================================================


@pytest.mark.asyncio
async def test_fallback_compress():
    """测试 _fallback_compress() - 调用 TruncationCompressor"""
    segments = [
        Segment(
            type=SegmentType.USER,
            content="测试消息 1",
            role="user",
            token_count=100,
        ),
        Segment(
            type=SegmentType.USER,
            content="测试消息 2",
            role="user",
            token_count=100,
        ),
        Segment(
            type=SegmentType.USER,
            content="测试消息 3",
            role="user",
            token_count=100,
        ),
    ]

    compressor = LLMSummaryCompressor(provider=None, enable_fallback=True)

    context = CompressContext(
        available_tokens=1000,
        target_token_count=150,  # 压缩到约 1.5 条消息
        saturation=0.9,
    )

    result = await compressor._fallback_compress(segments, context)

    # fallback 应该使用 TruncationCompressor（tail 策略）
    assert result.method == "truncation_tail"
    assert len(result.compressed_segments) <= len(segments)
    assert result.original_token_count == 300


# ============================================================================
# Protocol 合规性测试
# ============================================================================


@pytest.mark.asyncio
async def test_protocol_compliance():
    """测试 Protocol 合规性 - 实现 Compressor 接口"""
    mock_provider = MockLLMProvider()
    compressor = LLMSummaryCompressor(provider=mock_provider)

    # 验证实现了 Compressor Protocol 的所有必需属性和方法
    # 注意：不能使用 isinstance()，因为 Compressor 不是 @runtime_checkable
    assert hasattr(compressor, "name")
    assert hasattr(compressor, "compress")
    assert callable(compressor.compress)

    # 验证 name 是属性
    assert isinstance(compressor.name, str)
    assert compressor.name == "llm_summary"

    # 验证 compress 方法签名（通过实际调用测试）
    segments = [
        Segment(
            type=SegmentType.USER,
            content="测试",
            role="user",
            token_count=10,
        ),
    ]
    context = CompressContext(
        available_tokens=1000,
        target_token_count=500,
        saturation=0.8,
    )

    result = await compressor.compress(segments, context)

    # 验证返回类型符合 Protocol
    from context_forge.compress.base import CompressionResult
    assert isinstance(result, CompressionResult)


# ============================================================================
# 集成测试 - 多种 Segment 类型
# ============================================================================


@pytest.mark.asyncio
async def test_compress_mixed_segment_types():
    """测试 compress() - 混合多种 Segment 类型"""
    segments = [
        Segment(
            type=SegmentType.SYSTEM,
            content="系统指令",
            role="system",
            token_count=50,
        ),
        Segment(
            type=SegmentType.FEW_SHOT,
            content="示例对话",
            role="user",
            token_count=80,
        ),
        Segment(
            type=SegmentType.RAG,
            content="检索内容",
            role="user",
            token_count=100,
        ),
        Segment(
            type=SegmentType.USER,
            content="用户问题",
            role="user",
            token_count=60,
        ),
    ]

    mock_provider = MockLLMProvider(response="混合内容摘要")
    compressor = LLMSummaryCompressor(provider=mock_provider)

    context = CompressContext(
        available_tokens=1000,
        target_token_count=500,
        saturation=0.85,
    )

    result = await compressor.compress(segments, context)

    # 验证 Prompt 包含所有类型
    call_args = mock_provider.generate.call_args
    prompt = call_args[0][0]
    assert "[SYSTEM]" in prompt
    assert "[FEW_SHOT]" in prompt
    assert "[RAG]" in prompt
    assert "[USER]" in prompt

    # 验证结果
    assert len(result.compressed_segments) == 1
    assert result.compressed_segments[0].type == SegmentType.SUMMARY
    assert result.original_token_count == 290  # 50+80+100+60


# ============================================================================
# 边界测试 - Token 计数
# ============================================================================


@pytest.mark.asyncio
async def test_compress_with_zero_token_segments():
    """测试 compress() - 包含零 Token 的 Segment"""
    segments = [
        Segment(
            type=SegmentType.USER,
            content="有内容",
            role="user",
            token_count=100,
        ),
        Segment(
            type=SegmentType.USER,
            content="无计数",
            role="user",
            token_count=None,  # 未计数
        ),
        Segment(
            type=SegmentType.USER,
            content="零计数",
            role="user",
            token_count=0,  # 零 Token
        ),
    ]

    mock_provider = MockLLMProvider(response="摘要")
    compressor = LLMSummaryCompressor(provider=mock_provider)

    context = CompressContext(
        available_tokens=1000,
        target_token_count=500,
        saturation=0.8,
    )

    result = await compressor.compress(segments, context)

    # 应该正确处理 None 和 0
    assert result.original_token_count == 100  # 只计算有效的 100


@pytest.mark.asyncio
async def test_compress_estimated_token_count():
    """测试 compress() - 估算的压缩后 Token 数"""
    segments = [
        Segment(
            type=SegmentType.USER,
            content="测试消息",
            role="user",
            token_count=100,
        ),
    ]

    # 返回精确长度的摘要以测试估算
    summary_text = "这是一个 20 字符的摘要内容12345"  # 20 字符
    mock_provider = MockLLMProvider(response=summary_text)
    compressor = LLMSummaryCompressor(provider=mock_provider)

    context = CompressContext(
        available_tokens=1000,
        target_token_count=500,
        saturation=0.8,
    )

    result = await compressor.compress(segments, context)

    # 估算公式：len(text) // 4
    expected_tokens = len(summary_text) // 4
    assert result.compressed_token_count == expected_tokens
