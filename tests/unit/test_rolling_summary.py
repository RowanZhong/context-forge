"""
RollingSummaryCompressor 单元测试。

测试滚动摘要压缩器的核心功能：
- 初始摘要生成
- 增量摘要更新
- 轮次感知拆分（keep_recent_turns）
- 状态管理（reset / has_state）
- Fallback 降级
- 无 provider 行为
- 空输入处理
"""

from __future__ import annotations

import pytest

from context_forge.compress.base import CompressContext, CompressionResult
from context_forge.compress.summary import RollingSummaryCompressor
from context_forge.models.metadata import SegmentMetadata
from context_forge.models.provenance import SourceType
from context_forge.models.segment import Segment, SegmentType


# ---------------------------------------------------------------------------
# Mock LLM Provider
# ---------------------------------------------------------------------------

class MockLLMProvider:
    """记录调用的 Mock LLM Provider。"""

    def __init__(self, response: str = "摘要内容"):
        self.response = response
        self.call_count = 0
        self.last_prompt: str | None = None

    async def generate(self, prompt: str, max_tokens: int = 500) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        return self.response


class FailingLLMProvider:
    """始终抛出异常的 Mock LLM Provider。"""

    async def generate(self, prompt: str, max_tokens: int = 500) -> str:
        raise RuntimeError("LLM 调用失败")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_context(
    available: int = 10000, target: int = 5000, saturation: float = 0.8
) -> CompressContext:
    return CompressContext(
        available_tokens=available,
        target_token_count=target,
        saturation=saturation,
    )


def _make_segment(
    content: str,
    turn: int | None = None,
    seg_type: SegmentType = SegmentType.USER,
    token_count: int = 100,
) -> Segment:
    metadata = SegmentMetadata(turn_number=turn) if turn is not None else None
    return Segment(
        type=seg_type,
        content=content,
        role="user" if seg_type == SegmentType.USER else "assistant",
        token_count=token_count,
        metadata=metadata,
    )


def _make_turn_segments(turn: int, user_msg: str, asst_msg: str) -> list[Segment]:
    """生成一轮对话（user + assistant）。"""
    return [
        _make_segment(user_msg, turn=turn, seg_type=SegmentType.USER),
        _make_segment(asst_msg, turn=turn, seg_type=SegmentType.ASSISTANT),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRollingSummaryCompressorInit:
    """初始化与属性测试。"""

    def test_name(self):
        compressor = RollingSummaryCompressor()
        assert compressor.name == "rolling_summary"

    def test_initial_state(self):
        compressor = RollingSummaryCompressor()
        assert compressor.has_state is False
        assert compressor.previous_summary is None

    def test_keep_recent_turns_clamped(self):
        compressor = RollingSummaryCompressor(keep_recent_turns=-5)
        assert compressor._keep_recent_turns == 0


class TestRollingSummaryEmpty:
    """空输入测试。"""

    @pytest.mark.asyncio
    async def test_empty_segments(self):
        compressor = RollingSummaryCompressor()
        result = await compressor.compress([], _make_context())
        assert result.compressed_segments == []
        assert result.original_token_count == 0
        assert result.method == "rolling_summary"


class TestRollingSummaryInitialSummary:
    """初始摘要生成测试（无 _previous_summary）。"""

    @pytest.mark.asyncio
    async def test_initial_summary_generated(self):
        """第一次压缩应生成初始摘要。"""
        provider = MockLLMProvider(response="初始摘要：用户想去日本旅行")
        compressor = RollingSummaryCompressor(
            provider=provider, keep_recent_turns=1
        )

        segments = (
            _make_turn_segments(1, "我想去日本", "好的") +
            _make_turn_segments(2, "预算2万", "没问题") +
            _make_turn_segments(3, "最近的问题", "最近的回答")
        )

        result = await compressor.compress(segments, _make_context())

        assert provider.call_count == 1
        assert compressor.has_state is True
        assert compressor.previous_summary == "初始摘要：用户想去日本旅行"
        # 最近 1 轮保留原文 + 1 条摘要
        assert len(result.compressed_segments) == 3  # 1 summary + 2 recent
        assert result.compressed_segments[0].type == SegmentType.SUMMARY
        assert result.metadata["rolling_state"] == "initial"
        assert result.metadata["older_count"] == 4
        assert result.metadata["recent_count"] == 2

    @pytest.mark.asyncio
    async def test_initial_prompt_no_previous(self):
        """初始摘要的 Prompt 不应包含'上一轮摘要'。"""
        provider = MockLLMProvider(response="摘要")
        compressor = RollingSummaryCompressor(
            provider=provider, keep_recent_turns=1
        )

        segments = (
            _make_turn_segments(1, "消息1", "回复1") +
            _make_turn_segments(2, "消息2", "回复2")
        )

        await compressor.compress(segments, _make_context())

        assert "上一轮摘要" not in provider.last_prompt
        assert "总结" in provider.last_prompt


class TestRollingSummaryIncremental:
    """增量更新测试（有 _previous_summary）。"""

    @pytest.mark.asyncio
    async def test_incremental_update(self):
        """第二次压缩应使用增量 Prompt（包含上一轮摘要）。"""
        provider = MockLLMProvider(response="更新后的摘要")
        compressor = RollingSummaryCompressor(
            provider=provider, keep_recent_turns=1
        )

        # 第一次压缩
        segments_r1 = (
            _make_turn_segments(1, "你好", "你好") +
            _make_turn_segments(2, "问题1", "回答1")
        )
        provider.response = "初始摘要"
        await compressor.compress(segments_r1, _make_context())
        assert compressor.previous_summary == "初始摘要"

        # 第二次压缩（新轮次）
        segments_r2 = (
            _make_turn_segments(1, "你好", "你好") +
            _make_turn_segments(2, "问题1", "回答1") +
            _make_turn_segments(3, "新问题", "新回答")
        )
        provider.response = "更新后的摘要：包含新旧内容"
        result = await compressor.compress(segments_r2, _make_context())

        assert provider.call_count == 2
        assert "上一轮摘要" in provider.last_prompt
        assert "初始摘要" in provider.last_prompt
        assert compressor.previous_summary == "更新后的摘要：包含新旧内容"
        assert result.metadata["rolling_state"] == "incremental"

    @pytest.mark.asyncio
    async def test_state_accumulates(self):
        """连续多次调用应持续累积状态。"""
        provider = MockLLMProvider()
        compressor = RollingSummaryCompressor(
            provider=provider, keep_recent_turns=1
        )

        for round_num in range(1, 4):
            segments = []
            for t in range(1, round_num + 2):
                segments.extend(_make_turn_segments(t, f"用户消息{t}", f"助手回复{t}"))

            provider.response = f"第{round_num}轮摘要"
            await compressor.compress(segments, _make_context())

        assert compressor.previous_summary == "第3轮摘要"
        assert provider.call_count == 3


class TestRollingSummaryKeepRecentTurns:
    """轮次感知拆分测试。"""

    @pytest.mark.asyncio
    async def test_keep_recent_turns_2(self):
        """keep_recent_turns=2 应保留最近 2 轮原文。"""
        provider = MockLLMProvider(response="摘要")
        compressor = RollingSummaryCompressor(
            provider=provider, keep_recent_turns=2
        )

        segments = []
        for t in range(1, 6):
            segments.extend(_make_turn_segments(t, f"用户{t}", f"助手{t}"))

        result = await compressor.compress(segments, _make_context())

        # 5 轮，保留最近 2 轮 = 4 条 + 1 条摘要 = 5
        assert len(result.compressed_segments) == 5
        assert result.compressed_segments[0].type == SegmentType.SUMMARY
        assert result.metadata["recent_count"] == 4

    @pytest.mark.asyncio
    async def test_all_turns_within_range(self):
        """当所有轮次都在保留范围内时，不应触发摘要。"""
        provider = MockLLMProvider()
        compressor = RollingSummaryCompressor(
            provider=provider, keep_recent_turns=5
        )

        segments = []
        for t in range(1, 4):
            segments.extend(_make_turn_segments(t, f"用户{t}", f"助手{t}"))

        result = await compressor.compress(segments, _make_context())

        assert provider.call_count == 0
        assert result.metadata["rolling_state"] == "no_older_turns"
        assert len(result.compressed_segments) == 6  # 全部保留

    @pytest.mark.asyncio
    async def test_keep_zero_turns(self):
        """keep_recent_turns=0 应将所有消息都纳入摘要。"""
        provider = MockLLMProvider(response="全部摘要")
        compressor = RollingSummaryCompressor(
            provider=provider, keep_recent_turns=0
        )

        segments = _make_turn_segments(1, "用户1", "助手1")
        result = await compressor.compress(segments, _make_context())

        # 全部被摘要，只剩 1 条摘要
        assert len(result.compressed_segments) == 1
        assert result.compressed_segments[0].type == SegmentType.SUMMARY

    @pytest.mark.asyncio
    async def test_fallback_to_position_without_turn_number(self):
        """没有 turn_number 时应按列表位置推断轮次。"""
        provider = MockLLMProvider(response="摘要")
        compressor = RollingSummaryCompressor(
            provider=provider, keep_recent_turns=1
        )

        # 6 个 Segment，无 turn_number
        segments = [
            _make_segment(f"消息{i}", turn=None, token_count=50)
            for i in range(6)
        ]

        result = await compressor.compress(segments, _make_context())

        # keep_recent_turns=1 → 保留末尾 2 条 + 1 条摘要
        assert len(result.compressed_segments) == 3


class TestRollingSummaryReset:
    """状态重置测试。"""

    @pytest.mark.asyncio
    async def test_reset_clears_state(self):
        """reset() 应清除所有累积状态。"""
        provider = MockLLMProvider(response="摘要")
        compressor = RollingSummaryCompressor(
            provider=provider, keep_recent_turns=1
        )

        segments = (
            _make_turn_segments(1, "旧", "旧") +
            _make_turn_segments(2, "新", "新")
        )
        await compressor.compress(segments, _make_context())
        assert compressor.has_state is True

        compressor.reset()

        assert compressor.has_state is False
        assert compressor.previous_summary is None

    @pytest.mark.asyncio
    async def test_compress_after_reset_is_initial(self):
        """reset 后再次压缩应回到 initial 状态。"""
        provider = MockLLMProvider(response="摘要")
        compressor = RollingSummaryCompressor(
            provider=provider, keep_recent_turns=1
        )

        segments = (
            _make_turn_segments(1, "旧", "旧") +
            _make_turn_segments(2, "新", "新")
        )

        await compressor.compress(segments, _make_context())
        compressor.reset()

        provider.response = "重新开始的摘要"
        result = await compressor.compress(segments, _make_context())

        # reset 后 Prompt 不应包含上一轮摘要
        assert "上一轮摘要" not in provider.last_prompt
        assert result.metadata["rolling_state"] == "initial"


class TestRollingSummaryFallback:
    """Fallback 降级测试。"""

    @pytest.mark.asyncio
    async def test_no_provider_with_fallback(self):
        """无 provider 且 enable_fallback=True 时应降级到截断。"""
        compressor = RollingSummaryCompressor(
            provider=None, enable_fallback=True, keep_recent_turns=1
        )

        segments = (
            _make_turn_segments(1, "旧消息", "旧回复") +
            _make_turn_segments(2, "新消息", "新回复")
        )

        result = await compressor.compress(segments, _make_context())

        # 降级到截断压缩，方法名应为 truncation
        assert "truncation" in result.method
        assert compressor.has_state is False

    @pytest.mark.asyncio
    async def test_no_provider_without_fallback(self):
        """无 provider 且 enable_fallback=False 时应抛出 CompressionError。"""
        from context_forge.errors.exceptions import CompressionError

        compressor = RollingSummaryCompressor(
            provider=None, enable_fallback=False, keep_recent_turns=1
        )

        segments = (
            _make_turn_segments(1, "旧消息", "旧回复") +
            _make_turn_segments(2, "新消息", "新回复")
        )

        with pytest.raises(CompressionError):
            await compressor.compress(segments, _make_context())

    @pytest.mark.asyncio
    async def test_llm_failure_with_fallback(self):
        """LLM 调用失败且 enable_fallback=True 时应降级到截断。"""
        compressor = RollingSummaryCompressor(
            provider=FailingLLMProvider(),
            enable_fallback=True,
            keep_recent_turns=1,
        )

        segments = (
            _make_turn_segments(1, "旧消息", "旧回复") +
            _make_turn_segments(2, "新消息", "新回复")
        )

        result = await compressor.compress(segments, _make_context())
        assert "truncation" in result.method

    @pytest.mark.asyncio
    async def test_llm_failure_without_fallback(self):
        """LLM 调用失败且 enable_fallback=False 时应抛出 CompressionError。"""
        from context_forge.errors.exceptions import CompressionError

        compressor = RollingSummaryCompressor(
            provider=FailingLLMProvider(),
            enable_fallback=False,
            keep_recent_turns=1,
        )

        segments = (
            _make_turn_segments(1, "旧消息", "旧回复") +
            _make_turn_segments(2, "新消息", "新回复")
        )

        with pytest.raises(CompressionError):
            await compressor.compress(segments, _make_context())


class TestRollingSummaryCompressionResult:
    """CompressionResult 数据完整性测试。"""

    @pytest.mark.asyncio
    async def test_result_fields(self):
        """验证 CompressionResult 字段完整性。"""
        provider = MockLLMProvider(response="摘要内容")
        compressor = RollingSummaryCompressor(
            provider=provider, keep_recent_turns=1
        )

        segments = (
            _make_turn_segments(1, "旧消息", "旧回复") +
            _make_turn_segments(2, "新消息", "新回复")
        )

        result = await compressor.compress(segments, _make_context())

        assert result.method == "rolling_summary"
        assert result.original_token_count == 400  # 4 * 100
        assert len(result.parent_segment_ids) == 4
        assert result.compressed_segments[0].provenance.compression_method == "rolling_summary"
        assert result.compressed_segments[0].provenance.source_type == SourceType.COMPRESSION

    @pytest.mark.asyncio
    async def test_provenance_parent_ids_only_older(self):
        """摘要 Segment 的 parent_segment_ids 应只包含旧轮次 Segment。"""
        provider = MockLLMProvider(response="摘要")
        compressor = RollingSummaryCompressor(
            provider=provider, keep_recent_turns=1
        )

        old_segs = _make_turn_segments(1, "旧", "旧")
        new_segs = _make_turn_segments(2, "新", "新")
        segments = old_segs + new_segs

        result = await compressor.compress(segments, _make_context())

        summary_seg = result.compressed_segments[0]
        old_ids = {seg.id for seg in old_segs}
        assert set(summary_seg.provenance.parent_segment_ids) == old_ids
