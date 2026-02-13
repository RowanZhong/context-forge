"""
冒烟测试 — 验证基础导入和最简用法。

这个测试的存在意义是：确保第一轮交付物（数据模型 + Facade + Pipeline）
可以端到端跑通，不依赖任何外部服务。
"""

import pytest

from context_forge import ContextForge, ContextPackage, Segment, SegmentType, Priority


class TestImport:
    """验证核心类型可以正常导入。"""

    def test_import_context_forge(self) -> None:
        assert ContextForge is not None

    def test_import_segment(self) -> None:
        assert Segment is not None

    def test_import_context_package(self) -> None:
        assert ContextPackage is not None


class TestSegment:
    """Segment 数据模型基础测试。"""

    def test_create_segment(self) -> None:
        seg = Segment(
            type=SegmentType.USER,
            content="你好",
            role="user",
        )
        assert seg.content == "你好"
        assert seg.type == SegmentType.USER
        assert seg.role == "user"
        assert seg.id.startswith("seg_")

    def test_default_priority(self) -> None:
        """SYSTEM 类型自动分配 CRITICAL 优先级。"""
        seg = Segment(
            type=SegmentType.SYSTEM,
            content="系统提示",
            role="system",
        )
        assert seg.effective_priority == Priority.CRITICAL

    def test_immutability(self) -> None:
        """Segment 是不可变的。"""
        seg = Segment(type=SegmentType.USER, content="原始内容", role="user")
        new_seg = seg.with_content("新内容")
        assert seg.content == "原始内容"  # 原对象不变
        assert new_seg.content == "新内容"  # 返回新对象

    def test_with_token_count(self) -> None:
        seg = Segment(type=SegmentType.USER, content="test", role="user")
        counted = seg.with_token_count(42)
        assert counted.token_count == 42
        assert seg.token_count is None  # 原对象不变

    def test_to_message(self) -> None:
        seg = Segment(type=SegmentType.USER, content="你好", role="user")
        msg = seg.to_message()
        assert msg == {"role": "user", "content": "你好"}


class TestContextForge:
    """ContextForge Facade 基础测试。"""

    def test_create_forge(self) -> None:
        forge = ContextForge(model="gpt-4o")
        assert forge.model == "gpt-4o"

    def test_create_with_alias(self) -> None:
        forge = ContextForge(model="sonnet")
        assert "claude" in forge.model

    @pytest.mark.asyncio
    async def test_build_minimal(self) -> None:
        """最简场景：只有 system_prompt 和一条消息。"""
        forge = ContextForge(model="gpt-4o")
        context = await forge.build(
            system_prompt="你是助手。",
            messages=[{"role": "user", "content": "你好"}],
        )
        assert isinstance(context, ContextPackage)
        assert len(context.segments) >= 2  # system + user
        assert context.token_usage.total_tokens > 0

    @pytest.mark.asyncio
    async def test_build_with_rag(self) -> None:
        """带 RAG 片段的组装。"""
        forge = ContextForge(model="gpt-4o")
        context = await forge.build(
            system_prompt="你是客服。",
            messages=[{"role": "user", "content": "退货政策？"}],
            rag_chunks=[
                {"content": "7天内可退货", "score": 0.9},
                {"content": "退款3天到账", "score": 0.8},
            ],
        )
        assert len(context.segments) >= 4  # system + user + 2 rag

    @pytest.mark.asyncio
    async def test_to_messages(self) -> None:
        """验证 to_messages() 输出格式。"""
        forge = ContextForge(model="gpt-4o")
        context = await forge.build(
            system_prompt="你是助手。",
            messages=[{"role": "user", "content": "你好"}],
        )
        messages = context.to_messages()
        assert isinstance(messages, list)
        assert all(isinstance(m, dict) for m in messages)
        assert all("role" in m and "content" in m for m in messages)

    def test_build_sync(self) -> None:
        """同步包装器测试。"""
        forge = ContextForge(model="gpt-4o")
        context = forge.build_sync(
            system_prompt="你是助手。",
            messages=[{"role": "user", "content": "你好"}],
        )
        assert isinstance(context, ContextPackage)
        assert len(context.segments) >= 2

    @pytest.mark.asyncio
    async def test_assembly_duration(self) -> None:
        """组装耗时应该很快（不含 LLM 调用）。"""
        forge = ContextForge(model="gpt-4o")
        context = await forge.build(
            system_prompt="你是助手。",
            messages=[{"role": "user", "content": "你好"}],
        )
        assert context.assembly_duration_ms < 5000  # 宽松阈值，CI 环境可能较慢

    @pytest.mark.asyncio
    async def test_budget_allocation_recorded(self) -> None:
        """预算分配记录应被填充。"""
        forge = ContextForge(model="gpt-4o")
        context = await forge.build(
            system_prompt="你是助手。",
            messages=[{"role": "user", "content": "你好"}],
        )
        assert context.budget_allocation is not None
        assert context.budget_allocation.total_budget > 0


class TestErrorHandling:
    """错误处理测试。"""

    def test_unknown_model(self) -> None:
        """未知模型应抛出 ModelNotFoundError。"""
        from context_forge.errors import ModelNotFoundError
        with pytest.raises(ModelNotFoundError) as exc_info:
            ContextForge(model="nonexistent-model-xyz")
        assert "nonexistent-model-xyz" in str(exc_info.value)
        assert "修复建议" in str(exc_info.value)
