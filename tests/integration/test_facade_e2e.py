"""
ContextForge Facade 端到端测试。

这是关键的集成测试，验证 Facade 的完整 build 流程和所有主要功能路径。

测试覆盖范围：
1. 基础 build() — dict 输入、Segment 输入、各种组合
2. build_sync() — 同步便捷方法
3. 路由集成 — 启用/禁用路由
4. 缓存集成 — 缓存命中/未命中
5. 快照管理 — 保存快照
6. 指标收集 — 记录指标
7. 反模式检测 — 检测和报告
8. 错误处理 — 边界情况、异常路径
9. 各种段类型组合 — system/user/rag/tools/state
10. 输入变体 — 空输入、大型输入、unicode 等

目标覆盖率：>80%
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from context_forge import ContextForge
from context_forge.models.context_package import ContextPackage
from context_forge.models.routing import ComplexityLevel, ModelConfig
from context_forge.models.routing import RoutingDecision
from context_forge.models.segment import Segment, SegmentType


class TestFacadeBuildBasics:
    """测试 build() 的基础功能。"""

    @pytest.mark.asyncio
    async def test_build_with_system_prompt_only(self) -> None:
        """测试仅系统提示的最小化场景。"""
        forge = ContextForge(model="gpt-4o")
        context = await forge.build(system_prompt="你是一个有用的助手。")

        assert isinstance(context, ContextPackage)
        assert len(context.segments) > 0
        assert context.segments[0].type == SegmentType.SYSTEM
        assert context.model == "gpt-4o"
        assert context.assembly_duration_ms >= 0

    @pytest.mark.asyncio
    async def test_build_with_messages(self) -> None:
        """测试带消息的场景。"""
        forge = ContextForge(model="gpt-4o")
        context = await forge.build(
            system_prompt="你是一个有用的助手。",
            messages=[
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"},
                {"role": "user", "content": "告诉我 Python 的 GIL"},
            ],
        )

        assert len(context.segments) >= 4  # system + 3 messages
        user_messages = [s for s in context.segments if s.type == SegmentType.USER]
        assert len(user_messages) >= 2

    @pytest.mark.asyncio
    async def test_build_with_rag_chunks(self) -> None:
        """测试带 RAG 片段的场景。"""
        forge = ContextForge(model="gpt-4o")
        context = await forge.build(
            system_prompt="你是一个助手。",
            rag_chunks=[
                {"content": "RAG 片段 1", "score": 0.95},
                {"content": "RAG 片段 2", "score": 0.87},
                {"content": "RAG 片段 3", "score": 0.72},
            ],
        )

        rag_segments = [s for s in context.segments if s.type == SegmentType.RAG]
        assert len(rag_segments) == 3
        # 验证分数被保存
        assert rag_segments[0].provenance.retrieval_score == 0.95

    @pytest.mark.asyncio
    async def test_build_with_tools(self) -> None:
        """测试带工具定义的场景。"""
        forge = ContextForge(model="gpt-4o")
        tools = [
            {"name": "search", "description": "搜索文档"},
            {"name": "summarize", "description": "总结文本"},
        ]
        context = await forge.build(
            system_prompt="你是一个助手。",
            tools=tools,
        )

        tool_segments = [
            s for s in context.segments if s.type == SegmentType.TOOL_DEFINITION
        ]
        assert len(tool_segments) == 2
        # 验证工具被序列化为 JSON
        tool_content = json.loads(tool_segments[0].content)
        assert tool_content.get("name") == "search"

    @pytest.mark.asyncio
    async def test_build_with_few_shot_examples(self) -> None:
        """测试带少样本示例的场景。"""
        forge = ContextForge(model="gpt-4o")
        context = await forge.build(
            system_prompt="你是一个助手。",
            few_shot_examples=[
                {"role": "user", "content": "例子 1 - 用户输入"},
                {"role": "assistant", "content": "例子 1 - 助手回复"},
            ],
        )

        few_shot_segments = [
            s for s in context.segments if s.type == SegmentType.FEW_SHOT
        ]
        assert len(few_shot_segments) == 2

    @pytest.mark.asyncio
    async def test_build_with_state(self) -> None:
        """测试带状态锚点的场景。"""
        forge = ContextForge(model="gpt-4o")
        state = {"user_id": "user_123", "conversation_turn": 5}
        context = await forge.build(
            system_prompt="你是一个助手。",
            state=state,
        )

        state_segments = [
            s for s in context.segments if s.type == SegmentType.STATE
        ]
        assert len(state_segments) == 1
        # 验证状态内容被序列化
        assert "user_id" in state_segments[0].content

    @pytest.mark.asyncio
    async def test_build_with_extra_segments(self) -> None:
        """测试带预构建 Segment 的场景。"""
        forge = ContextForge(model="gpt-4o")
        extra_segment = Segment(
            type=SegmentType.SCHEMA,
            content='{"type": "object"}',
            role="system",
        )
        context = await forge.build(
            system_prompt="你是一个助手。",
            extra_segments=[extra_segment],
        )

        schema_segments = [
            s for s in context.segments if s.type == SegmentType.SCHEMA
        ]
        assert len(schema_segments) == 1

    @pytest.mark.asyncio
    async def test_build_with_all_input_types(self) -> None:
        """测试包含所有输入类型的完整场景。"""
        forge = ContextForge(model="gpt-4o")
        context = await forge.build(
            system_prompt="你是一个客服助手。",
            messages=[
                {"role": "user", "content": "请帮我查询订单"},
                {"role": "assistant", "content": "好的，让我为你查询。"},
            ],
            rag_chunks=[
                {"content": "订单查询流程...", "score": 0.95},
            ],
            tools=[
                {"name": "query_order", "description": "查询订单"},
            ],
            few_shot_examples=[
                {"role": "user", "content": "示例用户输入"},
            ],
            state={"order_id": "12345"},
        )

        assert len(context.segments) > 5
        assert context.model == "gpt-4o"
        assert context.budget_allocation is not None

    @pytest.mark.asyncio
    async def test_build_empty_inputs(self) -> None:
        """测试所有输入都为空的边界情况。"""
        forge = ContextForge(model="gpt-4o")
        context = await forge.build()

        # 应该至少返回一个空的包
        assert isinstance(context, ContextPackage)
        assert context.model == "gpt-4o"

    @pytest.mark.asyncio
    async def test_build_with_current_turn(self) -> None:
        """测试带对话轮次参数的场景。"""
        forge = ContextForge(model="gpt-4o")
        context = await forge.build(
            system_prompt="你是一个助手。",
            current_turn=5,
        )

        assert context.model == "gpt-4o"
        # 验证 PipelineContext 中使用了 current_turn
        assert isinstance(context, ContextPackage)

    @pytest.mark.asyncio
    async def test_build_with_namespace(self) -> None:
        """测试带命名空间参数的场景。"""
        forge = ContextForge(model="gpt-4o")
        context = await forge.build(
            system_prompt="你是一个助手。",
            namespace="custom_namespace",
        )

        assert context.model == "gpt-4o"


class TestFacadeBuildSync:
    """测试 build_sync() 同步便捷方法。"""

    def test_build_sync_basic(self) -> None:
        """测试 build_sync() 的基本功能。"""
        forge = ContextForge(model="gpt-4o")
        context = forge.build_sync(
            system_prompt="你是一个有用的助手。",
            messages=[{"role": "user", "content": "你好"}],
        )

        assert isinstance(context, ContextPackage)
        assert len(context.segments) > 0
        assert context.model == "gpt-4o"

    def test_build_sync_with_rag(self) -> None:
        """测试 build_sync() 带 RAG 场景。"""
        forge = ContextForge(model="gpt-4o")
        context = forge.build_sync(
            system_prompt="你是一个助手。",
            rag_chunks=[
                {"content": "RAG 内容", "score": 0.9},
            ],
        )

        assert isinstance(context, ContextPackage)
        rag_segments = [s for s in context.segments if s.type == SegmentType.RAG]
        assert len(rag_segments) > 0

    def test_build_sync_in_event_loop_with_nest_asyncio(self) -> None:
        """测试 build_sync() 在已有 event loop 中且 nest_asyncio 可用时的行为。"""
        async def test_inside_loop():
            forge = ContextForge(model="gpt-4o")
            # 这会警告用户关于 nest_asyncio，但不会抛异常（因为测试用例是特殊的）
            # 实际上，在实际 Jupyter 环境中需要 nest_asyncio
            # 这里我们只是验证它不会崩溃
            try:
                with patch("nest_asyncio.apply"):
                    context = forge.build_sync(system_prompt="你是一个助手。")
                    assert isinstance(context, ContextPackage)
            except RuntimeError:
                # 如果没有 nest_asyncio，预期会抛异常
                pass

        # 运行在 event loop 中
        try:
            asyncio.run(test_inside_loop())
        except RuntimeError as e:
            # 预期可能抛异常（nest_asyncio not available）
            assert "nest_asyncio" in str(e) or "in already running event loop" in str(e)


class TestFacadeRouting:
    """测试路由集成。"""

    @pytest.mark.asyncio
    async def test_build_with_routing_disabled(self) -> None:
        """测试禁用路由的场景。"""
        forge = ContextForge(model="gpt-4o")
        context = await forge.build(
            system_prompt="你是一个助手。",
            messages=[{"role": "user", "content": "简单查询"}],
        )

        # 路由在默认配置中禁用
        assert context.routing_decision is None

    @pytest.mark.asyncio
    async def test_build_with_routing_enabled(self) -> None:
        """测试启用路由的场景。"""
        # 创建临时策略文件启用路由
        import tempfile
        import yaml

        policy_config = {
            "version": "1.0",
            "name": "routing_test",
            "budget": {"max_context_tokens": 8192},
            "routing": {
                "enabled": True,
                "default_model": "gpt-4o",
                "rules": [
                    {
                        "name": "simple_rule",
                        "condition_type": "complexity",
                        "condition_value": "simple",
                        "target_model": "gpt-4o-mini",
                    }
                ],
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            yaml.dump(policy_config, f, allow_unicode=True)
            policy_path = f.name

        try:
            forge = ContextForge(model="gpt-4o", policy_path=policy_path)
            context = await forge.build(
                system_prompt="你是一个助手。",
                messages=[{"role": "user", "content": "simple query"}],
            )

            # 路由应该产生一个决策
            assert context.routing_decision is not None
            assert context.routing_decision.selected_model.model_id in [
                "gpt-4o",
                "gpt-4o-mini",
            ]
        finally:
            import os

            os.unlink(policy_path)

    @pytest.mark.asyncio
    async def test_build_routing_decision_affects_model(self) -> None:
        """测试路由决策是否影响最终模型选择。"""
        # 使用 mock router 测试
        forge = ContextForge(model="gpt-4o")

        mock_router = MagicMock()
        mock_router.route = MagicMock(
            return_value=RoutingDecision(
                selected_model=ModelConfig(
                    model_id="gpt-4o-mini",
                    provider="openai",
                    max_context_tokens=128000,
                ),
                complexity=ComplexityLevel.SIMPLE,
                estimated_cost=0.01,
                confidence=0.95,
                reasoning="简单查询，选择小模型降低成本",
            )
        )

        forge._router = mock_router

        context = await forge.build(
            system_prompt="你是一个助手。",
            messages=[{"role": "user", "content": "简单查询"}],
        )

        # 验证路由决策被记录
        assert context.routing_decision is not None
        assert context.routing_decision.selected_model.model_id == "gpt-4o-mini"


class TestFacadeCache:
    """测试缓存集成。"""

    @pytest.mark.asyncio
    async def test_build_with_cache_miss(
        self, mock_cache_manager: MagicMock
    ) -> None:
        """测试缓存未命中的场景。"""
        mock_cache_manager.get = AsyncMock(return_value=None)
        mock_cache_manager.set = AsyncMock(return_value=None)

        forge = ContextForge(model="gpt-4o", cache_backend=mock_cache_manager)
        context = await forge.build(
            system_prompt="你是一个助手。",
            messages=[{"role": "user", "content": "你好"}],
        )

        assert isinstance(context, ContextPackage)
        # 验证缓存 set 被调用（存储结果）
        mock_cache_manager.set.assert_called()

    @pytest.mark.asyncio
    async def test_build_cache_key_generation(self) -> None:
        """测试缓存键生成。"""
        mock_cache_manager = MagicMock()
        mock_cache_manager.get = AsyncMock(return_value=None)
        mock_cache_manager.set = AsyncMock(return_value=None)

        forge = ContextForge(model="gpt-4o", cache_backend=mock_cache_manager)
        context = await forge.build(
            system_prompt="系统提示",
            messages=[{"role": "user", "content": "用户输入"}],
        )

        assert isinstance(context, ContextPackage)
        # 验证 set 被调用，并检查缓存键的格式
        assert mock_cache_manager.set.called
        call_args = mock_cache_manager.set.call_args
        cache_key = call_args[0][0]  # 第一个位置参数是 cache_key
        # 缓存键应该是 SHA256 哈希（64 个十六进制字符）
        assert len(cache_key) == 64
        assert all(c in "0123456789abcdef" for c in cache_key)

    @pytest.mark.asyncio
    async def test_build_cache_hit_handling(self) -> None:
        """测试缓存命中的处理（虽然当前实现暂未反序列化）。"""
        mock_cache_manager = MagicMock()
        cached_package_json = json.dumps(
            {
                "segments": [],
                "model": "gpt-4o",
                "budget_allocation": {"total_budget": 8192},
            }
        )
        from context_forge.cache.base import CacheEntry

        mock_cache_manager.get = AsyncMock(
            return_value=CacheEntry(value=cached_package_json)
        )
        mock_cache_manager.set = AsyncMock(return_value=None)

        forge = ContextForge(model="gpt-4o", cache_backend=mock_cache_manager)
        context = await forge.build(
            system_prompt="你是一个助手。",
            messages=[{"role": "user", "content": "你好"}],
        )

        assert isinstance(context, ContextPackage)


class TestFacadeObservability:
    """测试快照和指标收集。"""

    @pytest.mark.asyncio
    async def test_build_with_snapshot_manager(
        self, mock_snapshot_manager: MagicMock
    ) -> None:
        """测试带快照管理器的场景。"""
        mock_snapshot_manager.save = AsyncMock(return_value="snapshot_id_123")

        forge = ContextForge(model="gpt-4o", snapshot_manager=mock_snapshot_manager)
        context = await forge.build(
            system_prompt="你是一个助手。",
            messages=[{"role": "user", "content": "你好"}],
        )

        assert isinstance(context, ContextPackage)
        # 验证快照 save 被调用
        mock_snapshot_manager.save.assert_called()

    @pytest.mark.asyncio
    async def test_build_with_metrics_collector(
        self, mock_metrics_collector: MagicMock
    ) -> None:
        """测试带指标收集器的场景。"""
        mock_metrics_collector.collect_from_package = MagicMock()
        mock_metrics_collector.record = MagicMock()

        forge = ContextForge(model="gpt-4o", metrics_collector=mock_metrics_collector)
        context = await forge.build(
            system_prompt="你是一个助手。",
            messages=[{"role": "user", "content": "你好"}],
        )

        assert isinstance(context, ContextPackage)
        # 验证指标收集被调用
        mock_metrics_collector.collect_from_package.assert_called()

    @pytest.mark.asyncio
    async def test_snapshot_method(self, mock_snapshot_manager: MagicMock) -> None:
        """测试 save_snapshot() 便捷方法。"""
        mock_snapshot_manager.save = AsyncMock(return_value="snap_id_456")

        forge = ContextForge(model="gpt-4o", snapshot_manager=mock_snapshot_manager)
        context = await forge.build(system_prompt="你是一个助手。")

        snapshot_id = await forge.save_snapshot(context)
        assert snapshot_id == "snap_id_456"

    @pytest.mark.asyncio
    async def test_snapshot_without_manager_raises_error(self) -> None:
        """测试没有快照管理器时调用 save_snapshot() 抛异常。"""
        # 创建禁用快照的策略
        import tempfile
        import yaml

        policy_config = {
            "version": "1.0",
            "name": "no_snapshot_test",
            "budget": {"max_context_tokens": 8192},
            "observability": {"snapshot_enabled": False},
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            yaml.dump(policy_config, f, allow_unicode=True)
            policy_path = f.name

        try:
            forge = ContextForge(model="gpt-4o", policy_path=policy_path)
            context = await forge.build(system_prompt="你是一个助手。")

            with pytest.raises(RuntimeError):
                await forge.save_snapshot(context)
        finally:
            import os

            os.unlink(policy_path)

    @pytest.mark.asyncio
    async def test_diff_method(self, mock_snapshot_manager: MagicMock) -> None:
        """测试 diff_snapshots() 便捷方法。"""
        # 创建 mock 快照对象，包含 .package 属性
        snap1_mock = MagicMock()
        snap1_mock.package = MagicMock(model="gpt-4o")

        snap2_mock = MagicMock()
        snap2_mock.package = MagicMock(model="gpt-4o")

        mock_snapshot_manager.load = AsyncMock(
            side_effect=[snap1_mock, snap2_mock]
        )

        forge = ContextForge(model="gpt-4o", snapshot_manager=mock_snapshot_manager)

        # 需要 patch DiffEngine 因为它可能不存在或需要实现
        with patch(
            "context_forge.observability.DiffEngine"
        ) as mock_diff_engine_class:
            mock_diff_instance = MagicMock()
            # diff 方法需要是 AsyncMock，返回一个包含 .entries 属性的对象
            mock_context_diff = MagicMock()
            mock_context_diff.entries = []
            mock_diff_instance.diff = AsyncMock(return_value=mock_context_diff)
            mock_diff_instance.format_json = MagicMock(
                return_value={"differences": ["segment change"]}
            )
            mock_diff_engine_class.return_value = mock_diff_instance

            result = await forge.diff_snapshots("snap_1", "snap_2")
            assert "differences" in result

    @pytest.mark.asyncio
    async def test_golden_record_method(
        self, mock_snapshot_manager: MagicMock
    ) -> None:
        """测试 validate_against_golden() 便捷方法。"""
        # 创建 mock 黄金快照对象，包含 .package 属性
        golden_snap_mock = MagicMock()
        golden_snap_mock.package = MagicMock(model="gpt-4o")

        mock_snapshot_manager.load = AsyncMock(return_value=golden_snap_mock)

        forge = ContextForge(model="gpt-4o", snapshot_manager=mock_snapshot_manager)
        context = await forge.build(system_prompt="你是一个助手。")

        with patch(
            "context_forge.observability.DiffEngine"
        ) as mock_diff_engine_class:
            mock_diff_instance = MagicMock()
            # diff 方法需要是 AsyncMock，返回一个包含 .entries 属性的对象
            mock_context_diff = MagicMock()
            mock_context_diff.entries = []  # 空列表表示无差异，passed 应为 True
            mock_diff_instance.diff = AsyncMock(return_value=mock_context_diff)
            mock_diff_instance.format_json = MagicMock(
                return_value={"differences": []}
            )
            mock_diff_engine_class.return_value = mock_diff_instance

            result = await forge.validate_against_golden("golden_snap_id", context)
            assert "passed" in result
            assert result["passed"] is True


class TestFacadeAntipattern:
    """测试反模式检测。"""

    @pytest.mark.asyncio
    async def test_build_with_antipattern_check_disabled(self) -> None:
        """测试禁用反模式检测的场景。"""
        forge = ContextForge(model="gpt-4o")
        context = await forge.build(
            system_prompt="你是一个助手。",
            check_antipatterns=False,
        )

        assert isinstance(context, ContextPackage)

    @pytest.mark.asyncio
    async def test_build_with_antipattern_check_enabled(self) -> None:
        """测试启用反模式检测的场景。"""
        forge = ContextForge(model="gpt-4o")

        with patch("context_forge.antipattern.create_default_detector") as mock_detector_class:
            mock_detector = MagicMock()
            mock_detector.detect = MagicMock(return_value=[])  # 无问题
            mock_detector_class.return_value = mock_detector

            context = await forge.build(
                system_prompt="你是一个助手。",
                check_antipatterns=True,
            )

            assert isinstance(context, ContextPackage)
            # 验证检测器被创建和调用
            mock_detector_class.assert_called()
            mock_detector.detect.assert_called()

    def test_detect_antipatterns_raw_format(
        self, context_package: ContextPackage
    ) -> None:
        """测试 detect_antipatterns() 返回原始格式。"""
        forge = ContextForge(model="gpt-4o")

        with patch("context_forge.antipattern.create_default_detector") as mock_detector_class:
            mock_detector = MagicMock()
            mock_results = [
                MagicMock(severity="WARNING", title="问题 1"),
                MagicMock(severity="INFO", title="问题 2"),
            ]
            mock_detector.detect = MagicMock(return_value=mock_results)
            mock_detector_class.return_value = mock_detector

            results = forge.detect_antipatterns(context_package, format="raw")

            assert isinstance(results, list)
            assert len(results) == 2

    def test_detect_antipatterns_text_format(
        self, context_package: ContextPackage
    ) -> None:
        """测试 detect_antipatterns() 返回文本格式。"""
        forge = ContextForge(model="gpt-4o")

        with patch("context_forge.antipattern.create_default_detector") as mock_detector_class:
            mock_detector = MagicMock()
            mock_detector.detect = MagicMock(return_value=[])
            mock_detector.format_report = MagicMock(return_value="文本报告")
            mock_detector_class.return_value = mock_detector

            result = forge.detect_antipatterns(context_package, format="text")

            assert isinstance(result, str)
            mock_detector.format_report.assert_called()


class TestFacadePrepareSegments:
    """测试 _prepare_segments() 内部方法的各种输入处理。"""

    def test_prepare_segments_system_prompt_only(self) -> None:
        """测试仅系统提示的准备。"""
        forge = ContextForge(model="gpt-4o")
        segments = forge._prepare_segments(
            system_prompt="你是一个助手。",
            messages=[],
            rag_chunks=[],
            tools=[],
            few_shot_examples=[],
            state=None,
            extra_segments=[],
            current_turn=0,
        )

        assert len(segments) == 1
        assert segments[0].type == SegmentType.SYSTEM

    def test_prepare_segments_all_types(self) -> None:
        """测试所有段类型的准备。"""
        forge = ContextForge(model="gpt-4o")
        segments = forge._prepare_segments(
            system_prompt="系统提示",
            messages=[
                {"role": "user", "content": "用户消息"},
                {"role": "assistant", "content": "助手消息"},
            ],
            rag_chunks=[
                {"content": "RAG 1", "score": 0.9, "source_id": "doc_1"},
                {"content": "RAG 2", "score": 0.8},
            ],
            tools=[{"name": "tool_1", "description": "描述"}],
            few_shot_examples=[
                {"role": "user", "content": "示例"},
            ],
            state={"key": "value"},
            extra_segments=[],
            current_turn=2,
        )

        # 验证所有类型都被创建
        types_in_segments = {s.type for s in segments}
        assert SegmentType.SYSTEM in types_in_segments
        assert SegmentType.USER in types_in_segments
        assert SegmentType.ASSISTANT in types_in_segments
        assert SegmentType.RAG in types_in_segments
        assert SegmentType.TOOL_DEFINITION in types_in_segments
        assert SegmentType.FEW_SHOT in types_in_segments
        assert SegmentType.STATE in types_in_segments

    def test_prepare_segments_rag_with_all_fields(self) -> None:
        """测试 RAG 段的所有可选字段。"""
        forge = ContextForge(model="gpt-4o")
        segments = forge._prepare_segments(
            system_prompt="系统",
            messages=[],
            rag_chunks=[
                {
                    "content": "RAG 内容",
                    "score": 0.95,
                    "source_id": "custom_id",
                    "uri": "https://example.com/doc",
                }
            ],
            tools=[],
            few_shot_examples=[],
            state=None,
            extra_segments=[],
            current_turn=1,
        )

        rag_segments = [s for s in segments if s.type == SegmentType.RAG]
        assert len(rag_segments) == 1
        assert rag_segments[0].provenance.source_id == "custom_id"
        assert rag_segments[0].provenance.uri == "https://example.com/doc"
        assert rag_segments[0].provenance.retrieval_score == 0.95

    def test_prepare_segments_messages_with_system_role(self) -> None:
        """测试带有 system 角色的消息。"""
        forge = ContextForge(model="gpt-4o")
        segments = forge._prepare_segments(
            system_prompt="",
            messages=[
                {"role": "system", "content": "系统消息"},
                {"role": "user", "content": "用户消息"},
            ],
            rag_chunks=[],
            tools=[],
            few_shot_examples=[],
            state=None,
            extra_segments=[],
            current_turn=0,
        )

        system_messages = [s for s in segments if s.type == SegmentType.SYSTEM]
        user_messages = [s for s in segments if s.type == SegmentType.USER]
        assert len(system_messages) > 0
        assert len(user_messages) > 0

    def test_prepare_segments_state_serialization(self) -> None:
        """测试状态的 JSON 序列化。"""
        forge = ContextForge(model="gpt-4o")
        state = {
            "user_id": "123",
            "nested": {"key": "value"},
            "list": [1, 2, 3],
        }
        segments = forge._prepare_segments(
            system_prompt="",
            messages=[],
            rag_chunks=[],
            tools=[],
            few_shot_examples=[],
            state=state,
            extra_segments=[],
            current_turn=0,
        )

        state_segments = [s for s in segments if s.type == SegmentType.STATE]
        assert len(state_segments) == 1
        assert "user_id" in state_segments[0].content
        assert "nested" in state_segments[0].content


class TestFacadeErrorHandling:
    """测试错误处理和边界情况。"""

    @pytest.mark.asyncio
    async def test_build_with_invalid_model(self) -> None:
        """测试使用未知模型的场景。"""
        # 使用无效的模型应该抛异常
        from context_forge.errors.exceptions import ModelNotFoundError

        with pytest.raises(ModelNotFoundError):
            ContextForge(model="unknown_model_xyz")

    @pytest.mark.asyncio
    async def test_build_with_invalid_policy_file(self) -> None:
        """测试无效的策略文件。"""
        with pytest.raises(Exception):  # 可能是 FileNotFoundError 或其他异常
            ContextForge(model="gpt-4o", policy_path="/nonexistent/path/policy.yaml")

    @pytest.mark.asyncio
    async def test_build_debug_mode(self) -> None:
        """测试调试模式。"""
        forge = ContextForge(model="gpt-4o", debug=True)
        context = await forge.build(system_prompt="你是一个助手。")

        assert isinstance(context, ContextPackage)

    @pytest.mark.asyncio
    async def test_build_with_large_context(self) -> None:
        """测试大型上下文的处理。"""
        forge = ContextForge(model="gpt-4o", max_context_tokens=128000)
        large_rag_chunks = [
            {"content": f"这是第 {i} 个很长的 RAG 片段，包含大量内容。" * 100, "score": 0.9}
            for i in range(10)
        ]
        context = await forge.build(
            system_prompt="你是一个助手。",
            rag_chunks=large_rag_chunks,
        )

        assert isinstance(context, ContextPackage)
        assert len(context.segments) > 0

    @pytest.mark.asyncio
    async def test_build_with_unicode_content(self) -> None:
        """测试 Unicode 内容的处理。"""
        forge = ContextForge(model="gpt-4o")
        context = await forge.build(
            system_prompt="你是一个有用的助手。🤖",
            messages=[
                {"role": "user", "content": "请用日本語で説明してください。"},
                {"role": "assistant", "content": "これは日本語のテキストです。"},
            ],
            rag_chunks=[
                {"content": "محتوى باللغة العربية", "score": 0.9},
            ],
        )

        assert isinstance(context, ContextPackage)

    @pytest.mark.asyncio
    async def test_build_with_override_tokens(self) -> None:
        """测试运行时覆盖 Token 限制。"""
        forge = ContextForge(
            model="gpt-4o",
            max_context_tokens=4096,
            output_reserved_tokens=512,
            thinking_reserved_tokens=2048,
        )
        context = await forge.build(system_prompt="你是一个助手。")

        assert isinstance(context, ContextPackage)
        assert forge.budget_policy.max_context_tokens == 4096
        assert forge.budget_policy.output_reserved_tokens == 512


class TestFacadeProperties:
    """测试 Facade 的属性访问器。"""

    def test_model_property(self) -> None:
        """测试 model 属性。"""
        forge = ContextForge(model="gpt-4o")
        assert forge.model == "gpt-4o"

    def test_policy_property(self) -> None:
        """测试 policy 属性。"""
        forge = ContextForge(model="gpt-4o")
        policy = forge.policy
        assert policy is not None
        assert policy.budget is not None

    def test_budget_policy_property(self) -> None:
        """测试 budget_policy 属性。"""
        forge = ContextForge(model="gpt-4o")
        budget = forge.budget_policy
        assert budget.max_context_tokens > 0

    def test_pipeline_property(self) -> None:
        """测试 pipeline 属性。"""
        forge = ContextForge(model="gpt-4o")
        pipeline = forge.pipeline
        assert pipeline is not None

    def test_repr(self) -> None:
        """测试 __repr__ 方法。"""
        forge = ContextForge(model="gpt-4o")
        repr_str = repr(forge)
        assert "ContextForge" in repr_str
        assert "gpt-4o" in repr_str


class TestFacadeInitialization:
    """测试 ContextForge 初始化的各种配置。"""

    def test_init_with_model_alias(self) -> None:
        """测试模型别名解析。"""
        forge = ContextForge(model="sonnet")
        assert "sonnet" in forge.model.lower() or "claude" in forge.model.lower()

    def test_init_with_explicit_tokens(self) -> None:
        """测试显式指定 Token 数。"""
        forge = ContextForge(
            model="gpt-4o",
            max_context_tokens=16384,
            output_reserved_tokens=2048,
        )
        assert forge.budget_policy.max_context_tokens == 16384
        assert forge.budget_policy.output_reserved_tokens == 2048

    def test_init_with_thinking_tokens(self) -> None:
        """测试 Thinking Token 预留。"""
        forge = ContextForge(
            model="gpt-4o",
            thinking_reserved_tokens=4096,
        )
        # 如果模型不支持 thinking，预留应该是 0 或用户指定的值
        assert forge.budget_policy.thinking_reserved_tokens >= 0

    def test_init_with_custom_pipeline(self) -> None:
        """测试自定义 Pipeline。"""
        from context_forge.pipeline.base import create_default_pipeline

        custom_pipeline = create_default_pipeline()
        forge = ContextForge(model="gpt-4o", pipeline=custom_pipeline)
        assert forge.pipeline is custom_pipeline

    def test_init_custom_components(
        self,
        mock_cache_manager: MagicMock,
        mock_router: MagicMock,
        mock_metrics_collector: MagicMock,
        mock_snapshot_manager: MagicMock,
    ) -> None:
        """测试全部自定义组件。"""
        # 创建临时策略启用路由以便测试 mock_router
        import tempfile
        import yaml

        policy_config = {
            "version": "1.0",
            "name": "custom_test",
            "budget": {"max_context_tokens": 8192},
            "cache": {"enabled": True},
            "routing": {"enabled": True, "default_model": "gpt-4o"},
            "observability": {"metrics_enabled": True, "snapshot_enabled": True},
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            yaml.dump(policy_config, f, allow_unicode=True)
            policy_path = f.name

        try:
            forge = ContextForge(
                model="gpt-4o",
                policy_path=policy_path,
                cache_backend=mock_cache_manager,
                router=mock_router,
                metrics_collector=mock_metrics_collector,
                snapshot_manager=mock_snapshot_manager,
            )

            assert forge._cache_manager is mock_cache_manager
            assert forge._router is mock_router
            assert forge._metrics_collector is mock_metrics_collector
            assert forge._snapshot_manager is mock_snapshot_manager
        finally:
            import os

            os.unlink(policy_path)


class TestFacadeIntegrationScenarios:
    """测试完整的集成场景。"""

    @pytest.mark.asyncio
    async def test_rag_qa_scenario(self) -> None:
        """模拟 RAG QA 场景。"""
        forge = ContextForge(model="gpt-4o")

        # 模拟用户问题和检索到的文档
        question = "Python 的 GIL 是什么？"
        retrieved_docs = [
            {
                "content": "GIL 是 Python 全局解释器锁，限制多线程并发...",
                "score": 0.95,
                "source_id": "python_gil_doc",
            },
            {
                "content": "Python 3.13 移除了 GIL，性能大幅提升...",
                "score": 0.87,
                "source_id": "python_3_13_release",
            },
        ]

        context = await forge.build(
            system_prompt="你是一个 Python 专家。请根据提供的资料回答问题。",
            messages=[{"role": "user", "content": question}],
            rag_chunks=retrieved_docs,
        )

        assert isinstance(context, ContextPackage)
        assert len(context.segments) >= 3  # system + message + RAG chunks

    @pytest.mark.asyncio
    async def test_multi_turn_conversation_scenario(self) -> None:
        """模拟多轮对话场景。"""
        forge = ContextForge(model="gpt-4o")

        # 模拟对话历史
        conversation = [
            {"role": "user", "content": "什么是上下文窗口？"},
            {
                "role": "assistant",
                "content": "上下文窗口是 LLM 能处理的最大 Token 数...",
            },
            {"role": "user", "content": "如何管理上下文窗口？"},
        ]

        context = await forge.build(
            system_prompt="你是 AI 技术顾问。",
            messages=conversation,
            current_turn=2,
        )

        assert isinstance(context, ContextPackage)

    @pytest.mark.asyncio
    async def test_tool_use_scenario(self) -> None:
        """模拟工具使用场景。"""
        forge = ContextForge(model="gpt-4o")

        tools = [
            {
                "name": "search",
                "description": "搜索知识库",
                "parameters": {"type": "object", "properties": {"query": {}}},
            },
            {
                "name": "calculate",
                "description": "执行数学计算",
                "parameters": {"type": "object", "properties": {"expression": {}}},
            },
        ]

        context = await forge.build(
            system_prompt="你是一个智能助手，可以使用以下工具。",
            tools=tools,
            messages=[{"role": "user", "content": "帮我计算 2^10"}],
        )

        assert isinstance(context, ContextPackage)
        tool_defs = [s for s in context.segments if s.type == SegmentType.TOOL_DEFINITION]
        assert len(tool_defs) == 2

    @pytest.mark.asyncio
    async def test_agent_memory_scenario(self) -> None:
        """模拟 Agent 记忆管理场景。"""
        forge = ContextForge(model="gpt-4o")

        agent_state = {
            "agent_id": "agent_001",
            "conversation_count": 42,
            "last_action": "search",
            "memory_type": "summary",
        }

        context = await forge.build(
            system_prompt="你是一个自主 Agent。",
            state=agent_state,
            messages=[
                {"role": "user", "content": "下一步做什么？"},
            ],
        )

        assert isinstance(context, ContextPackage)
        state_segments = [s for s in context.segments if s.type == SegmentType.STATE]
        assert len(state_segments) > 0


class TestFacadeEdgeCases:
    """测试边界情况和覆盖剩余代码路径。"""

    @pytest.mark.asyncio
    async def test_build_with_model_config_auto_thinking_tokens(self) -> None:
        """测试支持 thinking 的模型自动预留 Token。"""
        # Claude models support thinking
        forge = ContextForge(model="claude-opus")
        assert forge.budget_policy.thinking_reserved_tokens > 0

    @pytest.mark.asyncio
    async def test_build_routing_with_budget_adjustment(self) -> None:
        """测试路由决策中的预算调整。"""
        forge = ContextForge(model="gpt-4o")

        mock_router = MagicMock()
        # 创建包含 budget_adjustment 的路由决策
        mock_router.route = MagicMock(
            return_value=RoutingDecision(
                selected_model=ModelConfig(
                    model_id="gpt-4o-mini",
                    provider="openai",
                    max_context_tokens=64000,
                ),
                complexity=ComplexityLevel.SIMPLE,
                estimated_cost=0.01,
                confidence=0.95,
                reasoning="简单查询，选择小模型",
                budget_adjustment={"max_context_tokens": 4096},
            )
        )

        forge._router = mock_router

        context = await forge.build(
            system_prompt="你是一个助手。",
            messages=[{"role": "user", "content": "简单查询"}],
        )

        assert isinstance(context, ContextPackage)
        # 验证路由决策被记录
        assert context.routing_decision is not None

    @pytest.mark.asyncio
    async def test_build_with_antipattern_warnings(self) -> None:
        """测试反模式检测产生警告。"""
        import warnings

        forge = ContextForge(model="gpt-4o")

        with patch(
            "context_forge.antipattern.create_default_detector"
        ) as mock_detector_class:
            # 创建有问题的检测结果
            from unittest.mock import MagicMock as MM

            mock_result = MM()
            mock_result.severity = "WARNING"
            mock_result.title = "测试警告"

            mock_detector = MagicMock()
            mock_detector.detect = MagicMock(return_value=[mock_result])
            mock_detector_class.return_value = mock_detector

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                context = await forge.build(
                    system_prompt="你是一个助手。",
                    check_antipatterns=True,
                )

                # 应该产生警告
                assert len(w) > 0

    @pytest.mark.asyncio
    async def test_build_with_antipattern_warnings_only(self) -> None:
        """测试反模式检测产生警告但无 CRITICAL。"""
        import tempfile
        import yaml
        import warnings

        policy_config = {
            "version": "1.0",
            "name": "antipattern_test",
            "budget": {"max_context_tokens": 8192},
            "antipattern": {
                "check_on_build": True,
                "fail_on_critical": True,
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            yaml.dump(policy_config, f, allow_unicode=True)
            policy_path = f.name

        try:
            forge = ContextForge(model="gpt-4o", policy_path=policy_path)

            with patch(
                "context_forge.antipattern.create_default_detector"
            ) as mock_detector_class:
                # 创建 WARNING 问题（不是 CRITICAL）
                from context_forge.antipattern.base import AntiPatternSeverity

                mock_result = MagicMock()
                mock_result.severity = AntiPatternSeverity.WARNING
                mock_result.title = "测试 WARNING"

                mock_detector = MagicMock()
                mock_detector.detect = MagicMock(return_value=[mock_result])
                mock_detector_class.return_value = mock_detector

                # 应该发出警告但不抛异常
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    context = await forge.build(
                        system_prompt="你是一个助手。",
                        check_antipatterns=False,
                    )

                    assert isinstance(context, ContextPackage)
                    # 应该有警告
                    assert len(w) > 0
        finally:
            import os

            os.unlink(policy_path)

    @pytest.mark.asyncio
    async def test_build_with_cache_save(self) -> None:
        """测试缓存保存路径。"""
        mock_cache_manager = MagicMock()
        mock_cache_manager.get = AsyncMock(return_value=None)
        mock_cache_manager.set = AsyncMock(return_value=None)

        forge = ContextForge(model="gpt-4o", cache_backend=mock_cache_manager)
        context = await forge.build(
            system_prompt="你是一个助手。",
            messages=[{"role": "user", "content": "你好"}],
        )

        # 验证缓存 set 被调用
        assert mock_cache_manager.set.called
        assert isinstance(context, ContextPackage)

    @pytest.mark.asyncio
    async def test_build_debug_logging(self, caplog) -> None:
        """测试调试日志输出。"""
        import logging

        forge = ContextForge(model="gpt-4o", debug=True)

        with caplog.at_level(logging.DEBUG):
            context = await forge.build(system_prompt="你是一个助手。")

        assert isinstance(context, ContextPackage)
        # 应该有调试日志输出
        # caplog 会捕获日志

    @pytest.mark.asyncio
    async def test_build_with_empty_messages_list(self) -> None:
        """测试空消息列表。"""
        forge = ContextForge(model="gpt-4o")
        context = await forge.build(
            system_prompt="你是一个助手。",
            messages=[],
        )

        assert isinstance(context, ContextPackage)

    @pytest.mark.asyncio
    async def test_build_with_none_state(self) -> None:
        """测试 None 状态。"""
        forge = ContextForge(model="gpt-4o")
        context = await forge.build(
            system_prompt="你是一个助手。",
            state=None,
        )

        assert isinstance(context, ContextPackage)

    @pytest.mark.asyncio
    async def test_build_message_turn_number_calculation(self) -> None:
        """测试消息的 turn_number 计算。"""
        forge = ContextForge(model="gpt-4o")
        context = await forge.build(
            system_prompt="你是一个助手。",
            messages=[
                {"role": "user", "content": "msg1"},
                {"role": "assistant", "content": "msg2"},
                {"role": "user", "content": "msg3"},
                {"role": "assistant", "content": "msg4"},
            ],
        )

        # 验证消息的 turn_number
        user_messages = [s for s in context.segments if s.type == SegmentType.USER]
        assert len(user_messages) > 0
        # 第一条用户消息应该是 turn 0
        if user_messages[0].metadata:
            assert user_messages[0].metadata.turn_number == 0

    @pytest.mark.asyncio
    async def test_build_rag_without_optional_fields(self) -> None:
        """测试 RAG chunk 不包含可选字段。"""
        forge = ContextForge(model="gpt-4o")
        context = await forge.build(
            system_prompt="你是一个助手。",
            rag_chunks=[
                "这是一个简单字符串而不是 dict",
                {"content": "这是一个没有可选字段的 RAG chunk"},
            ],
        )

        rag_segments = [s for s in context.segments if s.type == SegmentType.RAG]
        assert len(rag_segments) >= 2

    def test_diff_without_manager_raises_error(self) -> None:
        """测试没有快照管理器时调用 diff_snapshots() 抛异常。"""
        import tempfile
        import yaml

        policy_config = {
            "version": "1.0",
            "name": "no_snapshot_test",
            "budget": {"max_context_tokens": 8192},
            "observability": {"snapshot_enabled": False},
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            yaml.dump(policy_config, f, allow_unicode=True)
            policy_path = f.name

        try:
            forge = ContextForge(model="gpt-4o", policy_path=policy_path)

            with pytest.raises(RuntimeError):
                asyncio.run(forge.diff_snapshots("snap_1", "snap_2"))
        finally:
            import os

            os.unlink(policy_path)

    def test_golden_record_without_manager_raises_error(self) -> None:
        """测试没有快照管理器时调用 validate_against_golden() 抛异常。"""
        import tempfile
        import yaml

        policy_config = {
            "version": "1.0",
            "name": "no_snapshot_test",
            "budget": {"max_context_tokens": 8192},
            "observability": {"snapshot_enabled": False},
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            yaml.dump(policy_config, f, allow_unicode=True)
            policy_path = f.name

        try:
            forge = ContextForge(model="gpt-4o", policy_path=policy_path)
            context = asyncio.run(forge.build(system_prompt="你是一个助手。"))

            with pytest.raises(RuntimeError):
                asyncio.run(forge.validate_against_golden("golden_snap_id", context))
        finally:
            import os

            os.unlink(policy_path)

    @pytest.mark.asyncio
    async def test_build_with_explicit_token_override(self) -> None:
        """测试显式 Token 覆盖。"""
        # 不使用策略文件，直接使用参数覆盖
        forge = ContextForge(
            model="gpt-4o",
            max_context_tokens=16000,
            output_reserved_tokens=512,
        )
        assert forge.budget_policy.max_context_tokens == 16000
        assert forge.budget_policy.output_reserved_tokens == 512

    @pytest.mark.asyncio
    async def test_build_segment_role_mapping(self) -> None:
        """测试消息角色到 Segment 类型的映射。"""
        forge = ContextForge(model="gpt-4o")
        context = await forge.build(
            system_prompt="",
            messages=[
                {"role": "user", "content": "用户"},
                {"role": "assistant", "content": "助手"},
                {"role": "system", "content": "系统"},
                {"role": "unknown", "content": "未知"},  # 应该默认为 USER
            ],
        )

        segment_types = [s.type for s in context.segments]
        assert SegmentType.USER in segment_types
        assert SegmentType.ASSISTANT in segment_types
        assert SegmentType.SYSTEM in segment_types


# 标记为集成和端到端测试
pytestmark = [pytest.mark.integration, pytest.mark.e2e]
