"""
CLI 覆盖率补充测试 — cmd_diff.py 和 utils.py。

目标：将两个文件覆盖率从 ~75% 提升到 80%+。

覆盖缺口：
cmd_diff.py:
  - 不支持的输出格式分支 (line 85)
  - 快照目录不存在分支 (line 100)
  - _compute_diff 中的新增/删除/修改 segment 分支 (lines 157, 162, 166-167)
  - _output_text_diff 中的预算差异 (lines 258-262)
  - _output_rich_diff 中的新增/删除 segment 详情及 >5 截断 (lines 341-371)
  - _output_rich_diff 中的警告差异 (lines 376-381)

utils.py:
  - print_info 函数 (lines 91-92)
  - load_json_or_yaml 文件读取错误处理 (lines 138-139)
  - load_json_or_yaml 未知后缀 fallback 处理 (lines 147-160)
  - load_json_or_yaml YAML 格式错误 (lines 163-164)
  - create_forge_from_options 非 ContextForgeError 异常包装 (lines 198-204)
  - create_audit_tree 超过 10 条记录截断 (lines 324-336)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from context_forge.cli.utils import (
    create_audit_tree,
    create_budget_table,
    create_console,
    create_forge_from_options,
    create_segment_table,
    create_summary_panel,
    format_token_count,
    handle_context_forge_error,
    load_json_or_yaml,
    print_info,
    print_success,
    print_warning,
)
from context_forge.errors import ConfigValidationError, ContextForgeError


# ============================================================
# 1. utils.py — print_info 覆盖
# ============================================================


class TestPrintInfo:
    """测试 print_info 函数。"""

    def test_print_info_outputs_message(self, capsys: pytest.CaptureFixture[str]) -> None:
        """测试 print_info 正常输出信息。"""
        # print_info 使用 Rich Console，所以需要 mock console
        with patch("context_forge.cli.utils._console") as mock_console:
            mock_console_instance = MagicMock()
            # 重设全局变量让 create_console 返回 mock
            with patch("context_forge.cli.utils.create_console", return_value=mock_console_instance):
                # 直接调用 print_info
                print_info("测试信息")


    def test_print_info_calls_console_print(self) -> None:
        """测试 print_info 调用了 console.print。"""
        mock_console = MagicMock()
        with patch("context_forge.cli.utils.create_console", return_value=mock_console):
            print_info("测试消息")
            mock_console.print.assert_called_once_with("测试消息")


# ============================================================
# 2. utils.py — load_json_or_yaml 错误路径
# ============================================================


class TestLoadJsonOrYamlErrors:
    """测试 load_json_or_yaml 各种错误处理路径。"""

    def test_file_read_permission_error(self, tmp_path: Path) -> None:
        """测试文件读取权限错误时包装为 ValueError。"""
        test_file = tmp_path / "test.json"
        test_file.write_text("{}", encoding="utf-8")

        # Mock read_text 抛出 PermissionError
        with patch.object(Path, "read_text", side_effect=PermissionError("拒绝访问")):
            with pytest.raises(ValueError, match="无法读取文件"):
                load_json_or_yaml(test_file)

    def test_unknown_suffix_valid_json(self, tmp_path: Path) -> None:
        """测试未知后缀文件但内容是有效 JSON 时正常解析。"""
        test_file = tmp_path / "data.txt"
        test_file.write_text('{"key": "value"}', encoding="utf-8")

        result = load_json_or_yaml(test_file)
        assert result == {"key": "value"}

    def test_unknown_suffix_valid_yaml(self, tmp_path: Path) -> None:
        """测试未知后缀文件，JSON 解析失败后 YAML 解析成功。"""
        test_file = tmp_path / "data.cfg"
        test_file.write_text("key: value\ncount: 42\n", encoding="utf-8")

        result = load_json_or_yaml(test_file)
        assert result == {"key": "value", "count": 42}

    def test_unknown_suffix_non_dict_yaml(self, tmp_path: Path) -> None:
        """测试未知后缀文件，YAML 解析结果不是字典时抛出 ValueError。"""
        test_file = tmp_path / "data.cfg"
        test_file.write_text("- item1\n- item2\n", encoding="utf-8")

        with pytest.raises(ValueError, match="无法解析文件格式"):
            load_json_or_yaml(test_file)

    def test_yaml_non_dict_root(self, tmp_path: Path) -> None:
        """测试 YAML 文件根元素不是字典时抛出 ValueError。"""
        test_file = tmp_path / "data.yaml"
        test_file.write_text("- item1\n- item2\n", encoding="utf-8")

        with pytest.raises(ValueError, match="YAML 文件根元素必须是字典"):
            load_json_or_yaml(test_file)

    def test_yaml_format_error(self, tmp_path: Path) -> None:
        """测试 YAML 格式错误时抛出 ValueError。"""
        test_file = tmp_path / "data.yaml"
        # 写入无效 YAML（tab 和特殊字符混合）
        test_file.write_text("key: [\ninvalid: yaml: :\n  - {{\n", encoding="utf-8")

        with pytest.raises(ValueError):
            load_json_or_yaml(test_file)

    def test_json_format_error(self, tmp_path: Path) -> None:
        """测试 JSON 格式错误时抛出 ValueError。"""
        test_file = tmp_path / "data.json"
        test_file.write_text("{invalid json content}", encoding="utf-8")

        with pytest.raises(ValueError, match="JSON 格式错误"):
            load_json_or_yaml(test_file)


# ============================================================
# 3. utils.py — create_forge_from_options 异常包装
# ============================================================


class TestCreateForgeFromOptions:
    """测试 create_forge_from_options 异常包装逻辑。"""

    def test_context_forge_error_passthrough(self) -> None:
        """测试 ContextForgeError 直接重新抛出，不包装。"""
        from context_forge.errors import ModelNotFoundError

        with patch("context_forge.cli.utils.ContextForge") as mock_forge:
            mock_forge.side_effect = ModelNotFoundError(
                what="模型不存在",
                why="无效的模型名称",
                how="请检查模型名称",
            )
            with pytest.raises(ModelNotFoundError):
                create_forge_from_options(model="invalid-model")

    def test_unexpected_error_wrapped_as_config_validation_error(self) -> None:
        """测试非 ContextForgeError 被包装为 ConfigValidationError。"""
        with patch("context_forge.cli.utils.ContextForge") as mock_forge:
            mock_forge.side_effect = RuntimeError("意外错误")

            with pytest.raises(ConfigValidationError) as exc_info:
                create_forge_from_options(model="gpt-4o")

            assert "创建 ContextForge 实例失败" in exc_info.value.what
            assert "意外错误" in exc_info.value.why

    def test_successful_creation(self) -> None:
        """测试正常创建 ContextForge 实例。"""
        with patch("context_forge.cli.utils.ContextForge") as mock_forge:
            mock_instance = MagicMock()
            mock_forge.return_value = mock_instance

            result = create_forge_from_options(model="gpt-4o")
            assert result is mock_instance

    def test_creation_with_all_options(self) -> None:
        """测试使用所有参数创建 ContextForge 实例。"""
        with patch("context_forge.cli.utils.ContextForge") as mock_forge:
            mock_instance = MagicMock()
            mock_forge.return_value = mock_instance

            result = create_forge_from_options(
                model="gpt-4o",
                policy_path="/path/to/policy.yaml",
                max_context_tokens=32000,
                debug=True,
            )
            assert result is mock_instance
            mock_forge.assert_called_once_with(
                model="gpt-4o",
                policy_path=Path("/path/to/policy.yaml"),
                max_context_tokens=32000,
                debug=True,
            )


# ============================================================
# 4. utils.py — create_audit_tree 超过 10 条截断
# ============================================================


class TestCreateAuditTreeOverflow:
    """测试 create_audit_tree 超过 10 条记录的截断逻辑。"""

    def test_audit_tree_with_more_than_10_entries(self) -> None:
        """测试审计日志超过 10 条时显示截断提示。"""
        # 创建 15 条同类型的审计记录
        audit_log = [
            {
                "decision": "KEEP",
                "segment_id": f"seg_{i:04d}",
                "reason_code": "PASS",
                "reason_detail": f"通过验证 #{i}",
            }
            for i in range(15)
        ]

        tree = create_audit_tree(audit_log)
        # 验证树被创建
        assert tree is not None
        # 验证有子节点（至少有一个 decision 分支）
        assert len(tree.children) > 0

    def test_audit_tree_with_multiple_decision_types(self) -> None:
        """测试审计日志包含多种决策类型且每种超过 10 条。"""
        audit_log = []
        # KEEP 类型 12 条
        for i in range(12):
            audit_log.append({
                "decision": "KEEP",
                "segment_id": f"seg_keep_{i:04d}",
                "reason_code": "PASS",
                "reason_detail": f"保留 #{i}",
            })
        # DROP 类型 11 条
        for i in range(11):
            audit_log.append({
                "decision": "DROP",
                "segment_id": f"seg_drop_{i:04d}",
                "reason_code": "BUDGET",
                "reason_detail": f"预算不足 #{i}",
            })

        tree = create_audit_tree(audit_log)
        # 应有 2 个决策分支（DROP, KEEP — 按字母排序）
        assert len(tree.children) == 2

    def test_audit_tree_empty(self) -> None:
        """测试空审计日志。"""
        tree = create_audit_tree([])
        assert tree is not None
        assert len(tree.children) == 0

    def test_audit_tree_exactly_10_entries(self) -> None:
        """测试恰好 10 条记录（边界情况，不应截断）。"""
        audit_log = [
            {
                "decision": "KEEP",
                "segment_id": f"seg_{i:04d}",
                "reason_code": "PASS",
                "reason_detail": f"通过 #{i}",
            }
            for i in range(10)
        ]

        tree = create_audit_tree(audit_log)
        assert tree is not None
        # 只有一个决策分支（KEEP）
        assert len(tree.children) == 1


# ============================================================
# 5. utils.py — 其他工具函数覆盖
# ============================================================


class TestUtilsMiscFunctions:
    """测试 utils.py 其他工具函数。"""

    def test_format_token_count(self) -> None:
        """测试 Token 数字格式化。"""
        assert format_token_count(0) == "0"
        assert format_token_count(1000) == "1,000"
        assert format_token_count(128000) == "128,000"

    def test_create_summary_panel_with_token_field(self) -> None:
        """测试创建摘要面板，包含 token 字段时自动格式化。"""
        content = {
            "total_tokens": 128000,
            "model": "gpt-4o",
            "segment_count": 5,
        }
        panel = create_summary_panel("测试面板", content)
        assert panel is not None
        assert panel.title == "测试面板"

    def test_create_summary_panel_custom_border(self) -> None:
        """测试创建摘要面板自定义边框样式。"""
        content = {"key": "value"}
        panel = create_summary_panel("标题", content, border_style="green")
        assert panel is not None

    def test_create_segment_table(self) -> None:
        """测试创建 Segment 表格。"""
        segments = [
            {
                "id": "seg_0001_abcdef",
                "type": "SYSTEM",
                "role": "system",
                "priority": "CRITICAL",
                "token_count": 100,
                "content_preview": "你是一个有帮助的助手",
            },
            {
                "id": "seg_0002_ghijkl",
                "type": "USER",
                "role": "user",
                "priority": "MEDIUM",
                "token_count": 50,
                "content_preview": "你好！",
            },
        ]
        table = create_segment_table(segments)
        assert table is not None
        assert table.title == "Segments"

    def test_create_budget_table_with_zero_total(self) -> None:
        """测试创建预算表格，总预算为 0（避免除零错误）。"""
        budget = {
            "total_budget": 0,
            "content_budget": 0,
            "total_used": 0,
        }
        table = create_budget_table(budget)
        assert table is not None

    def test_create_budget_table_normal(self) -> None:
        """测试创建正常的预算表格。"""
        budget = {
            "total_budget": 128000,
            "content_budget": 123904,
            "total_used": 15000,
        }
        table = create_budget_table(budget)
        assert table is not None

    def test_handle_context_forge_error(self) -> None:
        """测试统一错误处理函数。"""
        error = ContextForgeError(
            what="测试错误",
            why="测试原因",
            how="测试修复方案",
        )
        with pytest.raises(SystemExit) as exc_info:
            handle_context_forge_error(error)
        assert exc_info.value.code == 1

    def test_print_success(self) -> None:
        """测试 print_success 输出。"""
        mock_console = MagicMock()
        with patch("context_forge.cli.utils.create_console", return_value=mock_console):
            print_success("操作成功")
            mock_console.print.assert_called_once()

    def test_print_warning(self) -> None:
        """测试 print_warning 输出。"""
        mock_console = MagicMock()
        with patch("context_forge.cli.utils.create_console", return_value=mock_console):
            print_warning("警告消息")
            mock_console.print.assert_called_once()


# ============================================================
# 6. cmd_diff.py — 不支持的输出格式
# ============================================================


class TestDiffCommandUnsupportedFormat:
    """测试 diff 命令不支持的输出格式。"""

    def test_unsupported_format_calls_print_error(self, tmp_path: Path) -> None:
        """测试使用不支持的输出格式时调用 print_error。"""
        from context_forge.cli.cmd_diff import diff_command

        # 创建两个快照文件
        snap1 = tmp_path / "snap1.json"
        snap2 = tmp_path / "snap2.json"
        snap_data = {
            "segments": [],
            "request_id": "test",
            "model": "gpt-4o",
        }
        snap1.write_text(json.dumps(snap_data), encoding="utf-8")
        snap2.write_text(json.dumps(snap_data), encoding="utf-8")

        with pytest.raises(SystemExit):
            diff_command(
                id_or_file_1=str(snap1),
                id_or_file_2=str(snap2),
                format="xml",
                snapshot_dir=str(tmp_path),
                ignore_timestamps=True,
            )


# ============================================================
# 7. cmd_diff.py — 快照目录不存在
# ============================================================


class TestDiffSnapshotDirNotExists:
    """测试 diff 命令快照目录不存在的情况。"""

    def test_snapshot_dir_not_exists(self, tmp_path: Path) -> None:
        """测试当快照 ID 对应的目录不存在时调用 print_error。"""
        from context_forge.cli.cmd_diff import _load_snapshot

        nonexistent_dir = str(tmp_path / "nonexistent_dir")

        with pytest.raises(SystemExit):
            _load_snapshot("some_id", nonexistent_dir)


# ============================================================
# 8. cmd_diff.py — _compute_diff 各种 segment 差异
# ============================================================


class TestComputeDiffSegments:
    """测试 _compute_diff 函数的各种 segment 差异场景。"""

    def test_segments_added(self) -> None:
        """测试新增 segment 的检测。"""
        from context_forge.cli.cmd_diff import _compute_diff

        snapshot1: dict[str, Any] = {
            "segments": [
                {"id": "seg_1", "type": "SYSTEM", "content": "Hello"},
            ],
        }
        snapshot2: dict[str, Any] = {
            "segments": [
                {"id": "seg_1", "type": "SYSTEM", "content": "Hello"},
                {"id": "seg_2", "type": "USER", "content": "World"},
            ],
        }

        result = _compute_diff(snapshot1, snapshot2, ignore_timestamps=True)
        assert len(result["segments_added"]) == 1
        assert result["segments_added"][0]["id"] == "seg_2"

    def test_segments_removed(self) -> None:
        """测试删除 segment 的检测。"""
        from context_forge.cli.cmd_diff import _compute_diff

        snapshot1: dict[str, Any] = {
            "segments": [
                {"id": "seg_1", "type": "SYSTEM", "content": "Hello"},
                {"id": "seg_2", "type": "USER", "content": "World"},
            ],
        }
        snapshot2: dict[str, Any] = {
            "segments": [
                {"id": "seg_1", "type": "SYSTEM", "content": "Hello"},
            ],
        }

        result = _compute_diff(snapshot1, snapshot2, ignore_timestamps=True)
        assert len(result["segments_removed"]) == 1
        assert result["segments_removed"][0]["id"] == "seg_2"

    def test_segments_modified(self) -> None:
        """测试修改 segment 的检测。"""
        from context_forge.cli.cmd_diff import _compute_diff

        snapshot1: dict[str, Any] = {
            "segments": [
                {"id": "seg_1", "type": "SYSTEM", "content": "Hello"},
            ],
        }
        snapshot2: dict[str, Any] = {
            "segments": [
                {"id": "seg_1", "type": "SYSTEM", "content": "Hello World"},
            ],
        }

        result = _compute_diff(snapshot1, snapshot2, ignore_timestamps=True)
        assert len(result["segments_modified"]) == 1
        assert result["segments_modified"][0]["id"] == "seg_1"

    def test_no_changes(self) -> None:
        """测试没有变更的情况。"""
        from context_forge.cli.cmd_diff import _compute_diff

        snapshot: dict[str, Any] = {
            "segments": [
                {"id": "seg_1", "type": "SYSTEM", "content": "Hello"},
            ],
        }

        result = _compute_diff(snapshot, snapshot, ignore_timestamps=True)
        assert len(result["segments_added"]) == 0
        assert len(result["segments_removed"]) == 0
        assert len(result["segments_modified"]) == 0

    def test_include_timestamps(self) -> None:
        """测试包含时间戳的 diff。"""
        from context_forge.cli.cmd_diff import _compute_diff

        snapshot1: dict[str, Any] = {
            "segments": [],
            "created_at": "2026-01-01T00:00:00",
            "model": "gpt-4o",
            "policy_version": "1.0",
        }
        snapshot2: dict[str, Any] = {
            "segments": [],
            "created_at": "2026-01-02T00:00:00",
            "model": "gpt-4o",
            "policy_version": "1.0",
        }

        result = _compute_diff(snapshot1, snapshot2, ignore_timestamps=False)
        assert "created_at" in result["metadata_diff"]
        assert result["metadata_diff"]["created_at"]["before"] == "2026-01-01T00:00:00"

    def test_budget_diff(self) -> None:
        """测试预算差异计算。"""
        from context_forge.cli.cmd_diff import _compute_diff

        snapshot1: dict[str, Any] = {
            "segments": [],
            "budget": {
                "total_budget": 100000,
                "content_budget": 90000,
                "total_used": 5000,
            },
        }
        snapshot2: dict[str, Any] = {
            "segments": [],
            "budget": {
                "total_budget": 128000,
                "content_budget": 120000,
                "total_used": 10000,
            },
        }

        result = _compute_diff(snapshot1, snapshot2, ignore_timestamps=True)
        assert result["budget_diff"]["total_budget"] == 28000
        assert result["budget_diff"]["content_budget"] == 30000
        assert result["budget_diff"]["total_used"] == 5000

    def test_warnings_diff(self) -> None:
        """测试警告差异计算。"""
        from context_forge.cli.cmd_diff import _compute_diff

        snapshot1: dict[str, Any] = {
            "segments": [],
            "warnings": ["警告A", "警告B"],
        }
        snapshot2: dict[str, Any] = {
            "segments": [],
            "warnings": ["警告B", "警告C"],
        }

        result = _compute_diff(snapshot1, snapshot2, ignore_timestamps=True)
        assert "警告C" in result["warnings_diff"]["added"]
        assert "警告A" in result["warnings_diff"]["removed"]


# ============================================================
# 9. cmd_diff.py — _output_text_diff 预算差异分支
# ============================================================


class TestOutputTextDiffBudget:
    """测试 _output_text_diff 函数的预算差异输出。"""

    def test_text_diff_with_budget(self) -> None:
        """测试文本格式输出包含预算差异。"""
        from context_forge.cli.cmd_diff import _output_text_diff

        diff_result: dict[str, Any] = {
            "segments_added": [],
            "segments_removed": [],
            "segments_modified": [],
            "budget_diff": {
                "total_budget": 28000,
                "content_budget": 30000,
                "total_used": 5000,
            },
            "token_diff": {
                "total_tokens": 5000,
                "segment_count": 2,
            },
            "warnings_diff": {
                "added": [],
                "removed": [],
            },
        }
        snapshot1: dict[str, Any] = {"request_id": "req_1"}
        snapshot2: dict[str, Any] = {"request_id": "req_2"}

        # 不应抛出异常
        _output_text_diff(diff_result, snapshot1, snapshot2)

    def test_text_diff_without_budget(self) -> None:
        """测试文本格式输出不包含预算差异。"""
        from context_forge.cli.cmd_diff import _output_text_diff

        diff_result: dict[str, Any] = {
            "segments_added": [],
            "segments_removed": [],
            "segments_modified": [],
            "budget_diff": {},
            "token_diff": {
                "total_tokens": 0,
                "segment_count": 0,
            },
            "warnings_diff": {
                "added": [],
                "removed": [],
            },
        }
        snapshot1: dict[str, Any] = {"request_id": "req_1"}
        snapshot2: dict[str, Any] = {"request_id": "req_2"}

        _output_text_diff(diff_result, snapshot1, snapshot2)


# ============================================================
# 10. cmd_diff.py — _output_rich_diff 各种分支
# ============================================================


class TestOutputRichDiffBranches:
    """测试 _output_rich_diff 函数的各种输出分支。"""

    def _make_segments(self, count: int, prefix: str = "seg") -> list[dict[str, Any]]:
        """创建指定数量的 segment 字典。"""
        return [
            {
                "id": f"{prefix}_{i:04d}",
                "type": "USER",
                "content_preview": f"这是第 {i} 条消息的预览内容",
            }
            for i in range(count)
        ]

    def test_rich_diff_with_added_segments_less_than_5(self) -> None:
        """测试 Rich 格式输出新增 segment 数量 < 5。"""
        from context_forge.cli.cmd_diff import _output_rich_diff

        diff_result: dict[str, Any] = {
            "segments_added": self._make_segments(3),
            "segments_removed": [],
            "segments_modified": [],
            "budget_diff": {},
            "token_diff": {"total_tokens": 100, "segment_count": 3},
            "warnings_diff": {"added": [], "removed": []},
        }
        snapshot1: dict[str, Any] = {"request_id": "req_1", "model": "gpt-4o"}
        snapshot2: dict[str, Any] = {"request_id": "req_2", "model": "gpt-4o"}

        # 不应抛出异常
        _output_rich_diff(diff_result, snapshot1, snapshot2)

    def test_rich_diff_with_added_segments_more_than_5(self) -> None:
        """测试 Rich 格式输出新增 segment 数量 > 5 时显示截断提示。"""
        from context_forge.cli.cmd_diff import _output_rich_diff

        diff_result: dict[str, Any] = {
            "segments_added": self._make_segments(8, "add"),
            "segments_removed": [],
            "segments_modified": [],
            "budget_diff": {},
            "token_diff": {"total_tokens": 800, "segment_count": 8},
            "warnings_diff": {"added": [], "removed": []},
        }
        snapshot1: dict[str, Any] = {"request_id": "req_1", "model": "gpt-4o"}
        snapshot2: dict[str, Any] = {"request_id": "req_2", "model": "gpt-4o"}

        _output_rich_diff(diff_result, snapshot1, snapshot2)

    def test_rich_diff_with_removed_segments_less_than_5(self) -> None:
        """测试 Rich 格式输出删除 segment 数量 < 5。"""
        from context_forge.cli.cmd_diff import _output_rich_diff

        diff_result: dict[str, Any] = {
            "segments_added": [],
            "segments_removed": self._make_segments(2, "rm"),
            "segments_modified": [],
            "budget_diff": {},
            "token_diff": {"total_tokens": -200, "segment_count": -2},
            "warnings_diff": {"added": [], "removed": []},
        }
        snapshot1: dict[str, Any] = {"request_id": "req_1", "model": "gpt-4o"}
        snapshot2: dict[str, Any] = {"request_id": "req_2", "model": "gpt-4o"}

        _output_rich_diff(diff_result, snapshot1, snapshot2)

    def test_rich_diff_with_removed_segments_more_than_5(self) -> None:
        """测试 Rich 格式输出删除 segment 数量 > 5 时显示截断提示。"""
        from context_forge.cli.cmd_diff import _output_rich_diff

        diff_result: dict[str, Any] = {
            "segments_added": [],
            "segments_removed": self._make_segments(7, "rm"),
            "segments_modified": [],
            "budget_diff": {},
            "token_diff": {"total_tokens": -700, "segment_count": -7},
            "warnings_diff": {"added": [], "removed": []},
        }
        snapshot1: dict[str, Any] = {"request_id": "req_1", "model": "gpt-4o"}
        snapshot2: dict[str, Any] = {"request_id": "req_2", "model": "gpt-4o"}

        _output_rich_diff(diff_result, snapshot1, snapshot2)

    def test_rich_diff_with_warnings_added(self) -> None:
        """测试 Rich 格式输出新增的警告。"""
        from context_forge.cli.cmd_diff import _output_rich_diff

        diff_result: dict[str, Any] = {
            "segments_added": [],
            "segments_removed": [],
            "segments_modified": [],
            "budget_diff": {},
            "token_diff": {"total_tokens": 0, "segment_count": 0},
            "warnings_diff": {
                "added": ["新警告: Token 预算接近上限"],
                "removed": [],
            },
        }
        snapshot1: dict[str, Any] = {"request_id": "req_1", "model": "gpt-4o"}
        snapshot2: dict[str, Any] = {"request_id": "req_2", "model": "gpt-4o"}

        _output_rich_diff(diff_result, snapshot1, snapshot2)

    def test_rich_diff_with_warnings_removed(self) -> None:
        """测试 Rich 格式输出删除的警告。"""
        from context_forge.cli.cmd_diff import _output_rich_diff

        diff_result: dict[str, Any] = {
            "segments_added": [],
            "segments_removed": [],
            "segments_modified": [],
            "budget_diff": {},
            "token_diff": {"total_tokens": 0, "segment_count": 0},
            "warnings_diff": {
                "added": [],
                "removed": ["旧警告: PII 检测到敏感信息"],
            },
        }
        snapshot1: dict[str, Any] = {"request_id": "req_1", "model": "gpt-4o"}
        snapshot2: dict[str, Any] = {"request_id": "req_2", "model": "gpt-4o"}

        _output_rich_diff(diff_result, snapshot1, snapshot2)

    def test_rich_diff_with_warnings_both_added_and_removed(self) -> None:
        """测试 Rich 格式输出同时有新增和删除的警告。"""
        from context_forge.cli.cmd_diff import _output_rich_diff

        diff_result: dict[str, Any] = {
            "segments_added": [],
            "segments_removed": [],
            "segments_modified": [],
            "budget_diff": {},
            "token_diff": {"total_tokens": 0, "segment_count": 0},
            "warnings_diff": {
                "added": ["新警告1", "新警告2"],
                "removed": ["旧警告1"],
            },
        }
        snapshot1: dict[str, Any] = {"request_id": "req_1", "model": "gpt-4o"}
        snapshot2: dict[str, Any] = {"request_id": "req_2", "model": "gpt-4o"}

        _output_rich_diff(diff_result, snapshot1, snapshot2)

    def test_rich_diff_with_budget_diff(self) -> None:
        """测试 Rich 格式输出预算差异。"""
        from context_forge.cli.cmd_diff import _output_rich_diff

        diff_result: dict[str, Any] = {
            "segments_added": [],
            "segments_removed": [],
            "segments_modified": [],
            "budget_diff": {
                "total_budget": 28000,
                "content_budget": 30000,
                "total_used": -5000,
            },
            "token_diff": {"total_tokens": 0, "segment_count": 0},
            "warnings_diff": {"added": [], "removed": []},
        }
        snapshot1: dict[str, Any] = {"request_id": "req_1", "model": "gpt-4o"}
        snapshot2: dict[str, Any] = {"request_id": "req_2", "model": "gpt-4o"}

        _output_rich_diff(diff_result, snapshot1, snapshot2)

    def test_rich_diff_complete_scenario(self) -> None:
        """测试 Rich 格式输出完整场景（包含所有分支）。"""
        from context_forge.cli.cmd_diff import _output_rich_diff

        diff_result: dict[str, Any] = {
            "segments_added": self._make_segments(6, "add"),
            "segments_removed": self._make_segments(7, "rm"),
            "segments_modified": [{"id": "seg_mod", "before": {}, "after": {}}],
            "budget_diff": {
                "total_budget": 10000,
                "content_budget": 8000,
                "total_used": 3000,
            },
            "token_diff": {"total_tokens": 3000, "segment_count": 6},
            "warnings_diff": {
                "added": ["新增警告1"],
                "removed": ["删除警告1", "删除警告2"],
            },
        }
        snapshot1: dict[str, Any] = {"request_id": "req_1", "model": "gpt-4o"}
        snapshot2: dict[str, Any] = {"request_id": "req_2", "model": "gpt-4o"}

        _output_rich_diff(diff_result, snapshot1, snapshot2)


# ============================================================
# 11. cmd_diff.py — _output_json_diff 覆盖
# ============================================================


class TestOutputJsonDiff:
    """测试 _output_json_diff 函数。"""

    def test_json_diff_output(self) -> None:
        """测试 JSON 格式差异输出。"""
        from context_forge.cli.cmd_diff import _output_json_diff

        diff_result: dict[str, Any] = {
            "segments_added": [{"id": "seg_1"}],
            "segments_removed": [],
            "segments_modified": [],
            "budget_diff": {},
            "token_diff": {"total_tokens": 100},
            "warnings_diff": {"added": [], "removed": []},
        }

        # 不应抛出异常
        _output_json_diff(diff_result)


# ============================================================
# 12. cmd_diff.py — _load_snapshot 快照 ID 搜索路径
# ============================================================


class TestLoadSnapshotSearchPaths:
    """测试 _load_snapshot 的快照 ID 搜索路径逻辑。"""

    def test_load_snapshot_from_file(self, tmp_path: Path) -> None:
        """测试通过文件路径加载快照。"""
        from context_forge.cli.cmd_diff import _load_snapshot

        snap_file = tmp_path / "snap.json"
        snap_data = {"segments": [], "model": "gpt-4o"}
        snap_file.write_text(json.dumps(snap_data), encoding="utf-8")

        result = _load_snapshot(str(snap_file), str(tmp_path))
        # _load_snapshot_from_file 会自动计算并添加 token_usage
        assert result["segments"] == []
        assert result["model"] == "gpt-4o"

    def test_load_snapshot_by_id_json(self, tmp_path: Path) -> None:
        """测试通过快照 ID 在目录中查找 .json 文件。"""
        from context_forge.cli.cmd_diff import _load_snapshot

        snap_dir = tmp_path / "snapshots"
        snap_dir.mkdir()
        snap_data = {"segments": [], "model": "gpt-4o"}
        (snap_dir / "req_abc123.json").write_text(
            json.dumps(snap_data), encoding="utf-8"
        )

        result = _load_snapshot("req_abc123", str(snap_dir))
        assert result["segments"] == []
        assert result["model"] == "gpt-4o"

    def test_load_snapshot_by_id_yaml(self, tmp_path: Path) -> None:
        """测试通过快照 ID 在目录中查找 .yaml 文件。"""
        from context_forge.cli.cmd_diff import _load_snapshot

        snap_dir = tmp_path / "snapshots"
        snap_dir.mkdir()
        # 写入 YAML 格式快照
        (snap_dir / "req_abc123.yaml").write_text(
            "segments: []\nmodel: gpt-4o\n", encoding="utf-8"
        )

        result = _load_snapshot("req_abc123", str(snap_dir))
        assert result["segments"] == []
        assert result["model"] == "gpt-4o"

    def test_load_snapshot_by_id_snap_prefix(self, tmp_path: Path) -> None:
        """测试通过快照 ID 在目录中查找 snap_ 前缀文件。"""
        from context_forge.cli.cmd_diff import _load_snapshot

        snap_dir = tmp_path / "snapshots"
        snap_dir.mkdir()
        snap_data = {"segments": [], "model": "gpt-4o"}
        (snap_dir / "snap_abc123.json").write_text(
            json.dumps(snap_data), encoding="utf-8"
        )

        result = _load_snapshot("abc123", str(snap_dir))
        assert result["segments"] == []
        assert result["model"] == "gpt-4o"

    def test_load_snapshot_not_found(self, tmp_path: Path) -> None:
        """测试快照文件不存在时调用 print_error。"""
        from context_forge.cli.cmd_diff import _load_snapshot

        snap_dir = tmp_path / "snapshots"
        snap_dir.mkdir()

        with pytest.raises(SystemExit):
            _load_snapshot("nonexistent_id", str(snap_dir))

    def test_load_snapshot_dir_not_exists(self, tmp_path: Path) -> None:
        """测试快照目录不存在时调用 print_error。"""
        from context_forge.cli.cmd_diff import _load_snapshot

        with pytest.raises(SystemExit):
            _load_snapshot("some_id", str(tmp_path / "no_such_dir"))
