"""
config/loader.py 覆盖率补充测试。

目标：将 loader.py 的覆盖率从 ~64% 提升到 80%+。
覆盖的未测试分支:
- 自动搜索策略文件路径（_SEARCH_PATHS 循环）
- 未找到策略文件时使用默认配置
- 运行时覆盖合并（overrides 参数 + _deep_merge）
- 文件读取异常处理（权限/编码错误）
- 空 YAML 文件处理（data is None）
- YAML 根元素非字典时的错误
- YAML 格式无效时的错误
- _deep_merge 深度合并逻辑
- validate_policy_file 函数（成功/失败/意外异常路径）
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from context_forge.config.loader import (
    _deep_merge,
    _load_yaml_file,
    _validate_config,
    load_policy,
    validate_policy_file,
)
from context_forge.config.schema import PolicyConfig
from context_forge.errors import ConfigValidationError, PolicyLoadError


# === _load_yaml_file 边界测试 ===


class TestLoadYamlFile:
    """_load_yaml_file 内部函数测试，覆盖各种错误分支。"""

    def test_file_not_exists_raises_policy_load_error(self, tmp_path: Path) -> None:
        """测试文件不存在时抛出 PolicyLoadError，并包含三段式信息。"""
        nonexistent = tmp_path / "nonexistent.yaml"
        with pytest.raises(PolicyLoadError) as exc_info:
            _load_yaml_file(nonexistent)

        error = exc_info.value
        assert "不存在" in error.what
        assert str(nonexistent) in error.file_path

    def test_file_read_error_raises_policy_load_error(self, tmp_path: Path) -> None:
        """测试文件读取失败时抛出 PolicyLoadError（权限/编码错误）。"""
        test_file = tmp_path / "unreadable.yaml"
        test_file.write_text("valid: yaml", encoding="utf-8")

        # Mock Path.read_text 抛出异常，模拟权限错误
        with patch.object(Path, "read_text", side_effect=PermissionError("权限不足")):
            with pytest.raises(PolicyLoadError) as exc_info:
                _load_yaml_file(test_file)

            error = exc_info.value
            assert "无法读取" in error.what
            assert "权限" in error.how or "UTF-8" in error.how

    def test_invalid_yaml_syntax_raises_policy_load_error(
        self, tmp_path: Path,
    ) -> None:
        """测试无效 YAML 语法时抛出 PolicyLoadError。"""
        invalid_file = tmp_path / "invalid_syntax.yaml"
        # 写入无效 YAML（缩进错误 + 重复键等场景）
        invalid_file.write_text(
            "key:\n  - item1\n item2\n",  # 缩进不一致
            encoding="utf-8",
        )

        with pytest.raises(PolicyLoadError) as exc_info:
            _load_yaml_file(invalid_file)

        error = exc_info.value
        assert "YAML 格式无效" in error.what
        assert "yamllint" in error.how

    def test_empty_yaml_returns_empty_dict(self, tmp_path: Path) -> None:
        """测试空 YAML 文件返回空字典（data is None 分支）。"""
        empty_file = tmp_path / "empty.yaml"
        empty_file.write_text("", encoding="utf-8")

        result = _load_yaml_file(empty_file)
        assert result == {}

    def test_yaml_with_only_comments_returns_empty_dict(
        self, tmp_path: Path,
    ) -> None:
        """测试仅包含注释的 YAML 文件返回空字典。"""
        comments_file = tmp_path / "comments_only.yaml"
        comments_file.write_text("# 这是一个注释\n# 另一个注释\n", encoding="utf-8")

        result = _load_yaml_file(comments_file)
        assert result == {}

    def test_yaml_root_is_list_raises_policy_load_error(
        self, tmp_path: Path,
    ) -> None:
        """测试 YAML 根元素是列表而非字典时抛出错误。"""
        list_file = tmp_path / "list_root.yaml"
        list_file.write_text("- item1\n- item2\n- item3\n", encoding="utf-8")

        with pytest.raises(PolicyLoadError) as exc_info:
            _load_yaml_file(list_file)

        error = exc_info.value
        assert "根元素必须是字典" in error.what
        assert "list" in error.why

    def test_yaml_root_is_string_raises_policy_load_error(
        self, tmp_path: Path,
    ) -> None:
        """测试 YAML 根元素是纯字符串时抛出错误。"""
        string_file = tmp_path / "string_root.yaml"
        string_file.write_text("just a plain string\n", encoding="utf-8")

        with pytest.raises(PolicyLoadError) as exc_info:
            _load_yaml_file(string_file)

        error = exc_info.value
        assert "根元素必须是字典" in error.what
        assert "str" in error.why

    def test_yaml_root_is_number_raises_policy_load_error(
        self, tmp_path: Path,
    ) -> None:
        """测试 YAML 根元素是数字时抛出错误。"""
        number_file = tmp_path / "number_root.yaml"
        number_file.write_text("42\n", encoding="utf-8")

        with pytest.raises(PolicyLoadError) as exc_info:
            _load_yaml_file(number_file)

        error = exc_info.value
        assert "根元素必须是字典" in error.what
        assert "int" in error.why

    def test_valid_yaml_returns_dict(self, tmp_path: Path) -> None:
        """测试有效 YAML 文件返回正确的字典。"""
        valid_file = tmp_path / "valid.yaml"
        valid_file.write_text(
            "version: '1.0'\nname: 'test'\n",
            encoding="utf-8",
        )

        result = _load_yaml_file(valid_file)
        assert result == {"version": "1.0", "name": "test"}


# === _deep_merge 测试 ===


class TestDeepMerge:
    """_deep_merge 深度合并函数测试。"""

    def test_merge_flat_dicts(self) -> None:
        """测试扁平字典合并，override 优先。"""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)

        assert result == {"a": 1, "b": 3, "c": 4}
        # 原字典不被修改
        assert base == {"a": 1, "b": 2}

    def test_merge_nested_dicts(self) -> None:
        """测试嵌套字典深度合并。"""
        base = {
            "budget": {"max_context_tokens": 128000, "output_reserved": 4096},
            "sanitize": {"strip_html": True},
        }
        override = {
            "budget": {"max_context_tokens": 8192},
            "cache": {"enabled": True},
        }
        result = _deep_merge(base, override)

        # budget 中被覆盖的字段更新，其他保留
        assert result["budget"]["max_context_tokens"] == 8192
        assert result["budget"]["output_reserved"] == 4096
        # sanitize 保持不变
        assert result["sanitize"]["strip_html"] is True
        # cache 新增
        assert result["cache"]["enabled"] is True

    def test_merge_override_replaces_non_dict_with_dict(self) -> None:
        """测试覆盖时将非字典值替换为字典。"""
        base: dict[str, Any] = {"key": "string_value"}
        override: dict[str, Any] = {"key": {"nested": True}}
        result = _deep_merge(base, override)

        assert result["key"] == {"nested": True}

    def test_merge_override_replaces_dict_with_non_dict(self) -> None:
        """测试覆盖时将字典值替换为非字典。"""
        base: dict[str, Any] = {"key": {"nested": True}}
        override: dict[str, Any] = {"key": "replaced"}
        result = _deep_merge(base, override)

        assert result["key"] == "replaced"

    def test_merge_empty_override(self) -> None:
        """测试空 override 返回 base 的副本。"""
        base = {"a": 1, "b": {"c": 2}}
        result = _deep_merge(base, {})

        assert result == base
        assert result is not base  # 应是副本

    def test_merge_empty_base(self) -> None:
        """测试空 base 返回 override 内容。"""
        override = {"a": 1, "b": {"c": 2}}
        result = _deep_merge({}, override)

        assert result == override

    def test_merge_deeply_nested(self) -> None:
        """测试三层嵌套的深度合并。"""
        base: dict[str, Any] = {
            "level1": {
                "level2": {
                    "level3_a": "original",
                    "level3_b": "keep",
                },
            },
        }
        override: dict[str, Any] = {
            "level1": {
                "level2": {
                    "level3_a": "updated",
                    "level3_c": "new",
                },
            },
        }
        result = _deep_merge(base, override)

        assert result["level1"]["level2"]["level3_a"] == "updated"
        assert result["level1"]["level2"]["level3_b"] == "keep"
        assert result["level1"]["level2"]["level3_c"] == "new"


# === _validate_config 测试 ===


class TestValidateConfig:
    """_validate_config 函数测试。"""

    def test_valid_config_returns_policy(self) -> None:
        """测试有效配置成功返回 PolicyConfig。"""
        raw = {"version": "2.0", "name": "test-config"}
        result = _validate_config(raw, "test.yaml")

        assert isinstance(result, PolicyConfig)
        assert result.version == "2.0"
        assert result.name == "test-config"

    def test_empty_config_uses_defaults(self) -> None:
        """测试空字典使用默认配置。"""
        result = _validate_config({}, "<default>")

        assert isinstance(result, PolicyConfig)
        assert result.version == "1.0"  # 默认版本
        assert result.budget.max_context_tokens > 0

    def test_invalid_config_raises_config_validation_error(self) -> None:
        """测试无效配置抛出 ConfigValidationError，含字段路径。"""
        raw = {
            "budget": {"max_context_tokens": -1000},
        }
        with pytest.raises(ConfigValidationError) as exc_info:
            _validate_config(raw, "invalid.yaml")

        error = exc_info.value
        assert "校验失败" in error.what
        assert "invalid.yaml" in error.what
        assert "字段" in error.why
        assert "validate" in error.how or "文档" in error.how

    def test_invalid_config_with_multiple_errors(self) -> None:
        """测试多个字段校验错误时全部包含在错误信息中。"""
        raw = {
            "budget": {"max_context_tokens": -1},
            "rerank": {"mmr_lambda": 5.0},
        }
        with pytest.raises(ConfigValidationError) as exc_info:
            _validate_config(raw, "multi_error.yaml")

        error = exc_info.value
        assert "2 个错误" in error.what or "个错误" in error.what


# === load_policy 高级测试 ===


class TestLoadPolicyAdvanced:
    """load_policy 函数高级测试 — 覆盖自动搜索和覆盖合并路径。"""

    def test_load_with_overrides(self, tmp_path: Path) -> None:
        """测试 overrides 参数能正确合并到配置。"""
        yaml_file = tmp_path / "base.yaml"
        yaml_file.write_text(
            "version: '1.0'\nname: 'base-policy'\n"
            "budget:\n  max_context_tokens: 128000\n",
            encoding="utf-8",
        )

        policy = load_policy(
            path=yaml_file,
            overrides={"budget": {"max_context_tokens": 8192}},
        )

        assert policy.budget.max_context_tokens == 8192
        assert policy.name == "base-policy"

    def test_load_with_overrides_no_path(self) -> None:
        """测试仅有 overrides、无路径时使用默认配置 + 覆盖。"""
        policy = load_policy(
            overrides={"version": "override-1.0", "name": "override-policy"},
        )

        assert policy.version == "override-1.0"
        assert policy.name == "override-policy"

    def test_auto_search_finds_existing_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """测试自动搜索路径找到存在的策略文件。"""
        # 在当前目录下创建 context_forge.yaml
        monkeypatch.chdir(tmp_path)
        config_file = tmp_path / "context_forge.yaml"
        config_file.write_text(
            "version: 'auto-found'\nname: 'auto-policy'\n",
            encoding="utf-8",
        )

        policy = load_policy()
        assert policy.version == "auto-found"
        assert policy.name == "auto-policy"

    def test_auto_search_finds_yml_extension(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """测试自动搜索路径找到 .yml 扩展名的文件。"""
        monkeypatch.chdir(tmp_path)
        config_file = tmp_path / "context_forge.yml"
        config_file.write_text(
            "version: 'yml-found'\nname: 'yml-policy'\n",
            encoding="utf-8",
        )

        policy = load_policy()
        assert policy.version == "yml-found"

    def test_auto_search_no_file_uses_defaults(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """测试自动搜索未找到任何文件时使用默认配置。"""
        monkeypatch.chdir(tmp_path)
        # tmp_path 是空的，不会找到任何策略文件

        policy = load_policy()
        assert isinstance(policy, PolicyConfig)
        assert policy.version == "1.0"  # 默认版本

    def test_load_empty_yaml_file(self, tmp_path: Path) -> None:
        """测试加载空 YAML 文件，应使用全部默认值。"""
        empty_file = tmp_path / "empty.yaml"
        empty_file.write_text("", encoding="utf-8")

        policy = load_policy(path=empty_file)
        assert isinstance(policy, PolicyConfig)
        assert policy.budget.max_context_tokens > 0

    def test_load_comments_only_yaml(self, tmp_path: Path) -> None:
        """测试加载仅含注释的 YAML 文件。"""
        comments_file = tmp_path / "comments.yaml"
        comments_file.write_text(
            "# 这是注释\n# 另一行注释\n",
            encoding="utf-8",
        )

        policy = load_policy(path=comments_file)
        assert isinstance(policy, PolicyConfig)


# === validate_policy_file 测试 ===


class TestValidatePolicyFile:
    """validate_policy_file 函数测试。"""

    def test_validate_valid_file_returns_empty_list(
        self, tmp_path: Path,
    ) -> None:
        """测试校验有效文件返回空错误列表。"""
        valid_file = tmp_path / "valid.yaml"
        valid_file.write_text(
            "version: '1.0'\nname: 'test'\n",
            encoding="utf-8",
        )

        errors = validate_policy_file(valid_file)
        assert errors == []

    def test_validate_nonexistent_file_returns_error(self) -> None:
        """测试校验不存在的文件返回 PolicyLoadError 信息。"""
        errors = validate_policy_file("/nonexistent/path/policy.yaml")
        assert len(errors) == 1
        assert "不存在" in errors[0]

    def test_validate_invalid_yaml_returns_error(self, tmp_path: Path) -> None:
        """测试校验无效 YAML 返回错误信息。"""
        invalid_file = tmp_path / "invalid.yaml"
        invalid_file.write_text(
            "key:\n  - item1\n item2\n",  # 缩进错误
            encoding="utf-8",
        )

        errors = validate_policy_file(invalid_file)
        assert len(errors) == 1
        assert "YAML" in errors[0] or "格式" in errors[0]

    def test_validate_invalid_config_returns_error(self, tmp_path: Path) -> None:
        """测试校验配置验证失败返回 ConfigValidationError 信息。"""
        invalid_config_file = tmp_path / "bad_config.yaml"
        invalid_config_file.write_text(
            "budget:\n  max_context_tokens: -1\n",
            encoding="utf-8",
        )

        errors = validate_policy_file(invalid_config_file)
        assert len(errors) == 1
        assert "校验失败" in errors[0]

    def test_validate_unexpected_error_returns_error(
        self, tmp_path: Path,
    ) -> None:
        """测试校验时出现意外异常时返回通用错误信息。"""
        valid_file = tmp_path / "trigger_unexpected.yaml"
        valid_file.write_text("version: '1.0'\n", encoding="utf-8")

        # Mock load_policy 抛出非预期的异常
        with patch(
            "context_forge.config.loader.load_policy",
            side_effect=RuntimeError("意外的内部错误"),
        ):
            errors = validate_policy_file(valid_file)

        assert len(errors) == 1
        assert "未预期的错误" in errors[0]
        assert "意外的内部错误" in errors[0]

    def test_validate_with_string_path(self, tmp_path: Path) -> None:
        """测试使用字符串路径（非 Path 对象）调用。"""
        valid_file = tmp_path / "valid_str.yaml"
        valid_file.write_text("version: '2.0'\n", encoding="utf-8")

        errors = validate_policy_file(str(valid_file))
        assert errors == []

    def test_validate_root_is_list_returns_error(self, tmp_path: Path) -> None:
        """测试校验根元素为列表的 YAML 文件。"""
        list_file = tmp_path / "list_root.yaml"
        list_file.write_text("- item1\n- item2\n", encoding="utf-8")

        errors = validate_policy_file(list_file)
        assert len(errors) == 1
        assert "根元素" in errors[0] or "字典" in errors[0]
