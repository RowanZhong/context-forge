"""
YAML 策略文件加载与校验。

→ 6.1.2.2 Policy-as-Code：基于 YAML/JSON 的策略编排与版本管理

策略文件是 Context Forge 配置的核心载体。本模块负责：
1. 从文件路径或目录加载 YAML 策略
2. 使用 Pydantic Schema 校验策略内容
3. 合并多层策略（默认 → 项目级 → 运行时覆盖）
4. 提供人类可读的校验错误信息

# [DX Decision] 策略加载失败时的错误信息必须精确到字段级别，
# 告诉用户哪个文件、哪个字段、什么值有问题、应该改成什么。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from context_forge.config.schema import PolicyConfig
from context_forge.errors import ConfigValidationError, PolicyLoadError

logger = logging.getLogger(__name__)

# 默认策略文件搜索路径
_SEARCH_PATHS = [
    Path("context_forge.yaml"),
    Path("context_forge.yml"),
    Path("configs/default_policy.yaml"),
    Path(".context_forge/policy.yaml"),
]


def load_policy(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> PolicyConfig:
    """
    加载并校验策略配置。

    加载优先级：
    1. 显式指定的路径
    2. 当前目录下的默认搜索路径
    3. 包内置的默认配置

    参数:
        path: YAML 文件路径。None 时自动搜索默认路径。
        overrides: 运行时覆盖的配置项（合并到 YAML 配置之上）

    返回:
        PolicyConfig 实例

    异常:
        PolicyLoadError: 文件不存在或格式错误
        ConfigValidationError: 配置校验失败
    """
    raw_config: dict[str, Any] = {}

    if path is not None:
        raw_config = _load_yaml_file(Path(path))
    else:
        # 自动搜索
        for search_path in _SEARCH_PATHS:
            if search_path.exists():
                logger.info("自动发现策略文件：%s", search_path)
                raw_config = _load_yaml_file(search_path)
                break
        # 如果没找到，使用空字典（全部走默认值）
        if not raw_config:
            logger.info("未找到策略文件，使用默认配置。")

    # 合并运行时覆盖
    if overrides:
        raw_config = _deep_merge(raw_config, overrides)

    # 校验
    return _validate_config(raw_config, str(path) if path else "<default>")


def _load_yaml_file(path: Path) -> dict[str, Any]:
    """加载并解析 YAML 文件。"""
    if not path.exists():
        raise PolicyLoadError(
            what=f"策略文件 '{path}' 不存在。",
            why=f"在路径 '{path.absolute()}' 下未找到该文件。",
            how="请检查文件路径是否正确。"
                "可以使用 'context-forge init' 在当前目录生成默认策略文件。",
            file_path=str(path),
        )

    try:
        content = path.read_text(encoding="utf-8")
    except Exception as e:
        raise PolicyLoadError(
            what=f"无法读取策略文件 '{path}'。",
            why=str(e),
            how="请检查文件权限和编码（需要 UTF-8）。",
            file_path=str(path),
        ) from e

    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as e:
        raise PolicyLoadError(
            what=f"策略文件 '{path}' 的 YAML 格式无效。",
            why=str(e),
            how="请使用 YAML 格式校验工具检查文件语法。"
                "在线工具：https://www.yamllint.com/",
            file_path=str(path),
        ) from e

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise PolicyLoadError(
            what=f"策略文件 '{path}' 的根元素必须是字典（mapping）。",
            why=f"实际类型为 {type(data).__name__}。",
            how="请确保 YAML 文件的根元素是键值对形式，例如：\n"
                "  version: '1.0'\n"
                "  budget:\n"
                "    max_context_tokens: 128000",
            file_path=str(path),
        )

    return data


def _validate_config(raw: dict[str, Any], source: str) -> PolicyConfig:
    """使用 Pydantic 校验配置字典。"""
    try:
        return PolicyConfig(**raw)
    except ValidationError as e:
        # 将 Pydantic 校验错误转换为用户友好的格式
        error_details = []
        for err in e.errors():
            field_path = " → ".join(str(loc) for loc in err["loc"])
            error_details.append(f"  字段 '{field_path}': {err['msg']}")

        raise ConfigValidationError(
            what=f"策略配置 '{source}' 校验失败（{len(e.errors())} 个错误）。",
            why="\n".join(error_details),
            how="请对照文档 docs/configuration.md 修正配置项。"
                "可以使用 'context-forge validate <path>' 命令进行预校验。",
            config_path=source,
        ) from e


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    深度合并两个字典。override 中的值优先。

    # [Design Decision] 深度合并而非浅覆盖，
    # 让用户可以只覆盖需要修改的字段，而非重写整个配置。
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def validate_policy_file(path: str | Path) -> list[str]:
    """
    校验策略文件，返回错误列表。

    这个方法不会抛出异常，而是收集所有错误并返回。
    用于 CLI 的 validate 命令和 CI 流程。

    参数:
        path: YAML 文件路径

    返回:
        错误信息列表（空列表表示校验通过）
    """
    errors: list[str] = []

    try:
        load_policy(path=path)
    except PolicyLoadError as e:
        errors.append(e.full_message)
    except ConfigValidationError as e:
        errors.append(e.full_message)
    except Exception as e:
        errors.append(f"未预期的错误：{e}")

    return errors
