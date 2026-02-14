"""
CLI 工具函数 — Rich 美化、文件加载、通用辅助。

→ 6.1.2.2 CLI 工具链设计

提供 CLI 各子命令共用的实用函数，包括：
- Rich Console 美化输出
- JSON/YAML 文件加载
- 错误/成功信息统一格式
- Token 数字格式化
- ContextForge 实例创建
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, NoReturn

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from context_forge import ContextForge
from context_forge.errors import ContextForgeError

# 全局 Console 实例
_console: Console | None = None


def create_console() -> Console:
    """
    创建或获取全局 Rich Console 实例。

    # [DX Decision] 全局单例 Console，确保所有 CLI 输出格式一致。
    """
    global _console
    if _console is None:
        _console = Console()
    return _console


def print_error(message: str, exit_code: int = 1) -> NoReturn:
    """
    打印错误信息并退出程序。

    参数:
        message: 错误信息
        exit_code: 退出码（默认 1）
    """
    console = create_console()
    # [DX Decision] 使用 X 而非 ✗，避免 Windows 终端编码问题
    console.print(f"[bold red]X 错误：[/bold red]{message}")
    sys.exit(exit_code)


def print_success(message: str) -> None:
    """
    打印成功信息。

    参数:
        message: 成功信息
    """
    console = create_console()
    # [DX Decision] 使用 OK 而非 ✓，避免 Windows 终端编码问题
    console.print(f"[bold green]OK[/bold green] {message}")


def print_warning(message: str) -> None:
    """
    打印警告信息。

    参数:
        message: 警告信息
    """
    console = create_console()
    # [DX Decision] 使用纯文本警告符号而非 Unicode 字符，避免 Windows 终端编码问题
    console.print(f"[bold yellow]![/bold yellow] {message}")


def print_info(message: str) -> None:
    """
    打印普通信息。

    参数:
        message: 信息内容
    """
    console = create_console()
    console.print(message)


def format_token_count(count: int) -> str:
    """
    格式化 Token 数字为带千分位分隔符的字符串。

    参数:
        count: Token 数量

    返回:
        格式化后的字符串，例如 "128,000"

    示例::

        >>> format_token_count(128000)
        "128,000"
        >>> format_token_count(1234)
        "1,234"
    """
    return f"{count:,}"


def load_json_or_yaml(file_path: str | Path) -> dict[str, Any]:
    """
    从文件加载 JSON 或 YAML 数据。

    根据文件扩展名自动判断格式。

    参数:
        file_path: 文件路径

    返回:
        解析后的字典

    异常:
        FileNotFoundError: 文件不存在
        ValueError: 文件格式无效
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")

    try:
        content = path.read_text(encoding="utf-8")
    except Exception as e:
        raise ValueError(f"无法读取文件 {path}: {e}") from e

    # 判断文件类型
    suffix = path.suffix.lower()

    try:
        if suffix in (".json",):
            return json.loads(content)
        elif suffix in (".yaml", ".yml"):
            data = yaml.safe_load(content)
            if not isinstance(data, dict):
                raise ValueError(f"YAML 文件根元素必须是字典，实际为 {type(data).__name__}")
            return data
        else:
            # 尝试 JSON，失败后尝试 YAML
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                data = yaml.safe_load(content)
                if not isinstance(data, dict):
                    raise ValueError(f"无法解析文件格式：{suffix}") from None
                return data
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 格式错误：{e}") from e
    except yaml.YAMLError as e:
        raise ValueError(f"YAML 格式错误：{e}") from e


def create_forge_from_options(
    model: str = "gpt-4o",
    policy_path: str | None = None,
    max_context_tokens: int | None = None,
    debug: bool = False,
) -> ContextForge:
    """
    根据 CLI 参数创建 ContextForge 实例。

    # [DX Decision] 集中处理 ContextForge 初始化逻辑，
    # 避免在各个子命令中重复代码。

    参数:
        model: 模型名称或别名
        policy_path: 策略文件路径
        max_context_tokens: 最大上下文 Token 数（覆盖策略配置）
        debug: 是否启用调试模式

    返回:
        ContextForge 实例

    异常:
        ContextForgeError: 初始化失败
    """
    try:
        return ContextForge(
            model=model,
            policy_path=Path(policy_path) if policy_path else None,
            max_context_tokens=max_context_tokens,
            debug=debug,
        )
    except ContextForgeError:
        # 重新抛出，由调用方处理
        raise
    except Exception as e:
        # 包装为 ContextForgeError
        from context_forge.errors import ConfigValidationError
        raise ConfigValidationError(
            what="创建 ContextForge 实例失败。",
            why=str(e),
            how="请检查模型名称、策略文件路径和参数是否正确。",
        ) from e


def create_summary_panel(
    title: str,
    content: dict[str, Any],
    border_style: str = "blue",
) -> Panel:
    """
    创建摘要信息面板。

    参数:
        title: 面板标题
        content: 内容字典
        border_style: 边框样式

    返回:
        Rich Panel 对象
    """
    lines = []
    for key, value in content.items():
        if isinstance(value, (int, float)) and key.lower().find("token") != -1:
            # Token 数量格式化
            value = format_token_count(int(value))
        lines.append(f"[bold]{key}:[/bold] {value}")

    return Panel(
        "\n".join(lines),
        title=title,
        border_style=border_style,
        expand=False,
    )


def create_segment_table(segments: list[dict[str, Any]]) -> Table:
    """
    创建 Segment 列表表格。

    参数:
        segments: Segment 字典列表

    返回:
        Rich Table 对象
    """
    table = Table(title="Segments", show_header=True, header_style="bold magenta")
    table.add_column("ID", style="dim", width=12)
    table.add_column("类型", style="cyan", width=12)
    table.add_column("角色", style="green", width=10)
    table.add_column("优先级", style="yellow", width=10)
    table.add_column("Token", justify="right", style="blue")
    table.add_column("内容预览", style="white", max_width=50)

    for seg in segments:
        table.add_row(
            seg.get("id", "")[:12],
            seg.get("type", ""),
            seg.get("role", ""),
            seg.get("priority", ""),
            format_token_count(seg.get("token_count", 0)),
            seg.get("content_preview", ""),
        )

    return table


def create_budget_table(budget: dict[str, Any]) -> Table:
    """
    创建预算分配表格。

    参数:
        budget: 预算分配字典

    返回:
        Rich Table 对象
    """
    table = Table(title="预算分配", show_header=True, header_style="bold cyan")
    table.add_column("项目", style="white")
    table.add_column("Token 数", justify="right", style="blue")
    table.add_column("百分比", justify="right", style="yellow")

    total = budget.get("total_budget", 0)

    # 添加核心指标
    content_budget = budget.get("content_budget", 0)
    total_used = budget.get("total_used", 0)
    remaining = total - total_used

    def _pct(value: int | float) -> float:
        return value / total * 100 if total > 0 else 0

    rows = [
        ("总预算", budget.get("total_budget", 0), 100.0),
        ("内容预算", content_budget, _pct(content_budget)),
        ("已使用", total_used, _pct(total_used)),
        ("剩余", remaining, _pct(remaining)),
    ]

    for name, tokens, percent in rows:
        table.add_row(
            name,
            format_token_count(tokens),
            f"{percent:.1f}%",
        )

    return table


def create_audit_tree(audit_log: list[dict[str, Any]]) -> Tree:
    """
    创建审计日志树形视图。

    参数:
        audit_log: 审计记录列表

    返回:
        Rich Tree 对象
    """
    tree = Tree("[bold]审计日志[/bold]")

    # 按决策类型分组
    by_decision: dict[str, list[dict[str, Any]]] = {}
    for entry in audit_log:
        decision = entry.get("decision", "UNKNOWN")
        if decision not in by_decision:
            by_decision[decision] = []
        by_decision[decision].append(entry)

    for decision, entries in sorted(by_decision.items()):
        decision_branch = tree.add(f"[bold cyan]{decision}[/bold cyan] ({len(entries)} 条)")
        for entry in entries[:10]:  # 最多显示前 10 条
            segment_id = entry.get("segment_id", "")[:12]
            reason = entry.get("reason_code", "")
            detail = entry.get("reason_detail", "")
            decision_branch.add(f"[dim]{segment_id}[/dim] {reason}: {detail}")
        if len(entries) > 10:
            decision_branch.add(f"[dim]... 还有 {len(entries) - 10} 条[/dim]")

    return tree


def handle_context_forge_error(error: ContextForgeError) -> NoReturn:
    """
    统一处理 ContextForgeError 异常。

    # [DX Decision] 三段式错误信息：What / Why / How
    # 直接显示 full_message，无需重新格式化

    参数:
        error: ContextForgeError 异常
    """
    console = create_console()
    console.print("\n[bold red]X 错误[/bold red]\n")
    console.print(error.full_message)
    sys.exit(1)
