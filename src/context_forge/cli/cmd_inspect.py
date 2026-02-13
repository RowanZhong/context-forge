"""
inspect 命令 — 查看快照或构建结果。

→ 6.5.1 Context Snapshot

支持两种模式：
1. 快照模式：从 snapshot_dir 加载指定 ID 的快照
2. 文件模式：直接加载 JSON 快照文件
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from context_forge.cli.utils import (
    create_audit_tree,
    create_budget_table,
    create_console,
    create_segment_table,
    create_summary_panel,
    format_token_count,
    load_json_or_yaml,
    print_error,
)

console = create_console()


def inspect_command(
    snapshot_id_or_file: str = typer.Argument(
        ...,
        help="快照 ID（如 req_abc123）或快照文件路径",
    ),
    format: str = typer.Option(
        "rich",
        "--format",
        "-f",
        help="输出格式：rich（Rich 面板）/ json（原始 JSON）/ text（纯文本摘要）",
    ),
    snapshot_dir: str = typer.Option(
        ".context_forge/snapshots",
        "--snapshot-dir",
        "-d",
        help="快照目录路径（快照模式）",
    ),
    show_audit: bool = typer.Option(
        False,
        "--audit",
        "-a",
        help="显示完整审计日志",
    ),
    show_content: bool = typer.Option(
        False,
        "--content",
        "-c",
        help="显示 Segment 完整内容（而非预览）",
    ),
) -> None:
    """
    查看快照或构建结果的详细信息。

    支持两种模式：
    1. 快照 ID 模式：inspect req_abc123
    2. 文件模式：inspect snapshots/req_abc123.json

    默认显示摘要信息，使用 --audit 查看完整审计日志，
    使用 --content 查看 Segment 完整内容。
    """
    # 判断是快照 ID 还是文件路径
    if Path(snapshot_id_or_file).exists():
        # 文件模式
        snapshot_data = _load_snapshot_from_file(snapshot_id_or_file)
    elif snapshot_id_or_file.startswith("req_") or snapshot_id_or_file.startswith("snap_"):
        # 快照 ID 模式
        snapshot_data = _load_snapshot_from_id(snapshot_id_or_file, snapshot_dir)
    else:
        # 尝试文件模式
        file_path = Path(snapshot_id_or_file)
        if file_path.exists():
            snapshot_data = _load_snapshot_from_file(str(file_path))
        else:
            print_error(
                f"无法找到快照：{snapshot_id_or_file}\n"
                f"请检查快照 ID 是否正确，或确认文件路径存在。\n"
                f"快照目录：{snapshot_dir}"
            )

    # 输出结果
    if format == "json":
        _output_json(snapshot_data)
    elif format == "text":
        _output_text(snapshot_data)
    elif format == "rich":
        _output_rich(snapshot_data, show_audit, show_content)
    else:
        print_error(f"不支持的输出格式：{format}")


def _load_snapshot_from_file(file_path: str) -> dict[str, Any]:
    """从文件加载快照。"""
    try:
        data = load_json_or_yaml(file_path)
        # 如果是嵌套格式（observability 模块保存的格式），提取 package 部分
        if "package" in data and "metadata" in data:
            # 合并 metadata 到 package 中
            package = data["package"]
            snapshot_metadata = data.get("metadata", {})
            # 确保有基础字段
            if "request_id" not in package and "request_id" in snapshot_metadata:
                package["request_id"] = snapshot_metadata["request_id"]
            if "model" not in package and "model" in snapshot_metadata:
                package["model"] = snapshot_metadata["model"]
            if "policy_version" not in package and "policy_version" in snapshot_metadata:
                package["policy_version"] = snapshot_metadata["policy_version"]
            if "created_at" not in package and "created_at" in snapshot_metadata:
                package["created_at"] = snapshot_metadata["created_at"]
            data = package

        # 如果缺少 token_usage，从 segments 计算
        if "token_usage" not in data and "segments" in data:
            data["token_usage"] = _calculate_token_usage(data["segments"])

        return data
    except Exception as e:
        print_error(f"加载快照文件失败：{e}")


def _calculate_token_usage(segments: list[dict[str, Any]]) -> dict[str, Any]:
    """从 segments 计算 token_usage。"""
    total_tokens = 0
    by_role: dict[str, int] = {}
    by_type: dict[str, int] = {}

    for seg in segments:
        tokens = seg.get("token_count", 0)
        total_tokens += tokens

        role = seg.get("role", "unknown")
        by_role[role] = by_role.get(role, 0) + tokens

        seg_type = seg.get("type", "unknown")
        by_type[seg_type] = by_type.get(seg_type, 0) + tokens

    return {
        "total_tokens": total_tokens,
        "by_role": by_role,
        "by_type": by_type,
        "segment_count": len(segments),
    }


def _load_snapshot_from_id(snapshot_id: str, snapshot_dir: str) -> dict[str, Any]:
    """从快照 ID 加载快照。"""
    dir_path = Path(snapshot_dir)
    if not dir_path.exists():
        print_error(
            f"快照目录不存在：{snapshot_dir}\n"
            f"请确认项目已初始化（运行 context-forge init）。"
        )

    # 尝试多种文件名格式
    possible_files = [
        dir_path / f"{snapshot_id}.json",
        dir_path / f"{snapshot_id}.yaml",
        dir_path / f"snap_{snapshot_id}.json",
    ]

    for file_path in possible_files:
        if file_path.exists():
            return _load_snapshot_from_file(str(file_path))

    print_error(
        f"快照文件不存在：{snapshot_id}\n"
        f"已尝试以下路径：\n" +
        "\n".join(f"  - {f}" for f in possible_files)
    )


def _output_json(snapshot_data: dict[str, Any]) -> None:
    """输出原始 JSON。"""
    json_text = json.dumps(snapshot_data, ensure_ascii=False, indent=2, default=str)
    console.print(json_text)


def _output_text(snapshot_data: dict[str, Any]) -> None:
    """输出纯文本摘要。"""
    lines = [
        f"═══ 快照 [{snapshot_data.get('request_id', 'N/A')}] ═══",
        f"模型: {snapshot_data.get('model', 'N/A')}",
        f"策略版本: {snapshot_data.get('policy_version', 'N/A')}",
        f"组装耗时: {snapshot_data.get('assembly_duration_ms', 0):.1f}ms",
        "",
        "── Segment 统计 ──",
    ]

    token_usage = snapshot_data.get("token_usage", {})
    lines.append(f"总数: {token_usage.get('segment_count', 0)}")
    lines.append(f"总 Token: {format_token_count(token_usage.get('total_tokens', 0))}")

    by_role = token_usage.get("by_role", {})
    for role, count in sorted(by_role.items()):
        lines.append(f"  {role}: {format_token_count(count)} tokens")

    warnings = snapshot_data.get("warnings", [])
    if warnings:
        lines.append("")
        lines.append(f"── 警告（{len(warnings)} 条）──")
        for w in warnings:
            lines.append(f"  ! {w}")

    console.print("\n".join(lines))


def _output_rich(snapshot_data: dict[str, Any], show_audit: bool, show_content: bool) -> None:
    """输出 Rich 美化格式。"""
    from rich.panel import Panel

    # 1. 摘要面板
    token_usage = snapshot_data.get("token_usage", {})
    summary_content = {
        "请求 ID": snapshot_data.get("request_id", "N/A"),
        "模型": snapshot_data.get("model", "N/A"),
        "策略版本": snapshot_data.get("policy_version", "N/A"),
        "创建时间": snapshot_data.get("created_at", "N/A"),
        "组装耗时": f"{snapshot_data.get('assembly_duration_ms', 0):.1f} ms",
        "Segment 总数": token_usage.get("segment_count", 0),
        "总 Token": token_usage.get("total_tokens", 0),
    }
    summary_panel = create_summary_panel("快照摘要", summary_content, border_style="blue")

    console.print("\n")
    console.print(summary_panel)
    console.print("\n")

    # 2. Segment 表格
    segments = snapshot_data.get("segments", [])
    if show_content:
        # 显示完整内容，需要重新构建表格
        from rich.table import Table
        table = Table(title="Segments（完整内容）", show_header=True, header_style="bold magenta")
        table.add_column("ID", style="dim", width=12)
        table.add_column("类型", style="cyan", width=12)
        table.add_column("角色", style="green", width=10)
        table.add_column("Token", justify="right", style="blue", width=10)
        table.add_column("内容", style="white")

        for seg in segments:
            # 尝试加载完整内容（如果快照中有的话）
            content = seg.get("content", seg.get("content_preview", ""))
            table.add_row(
                seg.get("id", "")[:12],
                seg.get("type", ""),
                seg.get("role", ""),
                format_token_count(seg.get("token_count", 0)),
                content,
            )
        console.print(table)
    else:
        # 显示预览
        segment_table = create_segment_table(segments)
        console.print(segment_table)

    console.print("\n")

    # 3. 预算表格
    budget = snapshot_data.get("budget")
    if budget:
        budget_table = create_budget_table(budget)
        console.print(budget_table)
        console.print("\n")

    # 4. 警告
    warnings = snapshot_data.get("warnings", [])
    if warnings:
        warning_panel = Panel(
            "\n".join(f"! {w}" for w in warnings),
            title=f"警告（{len(warnings)} 条）",
            border_style="yellow",
        )
        console.print(warning_panel)
        console.print("\n")

    # 5. 审计日志（可选）
    if show_audit:
        audit_log = snapshot_data.get("audit_log", [])
        if audit_log:
            audit_tree = create_audit_tree(audit_log)
            console.print(audit_tree)
            console.print("\n")

    # 6. Token 使用分布
    by_role = token_usage.get("by_role", {})
    total_tokens = token_usage.get("total_tokens", 0)
    if by_role:
        console.print("[bold]Token 使用分布（按角色）：[/bold]")
        for role, count in sorted(by_role.items(), key=lambda x: -x[1]):
            percentage = count / total_tokens * 100 if total_tokens > 0 else 0
            console.print(f"  [cyan]{role:12}[/cyan]: {format_token_count(count):>10} ({percentage:.1f}%)")
        console.print("\n")
