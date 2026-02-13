"""
diff 命令 — 比对两个快照或构建结果。

→ 6.5.2 Prompt Diff

支持：
- 快照 ID 比对
- 文件比对
- 混合模式（ID + 文件）
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from rich.panel import Panel
from rich.table import Table

from context_forge.cli.utils import (
    create_console,
    print_error,
)

console = create_console()


def diff_command(
    id_or_file_1: str = typer.Argument(
        ...,
        help="第一个快照 ID 或文件路径",
    ),
    id_or_file_2: str = typer.Argument(
        ...,
        help="第二个快照 ID 或文件路径",
    ),
    format: str = typer.Option(
        "rich",
        "--format",
        "-f",
        help="输出格式：rich（彩色差异）/ text（纯文本）/ json（结构化差异）",
    ),
    snapshot_dir: str = typer.Option(
        ".context_forge/snapshots",
        "--snapshot-dir",
        "-d",
        help="快照目录路径",
    ),
    ignore_timestamps: bool = typer.Option(
        True,
        "--ignore-timestamps/--include-timestamps",
        help="忽略时间戳差异（默认忽略）",
    ),
) -> None:
    """
    比对两个快照或构建结果的差异。

    支持的输入格式：
    - 快照 ID：req_abc123 req_def456
    - 文件路径：snapshot1.json snapshot2.json
    - 混合：req_abc123 snapshot2.json

    差异包括：
    - Segment 变更（新增、删除、修改）
    - 预算分配变化
    - Token 使用变化
    - 警告和审计日志差异
    """
    # 加载两个快照
    snapshot1 = _load_snapshot(id_or_file_1, snapshot_dir)
    snapshot2 = _load_snapshot(id_or_file_2, snapshot_dir)

    # 计算差异
    diff_result = _compute_diff(snapshot1, snapshot2, ignore_timestamps)

    # 输出结果
    if format == "json":
        _output_json_diff(diff_result)
    elif format == "text":
        _output_text_diff(diff_result, snapshot1, snapshot2)
    elif format == "rich":
        _output_rich_diff(diff_result, snapshot1, snapshot2)
    else:
        print_error(f"不支持的输出格式：{format}")


def _load_snapshot(id_or_file: str, snapshot_dir: str) -> dict[str, Any]:
    """加载快照（自动判断是 ID 还是文件路径）。"""
    # 复用 inspect 命令的加载逻辑
    from context_forge.cli.cmd_inspect import _load_snapshot_from_file

    # 判断是文件路径还是快照 ID
    if Path(id_or_file).exists():
        return _load_snapshot_from_file(id_or_file)
    else:
        # 快照 ID 模式
        dir_path = Path(snapshot_dir)
        if not dir_path.exists():
            print_error(f"快照目录不存在：{snapshot_dir}")

        possible_files = [
            dir_path / f"{id_or_file}.json",
            dir_path / f"{id_or_file}.yaml",
            dir_path / f"snap_{id_or_file}.json",
        ]

        for file_path in possible_files:
            if file_path.exists():
                return _load_snapshot_from_file(str(file_path))

        print_error(
            f"快照文件不存在：{id_or_file}\n"
            f"已尝试以下路径：\n" +
            "\n".join(f"  - {f}" for f in possible_files)
        )


def _compute_diff(
    snapshot1: dict[str, Any],
    snapshot2: dict[str, Any],
    ignore_timestamps: bool,
) -> dict[str, Any]:
    """
    计算两个快照的差异。

    # [Design Decision] 使用简单的基于 ID 的 diff 算法，
    # 生产环境中可使用更精确的语义 diff（Myers diff、LCS 等）。

    返回差异结构：
    {
        "segments_added": [...],
        "segments_removed": [...],
        "segments_modified": [...],
        "budget_diff": {...},
        "token_diff": {...},
        "warnings_diff": {...},
    }
    """
    diff: dict[str, Any] = {
        "segments_added": [],
        "segments_removed": [],
        "segments_modified": [],
        "budget_diff": {},
        "token_diff": {},
        "warnings_diff": {},
        "metadata_diff": {},
    }

    # 1. Segment 差异
    segs1 = {s["id"]: s for s in snapshot1.get("segments", [])}
    segs2 = {s["id"]: s for s in snapshot2.get("segments", [])}

    # 新增
    for seg_id, seg in segs2.items():
        if seg_id not in segs1:
            diff["segments_added"].append(seg)

    # 删除
    for seg_id, seg in segs1.items():
        if seg_id not in segs2:
            diff["segments_removed"].append(seg)

    # 修改
    for seg_id in segs1:
        if seg_id in segs2:
            if segs1[seg_id] != segs2[seg_id]:
                diff["segments_modified"].append({
                    "id": seg_id,
                    "before": segs1[seg_id],
                    "after": segs2[seg_id],
                })

    # 2. 预算差异
    budget1 = snapshot1.get("budget", {})
    budget2 = snapshot2.get("budget", {})
    if budget1 and budget2:
        diff["budget_diff"] = {
            "total_budget": budget2.get("total_budget", 0) - budget1.get("total_budget", 0),
            "content_budget": budget2.get("content_budget", 0) - budget1.get("content_budget", 0),
            "total_used": budget2.get("total_used", 0) - budget1.get("total_used", 0),
        }

    # 3. Token 使用差异
    usage1 = snapshot1.get("token_usage", {})
    usage2 = snapshot2.get("token_usage", {})
    diff["token_diff"] = {
        "total_tokens": usage2.get("total_tokens", 0) - usage1.get("total_tokens", 0),
        "segment_count": usage2.get("segment_count", 0) - usage1.get("segment_count", 0),
    }

    # 4. 警告差异
    warnings1 = set(snapshot1.get("warnings", []))
    warnings2 = set(snapshot2.get("warnings", []))
    diff["warnings_diff"] = {
        "added": list(warnings2 - warnings1),
        "removed": list(warnings1 - warnings2),
    }

    # 5. 元数据差异（如果不忽略时间戳）
    if not ignore_timestamps:
        diff["metadata_diff"]["created_at"] = {
            "before": snapshot1.get("created_at", "N/A"),
            "after": snapshot2.get("created_at", "N/A"),
        }

    diff["metadata_diff"]["model"] = {
        "before": snapshot1.get("model", "N/A"),
        "after": snapshot2.get("model", "N/A"),
    }
    diff["metadata_diff"]["policy_version"] = {
        "before": snapshot1.get("policy_version", "N/A"),
        "after": snapshot2.get("policy_version", "N/A"),
    }

    return diff


def _output_json_diff(diff_result: dict[str, Any]) -> None:
    """输出 JSON 格式差异。"""
    import json
    console.print(json.dumps(diff_result, ensure_ascii=False, indent=2, default=str))


def _output_text_diff(
    diff_result: dict[str, Any],
    snapshot1: dict[str, Any],
    snapshot2: dict[str, Any],
) -> None:
    """输出纯文本差异。"""
    lines = [
        "═══ 快照差异 ═══",
        f"快照 1: {snapshot1.get('request_id', 'N/A')}",
        f"快照 2: {snapshot2.get('request_id', 'N/A')}",
        "",
    ]

    # Segment 差异
    added = len(diff_result["segments_added"])
    removed = len(diff_result["segments_removed"])
    modified = len(diff_result["segments_modified"])

    lines.append("── Segment 变更 ──")
    lines.append(f"新增: {added}")
    lines.append(f"删除: {removed}")
    lines.append(f"修改: {modified}")
    lines.append("")

    # Token 差异
    token_diff = diff_result["token_diff"]
    lines.append("── Token 变化 ──")
    lines.append(f"总 Token: {token_diff.get('total_tokens', 0):+,}")
    lines.append(f"Segment 数: {token_diff.get('segment_count', 0):+,}")
    lines.append("")

    # 预算差异
    if diff_result["budget_diff"]:
        budget_diff = diff_result["budget_diff"]
        lines.append("── 预算变化 ──")
        lines.append(f"总预算: {budget_diff.get('total_budget', 0):+,}")
        lines.append(f"已使用: {budget_diff.get('total_used', 0):+,}")
        lines.append("")

    console.print("\n".join(lines))


def _output_rich_diff(
    diff_result: dict[str, Any],
    snapshot1: dict[str, Any],
    snapshot2: dict[str, Any],
) -> None:
    """输出 Rich 彩色差异。"""
    # 摘要面板
    summary_panel = Panel(
        f"[bold]快照 1:[/bold] {snapshot1.get('request_id', 'N/A')} ({snapshot1.get('model', 'N/A')})\n"
        f"[bold]快照 2:[/bold] {snapshot2.get('request_id', 'N/A')} ({snapshot2.get('model', 'N/A')})",
        title="快照比对",
        border_style="blue",
    )
    console.print("\n")
    console.print(summary_panel)
    console.print("\n")

    # Segment 变更表格
    seg_table = Table(title="Segment 变更", show_header=True, header_style="bold")
    seg_table.add_column("变更类型", style="cyan", width=12)
    seg_table.add_column("数量", justify="right", style="yellow", width=10)

    added = len(diff_result["segments_added"])
    removed = len(diff_result["segments_removed"])
    modified = len(diff_result["segments_modified"])

    seg_table.add_row("[green]新增 (ADD)[/green]", f"[green]{added}[/green]")
    seg_table.add_row("[red]删除 (REMOVE)[/red]", f"[red]{removed}[/red]")
    seg_table.add_row("[yellow]修改 (MODIFY)[/yellow]", f"[yellow]{modified}[/yellow]")

    console.print(seg_table)
    console.print("\n")

    # Token 差异表格
    token_diff = diff_result["token_diff"]
    token_table = Table(title="Token 变化", show_header=True, header_style="bold")
    token_table.add_column("指标", style="white")
    token_table.add_column("变化", justify="right", style="cyan")

    total_diff = token_diff.get("total_tokens", 0)
    seg_count_diff = token_diff.get("segment_count", 0)

    total_color = "green" if total_diff <= 0 else "red"
    seg_color = "green" if seg_count_diff <= 0 else "yellow"

    token_table.add_row("总 Token", f"[{total_color}]{total_diff:+,}[/{total_color}]")
    token_table.add_row("Segment 数", f"[{seg_color}]{seg_count_diff:+}[/{seg_color}]")

    console.print(token_table)
    console.print("\n")

    # 预算差异
    if diff_result["budget_diff"]:
        budget_diff = diff_result["budget_diff"]
        budget_table = Table(title="预算变化", show_header=True, header_style="bold")
        budget_table.add_column("项目", style="white")
        budget_table.add_column("变化", justify="right", style="cyan")

        for key, value in budget_diff.items():
            color = "green" if value <= 0 else "red"
            budget_table.add_row(
                key.replace("_", " ").title(),
                f"[{color}]{value:+,}[/{color}]",
            )

        console.print(budget_table)
        console.print("\n")

    # 新增的 Segment 详情
    if diff_result["segments_added"]:
        console.print(f"[bold green]新增的 Segment ({len(diff_result['segments_added'])} 个)：[/bold green]")
        for seg in diff_result["segments_added"][:5]:
            console.print(f"  [green]+[/green] [{seg.get('type', 'N/A')}] {seg.get('content_preview', '')[:60]}...")
        if len(diff_result["segments_added"]) > 5:
            console.print(f"  [dim]... 还有 {len(diff_result['segments_added']) - 5} 个[/dim]")
        console.print("\n")

    # 删除的 Segment 详情
    if diff_result["segments_removed"]:
        console.print(f"[bold red]删除的 Segment ({len(diff_result['segments_removed'])} 个)：[/bold red]")
        for seg in diff_result["segments_removed"][:5]:
            console.print(f"  [red]-[/red] [{seg.get('type', 'N/A')}] {seg.get('content_preview', '')[:60]}...")
        if len(diff_result["segments_removed"]) > 5:
            console.print(f"  [dim]... 还有 {len(diff_result['segments_removed']) - 5} 个[/dim]")
        console.print("\n")

    # 警告差异
    warnings_diff = diff_result["warnings_diff"]
    if warnings_diff["added"] or warnings_diff["removed"]:
        console.print("[bold yellow]警告变化：[/bold yellow]")
        for w in warnings_diff["added"]:
            console.print(f"  [green]+[/green] {w}")
        for w in warnings_diff["removed"]:
            console.print(f"  [red]-[/red] {w}")
        console.print("\n")
