"""
build 命令 — 从文件构建上下文。

→ 6.1.2 Context Builder Pipeline

从 JSON/YAML 输入文件构建上下文，支持多种输出格式。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import typer
from rich.panel import Panel

from context_forge.antipattern import create_default_detector
from context_forge.cli.utils import (
    create_budget_table,
    create_console,
    create_forge_from_options,
    create_segment_table,
    create_summary_panel,
    format_token_count,
    handle_context_forge_error,
    load_json_or_yaml,
    print_error,
    print_success,
)
from context_forge.errors import ContextForgeError

console = create_console()


def build_command(
    input_file: str = typer.Option(
        ...,
        "--input",
        "-i",
        help="输入文件路径（JSON 或 YAML）",
    ),
    model: str = typer.Option(
        "gpt-4o",
        "--model",
        "-m",
        help="目标模型名称或别名",
    ),
    policy: str | None = typer.Option(
        None,
        "--policy",
        "-p",
        help="策略文件路径（默认自动搜索）",
    ),
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="输出文件路径（不指定则输出到终端）",
    ),
    format: str = typer.Option(
        "rich",
        "--format",
        "-f",
        help="输出格式：text（摘要）/ json（完整快照）/ rich（Rich 面板）",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="详细输出（显示调试信息）",
    ),
    snapshot: bool = typer.Option(
        True,
        "--snapshot/--no-snapshot",
        help="是否保存快照（默认保存）",
    ),
    check_antipatterns: bool = typer.Option(
        False,
        "--check-antipatterns",
        help="检测反模式（10 个规则，CRITICAL/WARNING/INFO 三个级别）",
    ),
) -> None:
    """
    从 JSON/YAML 文件构建上下文。

    输入文件格式示例：
    {
      "system_prompt": "你是一个助手",
      "messages": [{"role": "user", "content": "你好"}],
      "rag_chunks": [{"content": "...", "score": 0.9}],
      "tools": [...],
      "few_shot_examples": [...],
      "state": {...}
    }

    使用 .context_forge/input_example.json 作为参考。

    输出格式：
    - text: 简洁摘要（适合快速查看）
    - json: 完整快照（适合存档和比对）
    - rich: Rich 美化输出（默认，最易读）
    """
    # 加载输入文件
    try:
        input_data = load_json_or_yaml(input_file)
    except Exception as e:
        print_error(f"加载输入文件失败：{e}")

    # 创建 ContextForge 实例
    try:
        forge = create_forge_from_options(
            model=model,
            policy_path=policy,
            debug=verbose,
        )
    except ContextForgeError as e:
        handle_context_forge_error(e)

    # 构建上下文
    # [DX Decision] 简化进度显示，避免 Windows 终端的 Unicode 问题
    if verbose:
        console.print("[dim]正在组装上下文...[/dim]")

    try:
        package = asyncio.run(forge.build(
            system_prompt=input_data.get("system_prompt", ""),
            messages=input_data.get("messages"),
            rag_chunks=input_data.get("rag_chunks"),
            tools=input_data.get("tools"),
            few_shot_examples=input_data.get("few_shot_examples"),
            state=input_data.get("state"),
        ))
    except ContextForgeError as e:
        handle_context_forge_error(e)
    except Exception as e:
        print_error(f"构建上下文失败：{e}")

    # 反模式检测
    if check_antipatterns:
        try:
            detector = create_default_detector()
            results = detector.detect_from_package(package)

            if results:
                # 生成反模式检测报告
                report = detector.format_report(results, format="rich")
                console.print("\n")
                console.print(report)
            else:
                console.print("\n[green]✓ 未检测到反模式[/green]")
        except Exception as e:
            print_error(f"反模式检测失败：{e}")
            if verbose:
                import traceback
                console.print(traceback.format_exc())

    # 输出结果
    if format == "text":
        _output_text(package, output)
    elif format == "json":
        _output_json(package, output)
    elif format == "rich":
        _output_rich(package, output)
    else:
        print_error(f"不支持的输出格式：{format}")

    # 显示快照 ID（如果保存了）
    if snapshot and forge._snapshot_manager:
        console.print(f"\n[dim]快照 ID: {package.request_id}[/dim]")
        console.print(f"[dim]快照路径: {forge._policy.observability.snapshot_dir}/{package.request_id}.json[/dim]")


def _output_text(package: Any, output_path: str | None) -> None:
    """输出纯文本摘要。"""
    text = package.summary()

    if output_path:
        Path(output_path).write_text(text, encoding="utf-8")
        print_success(f"已保存到 {output_path}")
    else:
        console.print(text)


def _output_json(package: Any, output_path: str | None) -> None:
    """输出完整 JSON 快照。"""
    snapshot = package.to_snapshot_dict()
    json_text = json.dumps(snapshot, ensure_ascii=False, indent=2, default=str)

    if output_path:
        Path(output_path).write_text(json_text, encoding="utf-8")
        print_success(f"已保存到 {output_path}")
    else:
        console.print(json_text)


def _output_rich(package: Any, output_path: str | None) -> None:
    """输出 Rich 美化格式。"""
    # 如果指定了输出文件，需要切换到文件 console
    if output_path:
        from rich.console import Console as RichConsole
        file_console = RichConsole(file=open(output_path, "w", encoding="utf-8"), width=120)
        _render_rich_output(package, file_console)
        print_success(f"已保存到 {output_path}")
    else:
        _render_rich_output(package, console)


def _render_rich_output(package: Any, target_console: Any) -> None:
    """渲染 Rich 输出。"""

    # 1. 摘要面板
    usage = package.token_usage
    summary_content = {
        "请求 ID": package.request_id,
        "模型": package.model,
        "策略版本": package.policy_version,
        "组装耗时": f"{package.assembly_duration_ms:.1f} ms",
        "Segment 总数": usage.segment_count,
        "总 Token": usage.total_tokens,
    }
    summary_panel = create_summary_panel("上下文包摘要", summary_content, border_style="green")

    # 2. Segment 表格
    segments_data = package.to_snapshot_dict()["segments"]
    segment_table = create_segment_table(segments_data)

    # 3. 预算表格
    budget_data = package.to_snapshot_dict()["budget"]
    if budget_data:
        budget_table = create_budget_table(budget_data)
    else:
        budget_table = None

    # 4. 警告面板
    warning_panel = None
    if package.warnings:
        warning_panel = Panel(
            "\n".join(f"! {w}" for w in package.warnings),
            title=f"警告（{len(package.warnings)} 条）",
            border_style="yellow",
        )

    # 5. 丢弃的 Segment
    drop_panel = None
    if package.has_drops:
        drops = package.dropped_segments
        drop_lines = [f"• {entry.summary}" for entry in drops[:5]]
        if len(drops) > 5:
            drop_lines.append(f"... 还有 {len(drops) - 5} 条")
        drop_panel = Panel(
            "\n".join(drop_lines),
            title=f"丢弃记录（{len(drops)} 条）",
            border_style="red",
        )

    # 组合输出
    target_console.print("\n")
    target_console.print(summary_panel)
    target_console.print("\n")
    target_console.print(segment_table)
    target_console.print("\n")
    if budget_table:
        target_console.print(budget_table)
        target_console.print("\n")
    if warning_panel:
        target_console.print(warning_panel)
        target_console.print("\n")
    if drop_panel:
        target_console.print(drop_panel)
        target_console.print("\n")

    # Token 使用分布
    if usage.by_role:
        target_console.print("[bold]Token 使用分布（按角色）：[/bold]")
        for role, count in sorted(usage.by_role.items(), key=lambda x: -x[1]):
            percentage = count / usage.total_tokens * 100 if usage.total_tokens > 0 else 0
            target_console.print(f"  [cyan]{role:12}[/cyan]: {format_token_count(count):>10} ({percentage:.1f}%)")
        target_console.print("\n")
