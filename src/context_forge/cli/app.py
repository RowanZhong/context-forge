"""
Context Forge CLI — 命令行工具入口。

→ 6.1.2.2 CLI 工具链设计

提供 init / build / inspect / diff / validate / serve 子命令。

用法::

    context-forge --help
    context-forge init
    context-forge build --input request.json
    context-forge inspect snap_abc123
    context-forge diff snap_1 snap_2
    context-forge validate configs/
    context-forge serve  # 第三轮实现
"""

from __future__ import annotations

import typer

from context_forge.cli.utils import create_console

# 创建主应用
app = typer.Typer(
    name="context-forge",
    help="Context Forge — 高性能动态上下文组装引擎 CLI",
    add_completion=False,
    no_args_is_help=True,
)

console = create_console()


# ============================================================
# 子命令注册
# ============================================================

@app.command(name="init")
def init(
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="强制覆盖已存在的文件",
    ),
) -> None:
    """在当前目录生成默认配置文件和示例代码。"""
    from context_forge.cli.cmd_init import init_command
    init_command(force=force)


@app.command(name="validate")
def validate(
    path: str = typer.Argument(
        "context_forge.yaml",
        help="YAML 策略文件路径或输入文件路径",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="严格模式：将警告视为错误",
    ),
    check_antipatterns: bool = typer.Option(
        False,
        "--check-antipatterns",
        help="检测反模式（需要第四轮实现的反模式检测器）",
    ),
) -> None:
    """校验 YAML 策略文件或输入文件的语法和语义正确性。"""
    from context_forge.cli.cmd_validate import validate_command
    validate_command(path=path, strict=strict, check_antipatterns=check_antipatterns)


@app.command(name="build")
def build(
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
        help="检测反模式（需要第四轮实现）",
    ),
) -> None:
    """从 JSON/YAML 文件构建上下文。"""
    from context_forge.cli.cmd_build import build_command
    build_command(
        input_file=input_file,
        model=model,
        policy=policy,
        output=output,
        format=format,
        verbose=verbose,
        snapshot=snapshot,
        check_antipatterns=check_antipatterns,
    )


@app.command(name="inspect")
def inspect(
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
    """查看快照或构建结果的详细信息。"""
    from context_forge.cli.cmd_inspect import inspect_command
    inspect_command(
        snapshot_id_or_file=snapshot_id_or_file,
        format=format,
        snapshot_dir=snapshot_dir,
        show_audit=show_audit,
        show_content=show_content,
    )


@app.command(name="diff")
def diff(
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
    """比对两个快照或构建结果的差异。"""
    from context_forge.cli.cmd_diff import diff_command
    diff_command(
        id_or_file_1=id_or_file_1,
        id_or_file_2=id_or_file_2,
        format=format,
        snapshot_dir=snapshot_dir,
        ignore_timestamps=ignore_timestamps,
    )


@app.command(name="serve")
def serve(
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        "-h",
        help="监听地址",
    ),
    port: int = typer.Option(
        8000,
        "--port",
        "-p",
        help="监听端口",
    ),
    model: str = typer.Option(
        "gpt-4o",
        "--model",
        "-m",
        help="默认模型",
    ),
    policy: str | None = typer.Option(
        None,
        "--policy",
        help="策略文件路径",
    ),
    cors: bool = typer.Option(
        False,
        "--cors",
        help="启用 CORS（跨域资源共享）",
    ),
    reload: bool = typer.Option(
        False,
        "--reload",
        help="启用热重载（开发模式）",
    ),
) -> None:
    """启动 HTTP API 服务器。"""
    from context_forge.cli.cmd_serve import serve_command
    serve_command(
        host=host,
        port=port,
        model=model,
        policy=policy,
        cors=cors,
        reload=reload,
    )


@app.command(name="version")
def version() -> None:
    """显示版本信息。"""
    from context_forge import __version__
    console.print(f"Context Forge v{__version__}")


# ============================================================
# CLI 入口点
# ============================================================

def main() -> None:
    """CLI 入口点。"""
    app()


if __name__ == "__main__":
    main()
