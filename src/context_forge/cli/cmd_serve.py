"""
CLI 命令: serve — 启动 HTTP API 服务器。

→ 6.5.4 HTTP API 层
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def serve_command(
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
    """
    启动 HTTP API 服务器。

    示例:

        # 使用默认配置启动
        context-forge serve

        # 自定义端口和模型
        context-forge serve --port 8080 --model claude-sonnet-4-5

        # 启用 CORS 和热重载（开发模式）
        context-forge serve --cors --reload

    # [DX Decision] 提供合理的默认值（127.0.0.1:8000），
    # 同时允许通过参数灵活配置，满足不同部署场景。
    """
    console.print("\n[bold cyan]Context Forge HTTP API Server[/bold cyan]")
    console.print("[dim]Version: 0.1.0[/dim]\n")

    # 验证策略文件
    policy_path: Path | None = None
    if policy:
        policy_path = Path(policy)
        if not policy_path.exists():
            console.print(f"[red]错误: 策略文件不存在: {policy}[/red]")
            raise typer.Exit(1)

    # 显示配置信息
    config_table = Table(show_header=False, box=None, padding=(0, 2))
    config_table.add_column("Key", style="cyan")
    config_table.add_column("Value", style="yellow")

    config_table.add_row("监听地址", f"{host}:{port}")
    config_table.add_row("默认模型", model)
    config_table.add_row("策略文件", policy or "[dim]使用内置默认策略[/dim]")
    config_table.add_row("CORS", "已启用" if cors else "已禁用")
    config_table.add_row("热重载", "已启用" if reload else "已禁用")

    console.print(Panel(config_table, title="[bold]配置信息[/bold]", border_style="blue"))

    # 显示端点列表
    endpoints_table = Table(show_header=True, box=None)
    endpoints_table.add_column("方法", style="green", width=8)
    endpoints_table.add_column("路径", style="cyan")
    endpoints_table.add_column("说明", style="white")

    endpoints = [
        ("POST", "/build", "执行上下文组装"),
        ("GET", "/snapshots", "列出所有快照"),
        ("GET", "/snapshots/{id}", "查看快照详情"),
        ("POST", "/diff", "比对两个快照"),
        ("GET", "/metrics", "查看系统指标"),
        ("POST", "/antipatterns", "检测反模式"),
        ("POST", "/golden/record", "记录 Golden Case"),
        ("POST", "/golden/verify", "验证 Golden Case"),
        ("GET", "/health", "健康检查"),
        ("GET", "/docs", "OpenAPI 文档（交互式）"),
        ("GET", "/redoc", "ReDoc 文档"),
    ]

    for method, path, description in endpoints:
        endpoints_table.add_row(method, path, description)

    console.print(Panel(endpoints_table, title="[bold]可用端点[/bold]", border_style="green"))

    # 启动提示
    console.print("\n[bold green]服务器正在启动...[/bold green]")
    console.print(f"[dim]访问 http://{host}:{port}/docs 查看交互式 API 文档[/dim]\n")

    # 启动 Uvicorn
    try:
        import uvicorn

        from context_forge.cli.server import create_app

        app = create_app(
            model=model,
            policy_path=str(policy_path) if policy_path else None,
            enable_cors=cors,
        )

        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload,
            log_level="info",
        )

    except KeyboardInterrupt:
        console.print("\n[yellow]服务器已停止[/yellow]")
        raise typer.Exit(0) from None
    except Exception as e:
        console.print(f"\n[red]服务器启动失败: {e}[/red]")
        raise typer.Exit(1) from e
