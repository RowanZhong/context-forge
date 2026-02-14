"""
validate 命令 — 校验策略文件和输入文件。

→ 6.1.2.2 Policy-as-Code 校验

支持：
- YAML 策略文件校验
- JSON/YAML 输入文件校验
- 反模式检测（可选）
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from context_forge.cli.utils import create_console, print_error, print_success
from context_forge.config.loader import validate_policy_file

console = create_console()


def validate_command(
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
    """
    校验 YAML 策略文件或输入文件的语法和语义正确性。

    支持的文件类型：
    - YAML 策略文件（context_forge.yaml）
    - JSON/YAML 输入文件（包含 messages、rag_chunks 等）

    使用 --strict 可将警告也视为错误（CI 流程中推荐）。
    使用 --check-antipatterns 可检测常见的反模式。
    """
    path_obj = Path(path)

    if not path_obj.exists():
        print_error(f"文件不存在：{path}")

    # 判断文件类型
    suffix = path_obj.suffix.lower()

    if suffix in (".yaml", ".yml"):
        # 策略文件校验
        _validate_policy(path, strict, check_antipatterns)
    elif suffix in (".json",):
        # 输入文件校验
        _validate_input_file(path, strict)
    else:
        # 尝试按策略文件处理
        console.print(f"[yellow]未知文件类型 {suffix}，尝试按策略文件校验...[/yellow]")
        _validate_policy(path, strict, check_antipatterns)


def _validate_policy(path: str, strict: bool, check_antipatterns: bool) -> None:
    """校验策略文件。"""
    from rich.panel import Panel

    console.print(f"[bold]校验策略文件：[/bold] {path}\n")

    errors = validate_policy_file(path)
    warnings: list[str] = []

    # 收集警告（从错误信息中提取 WARNING 标记）
    actual_errors = []
    for err in errors:
        if "WARNING" in err or "警告" in err:
            warnings.append(err)
        else:
            actual_errors.append(err)

    # 反模式检测（占位，第四轮实现）
    if check_antipatterns:
        console.print("[dim]正在检测反模式...[/dim]")
        antipattern_warnings = _check_antipatterns_placeholder(path)
        warnings.extend(antipattern_warnings)

    # 输出结果
    if actual_errors:
        console.print(Panel(
            "\n".join(f"[red]X[/red] {err}" for err in actual_errors),
            title=f"[bold red]校验失败（{len(actual_errors)} 个错误）[/bold red]",
            border_style="red",
        ))
        sys.exit(1)

    if warnings:
        console.print(Panel(
            "\n".join(f"[yellow]![/yellow] {w}" for w in warnings),
            title=f"[bold yellow]警告（{len(warnings)} 条）[/bold yellow]",
            border_style="yellow",
        ))
        if strict:
            console.print("\n[bold red]严格模式下警告视为错误。[/bold red]")
            sys.exit(1)

    if not actual_errors:
        print_success(f"{path} 校验通过")
        if warnings and not strict:
            console.print(f"[dim]（有 {len(warnings)} 条警告，但不影响使用）[/dim]")


def _validate_input_file(path: str, strict: bool) -> None:
    """校验输入文件。"""
    from pydantic import BaseModel, Field, ValidationError
    from rich.panel import Panel

    console.print(f"[bold]校验输入文件：[/bold] {path}\n")

    # 定义输入文件的 Pydantic 模型
    class InputFileSchema(BaseModel):
        """输入文件校验 Schema。"""
        system_prompt: str | None = Field(default=None, description="系统提示")
        messages: list[dict[str, str]] | None = Field(default=None, description="对话历史")
        rag_chunks: list[dict[str, str | float]] | None = Field(
            default=None, description="RAG 片段"
        )
        tools: list[dict[str, str]] | None = Field(default=None, description="工具定义")
        few_shot_examples: list[dict[str, str]] | None = Field(
            default=None, description="少样本示例"
        )
        state: dict[str, str] | None = Field(default=None, description="状态锚点")

    try:
        from context_forge.cli.utils import load_json_or_yaml

        data = load_json_or_yaml(path)

        # 使用 Pydantic 校验
        InputFileSchema(**data)

        # 额外检查
        warnings: list[str] = []

        if not data.get("system_prompt") and not data.get("messages"):
            warnings.append("缺少 system_prompt 和 messages，可能无法生成有效上下文")

        if data.get("messages"):
            for i, msg in enumerate(data["messages"]):
                if "role" not in msg or "content" not in msg:
                    warnings.append(f"messages[{i}] 缺少 role 或 content 字段")

        if data.get("rag_chunks"):
            for i, chunk in enumerate(data["rag_chunks"]):
                if "content" not in chunk:
                    warnings.append(f"rag_chunks[{i}] 缺少 content 字段")

        # 输出结果
        if warnings:
            console.print(Panel(
                "\n".join(f"[yellow]⚠[/yellow] {w}" for w in warnings),
                title=f"[bold yellow]警告（{len(warnings)} 条）[/bold yellow]",
                border_style="yellow",
            ))
            if strict:
                console.print("\n[bold red]严格模式下警告视为错误。[/bold red]")
                sys.exit(1)

        print_success(f"{path} 校验通过")
        if warnings and not strict:
            console.print(f"[dim]（有 {len(warnings)} 条警告，但不影响使用）[/dim]")

    except ValidationError as e:
        console.print(Panel(
            "\n".join(
                f"[red]X[/red] 字段 "
                f"'{'.'.join(str(loc) for loc in err['loc'])}'"
                f": {err['msg']}"
                for err in e.errors()
            ),
            title="[bold red]校验失败[/bold red]",
            border_style="red",
        ))
        sys.exit(1)
    except Exception as e:
        print_error(f"校验失败：{e}")


def _check_antipatterns_placeholder(path: str) -> list[str]:
    """
    反模式检测占位函数。

    # 🏭 第四轮实现：完整的反模式检测器
    # → 6.7 反模式检测与最佳实践

    当前返回空列表。
    """
    # 占位：在第四轮实现时，调用 AntiPatternDetector
    # from context_forge.antipatterns import AntiPatternDetector
    # detector = AntiPatternDetector()
    # return detector.detect_from_file(path)
    return []
