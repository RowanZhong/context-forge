"""
init 命令 — 初始化项目配置。

→ 6.1.2.2 零配置启动

创建完整的项目结构：
- .context_forge/ 目录
- context_forge.yaml 策略文件
- input_example.json 示例输入文件
"""

from __future__ import annotations

import shutil
from pathlib import Path

import typer

from context_forge.cli.utils import create_console, print_error, print_success, print_warning

console = create_console()


def init_command(
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="强制覆盖已存在的文件",
    ),
) -> None:
    """
    在当前目录生成默认配置文件和示例代码。

    创建的文件：
    - context_forge.yaml: 策略配置文件
    - .context_forge/: 工作目录（快照、缓存等）
    - .context_forge/input_example.json: 输入示例文件

    使用 --force 可强制覆盖已存在的文件。
    """
    current_dir = Path.cwd()
    created_files: list[str] = []

    # 1. 创建 .context_forge 目录
    context_dir = current_dir / ".context_forge"
    if not context_dir.exists():
        context_dir.mkdir(parents=True)
        created_files.append(".context_forge/")
    else:
        print_warning(".context_forge/ 目录已存在")

    # 创建子目录
    snapshots_dir = context_dir / "snapshots"
    if not snapshots_dir.exists():
        snapshots_dir.mkdir(parents=True)
        created_files.append(".context_forge/snapshots/")

    # 2. 复制策略文件
    config_dst = current_dir / "context_forge.yaml"
    if config_dst.exists() and not force:
        print_warning("context_forge.yaml 已存在，跳过（使用 --force 可强制覆盖）")
    else:
        # 尝试从包内复制默认配置
        try:
            config_src = (
                Path(__file__).parent.parent.parent.parent
                / "configs"
                / "default_policy.yaml"
            )
            if config_src.exists():
                shutil.copy(config_src, config_dst)
                created_files.append("context_forge.yaml")
            else:
                # 如果找不到包内的默认文件，生成一个最小配置
                _generate_minimal_config(config_dst)
                created_files.append("context_forge.yaml (最小版)")
        except Exception as e:
            print_error(f"创建配置文件失败: {e}")

    # 3. 生成示例输入文件
    example_input = context_dir / "input_example.json"
    if example_input.exists() and not force:
        print_warning(".context_forge/input_example.json 已存在，跳过")
    else:
        _generate_example_input(example_input)
        created_files.append(".context_forge/input_example.json")

    # 4. 生成 .gitignore（如果不存在）
    gitignore = context_dir / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text(
            "# Context Forge 运行时文件\n"
            "snapshots/\n"
            "*.cache\n"
            "*.log\n",
            encoding="utf-8",
        )
        created_files.append(".context_forge/.gitignore")

    # 显示结果
    if created_files:
        print_success("项目初始化完成！已创建以下文件：")
        for f in created_files:
            console.print(f"  [cyan]+ {f}[/cyan]")
    else:
        console.print("[yellow]所有文件均已存在，无需创建。[/yellow]")

    # 下一步提示
    console.print("\n[bold]下一步：[/bold]")
    console.print("  1. 编辑 [cyan]context_forge.yaml[/cyan] 调整策略参数")
    console.print("  2. 查看示例输入格式：[cyan].context_forge/input_example.json[/cyan]")
    console.print("  3. 在代码中使用：")
    console.print("     [dim]from context_forge import ContextForge[/dim]")
    console.print("     [dim]forge = ContextForge(model=\"gpt-4o\")[/dim]")
    console.print("     [dim]context = await forge.build(...)[/dim]")
    console.print("\n  或使用 CLI 命令：")
    console.print("     [dim]context-forge build --input .context_forge/input_example.json[/dim]")


def _generate_minimal_config(path: Path) -> None:
    """生成最小配置文件。"""
    content = """# Context Forge 策略配置
#
# 此文件定义了上下文组装的全部策略参数。
# 详细配置参考：https://github.com/yourorg/context-forge

version: "1.0"
name: "my-project"
description: "我的项目配置"

# 预算分配
budget:
  max_context_tokens: 128000
  output_reserved_tokens: 4096
  thinking_reserved_tokens: 0
  overflow_strategy: "truncate_lowest_priority"

# 清洗策略
sanitize:
  unicode_normalize: true
  strip_html: true
  injection_detection: true
  on_injection: "warn_and_remove"

# 缓存策略
cache:
  enabled: true
  backend: "memory"
  ttl_seconds: 3600

# 可观测性
observability:
  snapshot_enabled: true
  metrics_enabled: true
  snapshot_dir: ".context_forge/snapshots"
"""
    path.write_text(content, encoding="utf-8")


def _generate_example_input(path: Path) -> None:
    """生成示例输入文件。"""
    import json

    example = {
        "_comment": "Context Forge 输入示例文件",
        "_usage": "context-forge build --input input_example.json",
        "system_prompt": "你是一个专业的客服助手，负责回答用户关于产品和服务的问题。",
        "messages": [
            {
                "role": "user",
                "content": "你好，我想咨询一下退货政策。"
            },
            {
                "role": "assistant",
                "content": "您好！我很乐意帮您了解退货政策。请问您是想了解哪类商品的退货规定呢？"
            },
            {
                "role": "user",
                "content": "电子产品，比如手机。"
            }
        ],
        "rag_chunks": [
            {
                "content": (
                    "退货政策：电子产品（包括手机、平板、笔记本电脑）"
                    "在购买后 7 天内可无理由退货。"
                    "退货时需确保商品包装完好、配件齐全、未激活。"
                ),
                "score": 0.95,
                "source_id": "policy_doc_001"
            },
            {
                "content": (
                    "退款流程：1. 在订单详情页点击「申请退货」；"
                    "2. 选择退货原因；"
                    "3. 系统生成退货单号；4. 寄回商品；"
                    "5. 我们收到商品后 3-5 个工作日内完成退款。"
                ),
                "score": 0.87,
                "source_id": "policy_doc_002"
            }
        ],
        "tools": [
            {
                "name": "query_order_status",
                "description": "查询订单状态",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "订单编号"
                        }
                    },
                    "required": ["order_id"]
                }
            }
        ],
        "few_shot_examples": [
            {
                "role": "user",
                "content": "如何查看物流信息？"
            },
            {
                "role": "assistant",
                "content": (
                    "您可以在订单详情页查看物流信息，"
                    "点击「查看物流」按钮即可追踪包裹实时位置。"
                )
            }
        ],
        "state": {
            "user_id": "user_12345",
            "session_id": "sess_abc123",
            "conversation_turn": 3
        }
    }

    path.write_text(
        json.dumps(example, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
