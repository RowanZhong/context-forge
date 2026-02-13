"""
示例代码的共享基础设施。

→ 第五轮交付物 14：六大场景集成示例

提供所有示例共用的工具函数和 Mock 层：
- 命令行参数解析（--mock）
- Mock LLM 响应生成器
- Rich Console 格式化输出
- 测试数据生成器（RAG chunks / 对话历史 / 工具定义）

# [Design Decision] 集中管理 Mock 层，而非在每个示例中重复实现：
# 1. 确保所有示例的 Mock 行为一致
# 2. 无 API Key 时自动降级到 Mock 模式（用户友好）
# 3. Mock 响应应与真实场景相关，而非通用占位符
"""

from __future__ import annotations

import argparse
import os
import random
from datetime import datetime, timedelta, timezone
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from context_forge.models.audit import AuditEntry
from context_forge.models.context_package import ContextPackage
from context_forge.models.metadata import SegmentMetadata
from context_forge.models.provenance import Provenance, SourceType
from context_forge.models.segment import Priority, Segment, SegmentType

# Rich Console 实例（禁用 emoji 避免 Windows 乱码）
console = Console(highlight=False, emoji=False)


class MockLLM:
    """
    内置的 Mock LLM 响应生成器。

    支持场景相关的响应模拟：
    - RAG 问答场景：返回基于检索内容的回答
    - 对话记忆场景：返回记忆式回复
    - 多 Agent 场景：返回协作式响应
    - 安全合规场景：返回合规检查结果

    # [DX Decision] Mock 响应不是通用占位符（如 "This is a mock response"），
    # 而是根据场景上下文生成相关回答，让示例输出更真实、更有教学价值。
    """

    SCENARIO_RESPONSES = {
        "rag": [
            "根据检索到的文档，Python 3.13 已经移除了 GIL，这意味着多线程性能将大幅提升。",
            "从提供的上下文来看，建议使用异步 I/O 处理并发请求，可以参考 asyncio 文档。",
            "综合检索结果，推荐使用 Pydantic v2 进行数据校验，它比 v1 快 5-10 倍。",
        ],
        "conversation": [
            "我记得你之前提到过喜欢 Python，所以我推荐你试试 FastAPI 框架。",
            "根据我们之前的对话，你正在做 LLM 应用开发，Context Forge 应该很适合你。",
            "你刚才问的问题让我想起我们讨论过的 RAG 架构，我可以详细解释一下。",
        ],
        "multi_agent": [
            "Agent A: 我已经完成用户意图识别，结果是技术咨询类问题。",
            "Agent B: 收到，我来检索相关文档。已找到 5 篇匹配文档。",
            "Agent C: 好的，我基于检索结果生成最终回答：...",
        ],
        "security": [
            "安全检查通过：未检测到 Prompt Injection 攻击。",
            "PII 脱敏完成：已自动隐藏 3 个手机号、2 个邮箱地址。",
            "合规审计：所有用户输入均已通过清洗管道，符合零信任原则。",
        ],
        "prompt_version": [
            "Prompt v1.2 测试通过，输出格式符合预期。",
            "与 Golden Set 对比：一致性 95%，无回归问题。",
            "建议：当前 Prompt 在边界 case 上表现良好，可以发布。",
        ],
        "cost_optimization": [
            "路由决策：检测到简单问题，使用 GPT-4o-mini（节省 90% 成本）。",
            "缓存命中：System Prompt 复用，节省 2048 Tokens。",
            "压缩完成：对话历史从 8K 压缩到 2K，保留关键信息。",
        ],
    }

    DEFAULT_RESPONSE = "这是一个 Mock LLM 响应。在生产环境中，这里会是真实的模型输出。"

    @classmethod
    def generate(cls, scenario: str = "default", context_hint: str = "") -> str:
        """
        生成场景相关的 Mock 响应。

        参数:
            scenario: 场景名称（rag / conversation / multi_agent / security / ...）
            context_hint: 上下文提示（用于生成更相关的响应）

        返回:
            Mock LLM 响应文本
        """
        responses = cls.SCENARIO_RESPONSES.get(scenario, [cls.DEFAULT_RESPONSE])
        response = random.choice(responses)

        # 如果有上下文提示，尝试融入响应
        if context_hint and scenario == "rag":
            response = f"基于上下文「{context_hint[:30]}...」，{response}"

        return response

    @classmethod
    async def async_generate(cls, scenario: str = "default", context_hint: str = "") -> str:
        """异步版本的 generate（适配异步示例）。"""
        return cls.generate(scenario, context_hint)

    # 保持旧 API 兼容性
    def __init__(self):
        self.responses = {
            "summarize": "这是一段摘要内容，将原始文本压缩到约 30% 的长度。核心观点已保留，细节被省略。",
            "compress": "压缩后的内容",
            "classify": "safe",
            "route": "gpt-4o-mini",
        }

    async def summarize(self, text: str, max_tokens: int = 100) -> str:
        """生成摘要（Mock）。"""
        return self.responses["summarize"][:max_tokens]

    async def compress(self, text: str, target_ratio: float = 0.3) -> str:
        """压缩文本（Mock）。"""
        target_length = int(len(text) * target_ratio)
        return text[:target_length] + "..."

    async def classify_injection(self, text: str) -> tuple[bool, float]:
        """检测 Prompt Injection（Mock）。"""
        dangerous_patterns = [
            "ignore previous",
            "disregard",
            "system:",
            "<script>",
            "INSERT INTO",
        ]
        is_injection = any(p.lower() in text.lower() for p in dangerous_patterns)
        confidence = 0.9 if is_injection else 0.1
        return is_injection, confidence

    async def route(self, context: dict[str, Any]) -> str:
        """路由决策（Mock）。"""
        total_chars = sum(len(str(v)) for v in context.values())
        if total_chars < 500:
            return "gpt-4o-mini"
        elif total_chars < 2000:
            return "gpt-4o"
        else:
            return "claude-sonnet-4-5"


def section_header(title: str, con: Console | None = None) -> None:
    """
    打印区块标题。

    参数:
        title: 标题文本
        con: Rich Console（None 时使用全局 console）
    """
    if con is None:
        con = console

    con.print()
    con.print(
        Panel(
            Text(title, style="bold cyan", justify="center"),
            border_style="cyan",
            padding=(0, 2),
        )
    )
    con.print()


def section_footer(con: Console | None = None) -> None:
    """
    打印区块分隔符。

    参数:
        con: Rich Console（None 时使用全局 console）
    """
    if con is None:
        con = console

    con.print("\n" + "=" * 60 + "\n", style="dim")


def print_header(title: str, subtitle: str = ""):
    """打印美化的标题（兼容旧版）。"""
    console.print()
    if subtitle:
        console.print(Panel(
            f"[bold cyan]{title}[/bold cyan]\n[dim]{subtitle}[/dim]",
            border_style="cyan",
        ))
    else:
        console.print(Panel(
            f"[bold cyan]{title}[/bold cyan]",
            border_style="cyan",
        ))
    console.print()


def print_section(title: str):
    """打印章节标题（兼容旧版）。"""
    console.print(f"\n[bold yellow]{'='*70}[/bold yellow]")
    console.print(f"[bold yellow]{title}[/bold yellow]")
    console.print(f"[bold yellow]{'='*70}[/bold yellow]\n")


def print_success(message: str):
    """打印成功消息。"""
    # [DX Decision] 使用 ASCII 符号避免 Windows 乱码
    console.print(f"[green][*][/green] {message}")


def print_warning(message: str):
    """打印警告消息。"""
    console.print(f"[yellow][!][/yellow] {message}")


def print_error(message: str):
    """打印错误消息。"""
    console.print(f"[red][X][/red] {message}")


def print_info(message: str):
    """打印信息消息。"""
    console.print(f"[blue][i][/blue] {message}")


def create_comparison_table(
    title: str,
    headers: list[str],
    rows: list[list[str]],
) -> Table:
    """创建对比表格。"""
    table = Table(title=title, show_header=True, header_style="bold magenta")
    for header in headers:
        table.add_column(header)
    for row in rows:
        table.add_row(*row)
    return table


def create_tree(title: str) -> Tree:
    """创建树形结构。"""
    return Tree(f"[bold cyan]{title}[/bold cyan]")


def format_tokens(count: int) -> str:
    """格式化 Token 数。"""
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    elif count >= 1_000:
        return f"{count / 1_000:.1f}K"
    else:
        return str(count)


def format_percentage(value: float) -> str:
    """格式化百分比。"""
    return f"{value * 100:.1f}%"


def format_duration(ms: float) -> str:
    """格式化耗时。"""
    if ms >= 1000:
        return f"{ms / 1000:.2f}s"
    else:
        return f"{ms:.1f}ms"


def print_package_summary(package: ContextPackage, con: Console | None = None) -> None:
    """
    打印 ContextPackage 摘要信息。

    参数:
        package: 组装好的 ContextPackage
        con: Rich Console（None 时使用全局 console）
    """
    if con is None:
        con = console

    # 统计信息表
    table = Table(title="Context Package 摘要", show_header=True, header_style="bold magenta")
    table.add_column("指标", style="cyan", width=30)
    table.add_column("数值", justify="right", style="green")

    table.add_row("Segment 总数", str(len(package.segments)))
    table.add_row("总 Token 数", f"{package.total_tokens:,}")
    table.add_row("Budget 使用率", f"{package.saturation * 100:.1f}%")

    # 按类型统计
    type_counts: dict[str, int] = {}
    for seg in package.segments:
        type_name = seg.type.value
        type_counts[type_name] = type_counts.get(type_name, 0) + 1

    for type_name, count in sorted(type_counts.items()):
        table.add_row(f"  +-- {type_name}", str(count))

    con.print(table)


def print_audit_log(entries: list[AuditEntry], con: Console | None = None) -> None:
    """
    打印审计日志。

    参数:
        entries: 审计条目列表
        con: Rich Console（None 时使用全局 console）
    """
    if con is None:
        con = console

    if not entries:
        con.print("[dim]（无审计记录）[/dim]")
        return

    table = Table(title="审计日志", show_header=True, header_style="bold yellow")
    table.add_column("阶段", style="cyan", width=15)
    table.add_column("决策", style="magenta", width=20)
    table.add_column("原因", style="yellow", width=40)

    for entry in entries[:10]:  # 只显示前 10 条
        stage_name = entry.stage_name or "unknown"
        decision = entry.decision.value if entry.decision else "N/A"
        reason = entry.reason_detail or (entry.reason_code.value if entry.reason_code else "N/A")

        table.add_row(stage_name, decision, str(reason)[:40])

    if len(entries) > 10:
        table.add_row("...", "...", f"(共 {len(entries)} 条记录，仅显示前 10 条)")

    con.print(table)


def generate_rag_chunks(
    n: int = 5,
    topic: str = "Python",
    avg_length: int = 200,
    score_range: tuple[float, float] = (0.6, 0.95),
) -> list[Segment]:
    """
    生成合成 RAG 检索片段。

    参数:
        n: 生成数量
        topic: 主题关键词
        avg_length: 平均长度（字符数）
        score_range: 检索分数范围（最小值，最大值）

    返回:
        RAG Segment 列表
    """
    templates = [
        f"{topic} 是一种广泛使用的编程语言，具有简洁的语法和丰富的生态系统。",
        f"在 {topic} 中，推荐使用类型标注提升代码可维护性，mypy 是常用的类型检查工具。",
        f"{topic} 的异步编程模型基于 asyncio 库，适用于 I/O 密集型任务。",
        f"关于 {topic} 性能优化：避免全局解释器锁（GIL）的影响，可以使用多进程或异步 I/O。",
        f"{topic} 的包管理工具包括 pip、poetry、uv 等，uv 是最新的高性能选项。",
        f"在 {topic} 项目中，建议使用 Pydantic 进行数据校验，它提供了运行时类型检查。",
        f"{topic} 的测试框架包括 pytest、unittest，pytest 是目前最流行的选择。",
        f"关于 {topic} 的最佳实践：遵循 PEP 8 代码风格，使用 ruff 进行 lint 和格式化。",
    ]

    chunks = []
    for i in range(n):
        content = templates[i % len(templates)]
        # 填充到目标长度
        if len(content) < avg_length:
            content += " " + "示例内容填充。" * ((avg_length - len(content)) // 10)

        score = random.uniform(*score_range)

        chunk = Segment(
            type=SegmentType.RAG,
            content=content[:avg_length],
            role="user",
            provenance=Provenance(
                source_type=SourceType.RAG_RETRIEVAL,
                source_id=f"doc_{i:03d}",
                retrieval_score=score,
            ),
            metadata=SegmentMetadata(
                rerank_score=score,
                debug_labels={"chunk_id": f"chunk_{i}", "topic": topic.lower()},
            ),
        )
        chunks.append(chunk)

    return chunks


def generate_conversation_history(
    n_turns: int = 10,
    topic: str = "LLM 应用开发",
) -> list[Segment]:
    """
    生成合成对话历史。

    参数:
        n_turns: 对话轮次
        topic: 对话主题

    返回:
        User + Assistant Segment 交替列表
    """
    user_templates = [
        f"我想了解 {topic} 的最佳实践。",
        f"在 {topic} 中如何处理错误？",
        f"能否推荐一些 {topic} 的工具和库？",
        f"{topic} 的性能优化有哪些技巧？",
        f"我遇到了一个关于 {topic} 的问题...",
    ]

    assistant_templates = [
        f"关于 {topic}，我推荐你从以下几个方面入手：...",
        f"在 {topic} 中，错误处理的最佳实践是使用结构化异常...",
        f"以下是一些常用的 {topic} 工具：...",
        f"{topic} 的性能优化可以从缓存、异步、批处理等方面考虑...",
        "我理解你的问题，让我详细解释一下...",
    ]

    history = []
    base_time = datetime.now(timezone.utc) - timedelta(minutes=n_turns * 2)

    for i in range(n_turns):
        # User message
        user_msg = Segment(
            type=SegmentType.USER,
            content=user_templates[i % len(user_templates)],
            role="user",
            provenance=Provenance(
                source_type=SourceType.USER_INPUT,
                source_id="user_001",
            ),
            created_at=base_time + timedelta(minutes=i * 2),
        )
        history.append(user_msg)

        # Assistant message
        assistant_msg = Segment(
            type=SegmentType.ASSISTANT,
            content=assistant_templates[i % len(assistant_templates)],
            role="assistant",
            provenance=Provenance(
                source_type=SourceType.MANUAL_INJECTION,
                source_id="assistant",
            ),
            created_at=base_time + timedelta(minutes=i * 2 + 1),
        )
        history.append(assistant_msg)

    return history


def generate_tools(n: int = 3) -> list[Segment]:
    """
    生成合成工具定义。

    参数:
        n: 工具数量

    返回:
        Tool Definition Segment 列表
    """
    tool_templates = [
        {
            "name": "search_documents",
            "description": "在知识库中搜索相关文档",
            "parameters": {"query": "string", "max_results": "integer"},
        },
        {
            "name": "get_current_time",
            "description": "获取当前时间",
            "parameters": {},
        },
        {
            "name": "calculate",
            "description": "执行数学计算",
            "parameters": {"expression": "string"},
        },
        {
            "name": "send_email",
            "description": "发送邮件通知",
            "parameters": {"to": "string", "subject": "string", "body": "string"},
        },
        {
            "name": "query_database",
            "description": "查询数据库",
            "parameters": {"sql": "string"},
        },
    ]

    tools = []
    for i in range(min(n, len(tool_templates))):
        tool_def = tool_templates[i]
        content = f"Tool: {tool_def['name']}\nDescription: {tool_def['description']}\nParameters: {tool_def['parameters']}"

        tool = Segment(
            type=SegmentType.TOOL_DEFINITION,
            content=content,
            role="system",
            priority=Priority.HIGH,
            provenance=Provenance(
                source_type=SourceType.SYSTEM_CONFIG,
                source_id=f"tool_{tool_def['name']}",
            ),
        )
        tools.append(tool)

    return tools


def truncate_text(text: str, max_length: int = 60) -> str:
    """截断文本。"""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def generate_fake_timestamps(
    count: int,
    start_days_ago: int = 30,
    end_days_ago: int = 0,
) -> list[datetime]:
    """生成假的时间戳序列。"""
    now = datetime.now()
    start = now - timedelta(days=start_days_ago)
    end = now - timedelta(days=end_days_ago)
    delta = (end - start) / (count - 1) if count > 1 else timedelta(0)
    return [start + delta * i for i in range(count)]


def parse_args(description: str = "Context Forge 示例") -> argparse.Namespace:
    """
    解析示例的命令行参数。

    所有示例支持 --mock 标志强制使用 Mock 模式。
    如果不提供 --mock 但环境变量中无 API Key，会自动降级并提示。

    返回:
        args.mock (bool): 是否使用 Mock 模式
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="使用 Mock LLM 响应（无需 API Key）",
    )
    return parser.parse_args()


def check_api_key(mock: bool) -> bool:
    """
    检查 API Key 是否存在（兼容旧版）。

    参数:
        mock: 是否使用 Mock 模式

    返回:
        是否可以继续运行
    """
    if mock:
        print_info("使用 Mock 模式，无需 API Key")
        return True

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print_warning("未检测到 API Key，自动切换到 Mock 模式")
        return False
    return True


def check_and_warn_mock(args: argparse.Namespace, con: Console | None = None) -> bool:
    """
    检查是否需要 Mock 模式，并打印提示。

    参数:
        args: 命令行参数
        con: Rich Console

    返回:
        是否使用 Mock 模式
    """
    if con is None:
        con = console

    # 显式指定 --mock
    if args.mock:
        con.print("[cyan]已启用 Mock 模式（通过 --mock 标志）[/cyan]\n")
        return True

    # 检查环境变量
    has_api_key = bool(
        os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("GOOGLE_API_KEY")
    )

    if not has_api_key:
        con.print()
        con.print("[yellow]提示：未检测到 API Key，已自动启用 Mock 模式。[/yellow]")
        con.print("[dim]Mock 模式使用内置响应生成器模拟 LLM 输出，无需真实 API 调用。")
        con.print("如需使用真实 LLM，请设置环境变量 OPENAI_API_KEY 或 ANTHROPIC_API_KEY。[/dim]")
        con.print()
        return True

    return False
