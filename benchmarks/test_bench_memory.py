"""
内存占用基准测试。

验证 CLAUDE.md 中定义的性能指标：
- 内存占用：RSS < 512MB（200K Token 上下文）

运行方式::

    python -m pytest benchmarks/bench_memory.py -v
    python benchmarks/bench_memory.py          # 独立运行
"""

from __future__ import annotations

import asyncio
import os
import sys

import pytest

from context_forge import ContextForge


def _get_rss_mb() -> float:
    """获取当前进程的 RSS 内存（MB）。"""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        pass

    if sys.platform != "win32":
        try:
            import resource
            # macOS 返回 bytes，Linux 返回 KB
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform == "darwin":
                return rss / (1024 * 1024)
            return rss / 1024
        except ImportError:
            pass

    return 0.0


def _build_large_context(target_tokens: int = 200_000) -> tuple[str, list[dict[str, str]], list[dict[str, object]]]:
    """
    构建接近 200K Token 的大型上下文输入。

    使用中文文本：约 2 字符 ≈ 1 token（中文 tiktoken 编码）。
    """
    system_prompt = "你是一个专业的技术助手。" * 20  # ~200 token

    # 对话历史：50 轮 × ~1000 token/轮 = ~50K token
    messages: list[dict[str, str]] = []
    msg_text = "这是一段关于 Python 编程语言的详细技术讨论内容，" * 50  # ~500 chars ≈ 250 token
    for i in range(50):
        messages.append({"role": "user", "content": f"第{i}轮：{msg_text}"})
        messages.append({"role": "assistant", "content": f"回复{i}：{msg_text}"})

    # RAG 片段：100 个 × ~1500 token = ~150K token
    rag_chunks: list[dict[str, object]] = []
    chunk_text = "这是一段来自技术文档的 RAG 检索结果，包含关于机器学习和自然语言处理的详细内容。" * 30  # ~750 chars ≈ 375 token
    for i in range(100):
        rag_chunks.append({
            "content": f"文档{i}：{chunk_text}",
            "score": max(0.5, 0.99 - i * 0.005),
        })

    return system_prompt, messages, rag_chunks


@pytest.mark.slow
class TestMemoryUsage:
    """内存占用基准测试。"""

    RSS_THRESHOLD_MB = 512.0

    @pytest.mark.asyncio
    async def test_rss_under_512mb_for_200k_tokens(self) -> None:
        """
        200K Token 上下文组装后 RSS < 512MB。
        """
        forge = ContextForge(model="gpt-4o")
        system_prompt, messages, rag_chunks = _build_large_context(200_000)

        rss_before = _get_rss_mb()

        context = await forge.build(
            system_prompt=system_prompt,
            messages=messages,
            rag_chunks=rag_chunks,
        )

        rss_after = _get_rss_mb()
        rss_delta = rss_after - rss_before

        total_tokens = context.token_usage.total_tokens
        segment_count = len(context.segments)

        print(
            f"\n{'='*60}\n"
            f"内存占用基准（200K Token 目标）\n"
            f"{'='*60}\n"
            f"  实际 Token:     {total_tokens:,}\n"
            f"  Segment 数:     {segment_count}\n"
            f"  RSS (前):       {rss_before:.1f} MB\n"
            f"  RSS (后):       {rss_after:.1f} MB\n"
            f"  RSS 增量:       {rss_delta:.1f} MB\n"
            f"  阈值:           {self.RSS_THRESHOLD_MB:.1f} MB\n"
            f"{'='*60}"
        )

        assert rss_after < self.RSS_THRESHOLD_MB, (
            f"RSS {rss_after:.1f}MB 超过阈值 {self.RSS_THRESHOLD_MB}MB。"
            f"增量: {rss_delta:.1f}MB, Token: {total_tokens:,}"
        )

    @pytest.mark.asyncio
    async def test_no_memory_leak_on_repeated_builds(self) -> None:
        """
        多次 build 后内存不应持续增长（排除内存泄漏）。

        策略：执行 20 次 build，后 10 次的增量不超过前 10 次的 2 倍。
        """
        forge = ContextForge(model="gpt-4o")
        system_prompt = "你是一个助手。"
        messages = [
            {"role": "user", "content": "测试消息" * 100},
            {"role": "assistant", "content": "测试回复" * 100},
        ]
        rag_chunks: list[dict[str, object]] = [
            {"content": f"RAG 内容 {i}" * 50, "score": 0.9}
            for i in range(10)
        ]

        rss_start = _get_rss_mb()

        # 前 10 轮
        for _ in range(10):
            await forge.build(
                system_prompt=system_prompt,
                messages=messages,
                rag_chunks=rag_chunks,
            )
        rss_mid = _get_rss_mb()
        delta_first_half = rss_mid - rss_start

        # 后 10 轮
        for _ in range(10):
            await forge.build(
                system_prompt=system_prompt,
                messages=messages,
                rag_chunks=rag_chunks,
            )
        rss_end = _get_rss_mb()
        delta_second_half = rss_end - rss_mid

        print(
            f"\n{'='*60}\n"
            f"内存泄漏检测（20 轮 build）\n"
            f"{'='*60}\n"
            f"  RSS 起始:       {rss_start:.1f} MB\n"
            f"  RSS 第10轮:     {rss_mid:.1f} MB (+{delta_first_half:.1f})\n"
            f"  RSS 第20轮:     {rss_end:.1f} MB (+{delta_second_half:.1f})\n"
            f"{'='*60}"
        )

        # 允许第二轮有一些增长（GC 未回收），但不应超过第一轮的 2 倍
        # 如果 delta_first_half 很小（< 1MB），放宽至 5MB 绝对阈值
        threshold = max(delta_first_half * 2, 5.0)
        assert delta_second_half < threshold, (
            f"后 10 轮内存增量 ({delta_second_half:.1f}MB) "
            f"超过前 10 轮 ({delta_first_half:.1f}MB) 的 2 倍，"
            f"可能存在内存泄漏。"
        )


if __name__ == "__main__":
    async def main() -> None:
        bench = TestMemoryUsage()
        await bench.test_rss_under_512mb_for_200k_tokens()
        await bench.test_no_memory_leak_on_repeated_builds()

    asyncio.run(main())
