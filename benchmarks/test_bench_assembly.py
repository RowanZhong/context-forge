"""
组装延迟基准测试。

验证 CLAUDE.md 中定义的性能指标：
- 单次组装延迟（不含 LLM）：< 50ms P99（10 Segment, 128K 窗口）

运行方式::

    python -m pytest benchmarks/bench_assembly.py -v
    python benchmarks/bench_assembly.py          # 独立运行，打印详细统计
"""

from __future__ import annotations

import asyncio
import statistics
import time

import pytest

from context_forge import ContextForge
from context_forge.models.control import ControlFlags
from context_forge.models.metadata import SegmentMetadata
from context_forge.models.provenance import Provenance, SourceType
from context_forge.models.segment import Priority, Segment, SegmentType


def _build_segments(count: int = 10) -> tuple[str, list[dict[str, str]], list[dict[str, object]]]:
    """
    构建典型的 10-Segment 输入场景。

    返回:
        (system_prompt, messages, rag_chunks)
    """
    system_prompt = "你是一个专业的技术助手，专注于 Python 和 LLM 应用开发。请用中文回答。"

    messages: list[dict[str, str]] = []
    for i in range(3):
        messages.append({"role": "user", "content": f"第 {i + 1} 轮用户消息：请介绍 Python 的高级特性。"})
        messages.append({"role": "assistant", "content": f"第 {i + 1} 轮助手回复：Python 有很多高级特性，包括装饰器、上下文管理器、元类等。"})

    rag_chunks: list[dict[str, object]] = []
    for i in range(4):
        rag_chunks.append({
            "content": f"RAG 片段 {i + 1}：这是一段关于 Python 高级特性的技术文档内容，包含详细的代码示例和最佳实践指南。" * 5,
            "score": 0.95 - i * 0.1,
        })

    return system_prompt, messages, rag_chunks


@pytest.mark.slow
class TestAssemblyLatency:
    """组装延迟基准测试。"""

    WARMUP_ROUNDS = 3
    BENCHMARK_ROUNDS = 50
    P99_THRESHOLD_MS = 50.0

    @pytest.fixture
    def forge(self) -> ContextForge:
        """128K 窗口的 ContextForge 实例。"""
        return ContextForge(model="gpt-4o")

    async def _measure_build(self, forge: ContextForge) -> float:
        """执行一次 build 并返回耗时（ms）。"""
        system_prompt, messages, rag_chunks = _build_segments(10)

        start = time.perf_counter()
        await forge.build(
            system_prompt=system_prompt,
            messages=messages,
            rag_chunks=rag_chunks,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        return elapsed_ms

    @pytest.mark.asyncio
    async def test_assembly_p99_under_50ms(self, forge: ContextForge) -> None:
        """
        P99 延迟 < 50ms（10 Segment, 128K 窗口，不含 LLM）。

        策略：
        1. 预热 3 轮（排除 JIT / 缓存冷启动影响）
        2. 测量 50 轮
        3. 计算 P99 并断言
        """
        # 预热
        for _ in range(self.WARMUP_ROUNDS):
            await self._measure_build(forge)

        # 基准测量
        latencies: list[float] = []
        for _ in range(self.BENCHMARK_ROUNDS):
            elapsed = await self._measure_build(forge)
            latencies.append(elapsed)

        # 统计
        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p95_idx = int(len(latencies) * 0.95)
        p99_idx = int(len(latencies) * 0.99)
        p95 = latencies[min(p95_idx, len(latencies) - 1)]
        p99 = latencies[min(p99_idx, len(latencies) - 1)]
        avg = statistics.mean(latencies)
        std = statistics.stdev(latencies) if len(latencies) > 1 else 0.0

        print(
            f"\n{'='*60}\n"
            f"组装延迟基准（{self.BENCHMARK_ROUNDS} 轮，10 Segment, 128K 窗口）\n"
            f"{'='*60}\n"
            f"  平均:  {avg:.2f} ms\n"
            f"  标准差: {std:.2f} ms\n"
            f"  P50:   {p50:.2f} ms\n"
            f"  P95:   {p95:.2f} ms\n"
            f"  P99:   {p99:.2f} ms\n"
            f"  最小:  {min(latencies):.2f} ms\n"
            f"  最大:  {max(latencies):.2f} ms\n"
            f"  阈值:  {self.P99_THRESHOLD_MS:.2f} ms\n"
            f"{'='*60}"
        )

        assert p99 < self.P99_THRESHOLD_MS, (
            f"P99 延迟 {p99:.2f}ms 超过阈值 {self.P99_THRESHOLD_MS}ms。"
            f"平均: {avg:.2f}ms, 标准差: {std:.2f}ms"
        )

    @pytest.mark.asyncio
    async def test_assembly_scales_linearly(self, forge: ContextForge) -> None:
        """
        验证延迟随 Segment 数量线性扩展（非指数增长）。

        策略：分别测量 5/10/20 个 Segment 的延迟，
        验证 20-Segment 不超过 10-Segment 的 3 倍。
        """
        async def measure_n_segments(n_rag: int, rounds: int = 10) -> float:
            system_prompt = "你是一个助手。"
            messages = [
                {"role": "user", "content": "测试消息"},
                {"role": "assistant", "content": "测试回复"},
            ]
            rag_chunks = [
                {"content": f"RAG 片段 {i}" * 10, "score": 0.9 - i * 0.01}
                for i in range(n_rag)
            ]

            latencies = []
            for _ in range(rounds):
                start = time.perf_counter()
                await forge.build(
                    system_prompt=system_prompt,
                    messages=messages,
                    rag_chunks=rag_chunks,
                )
                latencies.append((time.perf_counter() - start) * 1000)

            return statistics.median(latencies)

        # 预热
        await measure_n_segments(5, rounds=2)

        lat_5 = await measure_n_segments(5)
        lat_10 = await measure_n_segments(10)
        lat_20 = await measure_n_segments(20)

        print(
            f"\n{'='*60}\n"
            f"线性扩展测试\n"
            f"{'='*60}\n"
            f"  5 Segment:  {lat_5:.2f} ms\n"
            f"  10 Segment: {lat_10:.2f} ms  (比例: {lat_10/lat_5:.2f}x)\n"
            f"  20 Segment: {lat_20:.2f} ms  (比例: {lat_20/lat_5:.2f}x)\n"
            f"{'='*60}"
        )

        # 20-Segment 不应超过 10-Segment 的 3 倍（允许常数开销）
        assert lat_20 < lat_10 * 3, (
            f"20-Segment 延迟 ({lat_20:.2f}ms) 超过 10-Segment ({lat_10:.2f}ms) 的 3 倍，"
            f"可能存在非线性扩展问题。"
        )


if __name__ == "__main__":
    async def main() -> None:
        forge = ContextForge(model="gpt-4o")
        bench = TestAssemblyLatency()
        bench.BENCHMARK_ROUNDS = 100
        await bench.test_assembly_p99_under_50ms(forge)
        await bench.test_assembly_scales_linearly(forge)

    asyncio.run(main())
