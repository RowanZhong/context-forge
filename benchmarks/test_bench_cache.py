"""
缓存命中延迟基准测试。

验证 CLAUDE.md 中定义的性能指标：
- 缓存命中时延迟降低 > 60%

运行方式::

    python -m pytest benchmarks/test_bench_cache.py -v --no-cov -s
"""

from __future__ import annotations

import asyncio
import statistics
import tempfile
import time

import pytest
import yaml

from context_forge import ContextForge


def _make_cache_policy_path() -> str:
    """创建启用缓存的临时策略文件。"""
    policy = {
        "version": "bench-1.0",
        "name": "bench-cache",
        "cache": {"enabled": True, "backend": "memory", "max_entries": 1000, "ttl_seconds": 600},
    }
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(policy, f, allow_unicode=True)
    f.close()
    return f.name


def _build_inputs() -> dict[str, object]:
    """构建一组固定的输入参数。"""
    return {
        "system_prompt": "你是一个专业的技术助手。",
        "messages": [
            {"role": "user", "content": "介绍一下 Python 的异步编程。"},
            {"role": "assistant", "content": "Python 的异步编程主要通过 asyncio 模块实现..."},
            {"role": "user", "content": "那和多线程有什么区别？"},
        ],
        "rag_chunks": [
            {"content": "asyncio 是 Python 标准库中的异步 I/O 框架..." * 5, "score": 0.92},
            {"content": "多线程受 GIL 限制，在 CPU 密集型任务中效果有限..." * 5, "score": 0.88},
            {"content": "协程的切换开销远小于线程切换，适合 I/O 密集型场景..." * 5, "score": 0.85},
        ],
    }


@pytest.mark.slow
class TestCacheLatency:
    """缓存命中延迟基准测试。"""

    IMPROVEMENT_THRESHOLD = 0.60  # 缓存命中后延迟降低 > 60%
    WARMUP_ROUNDS = 2
    BENCHMARK_ROUNDS = 30

    @pytest.mark.asyncio
    async def test_cache_hit_reduces_latency_by_60_percent(self) -> None:
        """
        缓存命中后，延迟应降低 > 60%。
        """
        policy_path = _make_cache_policy_path()
        forge = ContextForge(model="gpt-4o", policy_path=policy_path)

        inputs = _build_inputs()

        # === 预热 ===
        for _ in range(self.WARMUP_ROUNDS):
            await forge.build(
                system_prompt="预热" * 10,
                messages=[{"role": "user", "content": "warmup"}],
            )

        # === 测量冷启动延迟（无缓存命中）===
        cold_latencies: list[float] = []
        for i in range(self.BENCHMARK_ROUNDS):
            # 每轮使用略不同的输入，确保无缓存命中
            modified_inputs = {
                "system_prompt": "你是一个专业的技术助手。",
                "messages": [
                    {"role": "user", "content": f"第 {i} 次冷查询：介绍 Python 异步编程。"},
                ],
                "rag_chunks": [
                    {"content": f"冷查询 RAG {i}：asyncio 框架介绍..." * 5, "score": 0.9},
                ],
            }

            start = time.perf_counter()
            await forge.build(**modified_inputs)  # type: ignore[arg-type]
            cold_latencies.append((time.perf_counter() - start) * 1000)

        # === 测量缓存命中延迟 ===
        # 先执行一次写入缓存
        await forge.build(**inputs)  # type: ignore[arg-type]

        # 然后多次缓存命中
        hot_latencies: list[float] = []
        for _ in range(self.BENCHMARK_ROUNDS):
            start = time.perf_counter()
            await forge.build(**inputs)  # type: ignore[arg-type]
            hot_latencies.append((time.perf_counter() - start) * 1000)

        # === 统计 ===
        cold_median = statistics.median(cold_latencies)
        hot_median = statistics.median(hot_latencies)
        improvement = 1.0 - (hot_median / cold_median) if cold_median > 0 else 0.0

        print(
            f"\n{'='*60}\n"
            f"缓存命中延迟基准（{self.BENCHMARK_ROUNDS} 轮）\n"
            f"{'='*60}\n"
            f"  冷启动 P50:   {cold_median:.2f} ms\n"
            f"  缓存命中 P50: {hot_median:.2f} ms\n"
            f"  延迟降低:     {improvement*100:.1f}%\n"
            f"  阈值:         {self.IMPROVEMENT_THRESHOLD*100:.1f}%\n"
            f"{'='*60}"
        )

        assert improvement > self.IMPROVEMENT_THRESHOLD, (
            f"缓存命中延迟降低 {improvement*100:.1f}% "
            f"未达到阈值 {self.IMPROVEMENT_THRESHOLD*100:.1f}%。"
            f"冷启动: {cold_median:.2f}ms, 命中: {hot_median:.2f}ms"
        )

    @pytest.mark.asyncio
    async def test_cache_miss_overhead_minimal(self) -> None:
        """
        缓存未命中时的额外开销 < 20%（相比无缓存）。
        """
        # 无缓存
        forge_no_cache = ContextForge(model="gpt-4o")
        # 有缓存
        policy_path = _make_cache_policy_path()
        forge_with_cache = ContextForge(model="gpt-4o", policy_path=policy_path)

        base_inputs = _build_inputs()

        # 预热
        for _ in range(3):
            await forge_no_cache.build(**base_inputs)  # type: ignore[arg-type]
            await forge_with_cache.build(**base_inputs)  # type: ignore[arg-type]

        # 测量无缓存
        no_cache_latencies: list[float] = []
        for i in range(20):
            inputs_i = {
                "system_prompt": "助手。",
                "messages": [{"role": "user", "content": f"无缓存测试 {i}"}],
            }
            start = time.perf_counter()
            await forge_no_cache.build(**inputs_i)  # type: ignore[arg-type]
            no_cache_latencies.append((time.perf_counter() - start) * 1000)

        # 测量有缓存但未命中（使用不同输入确保未命中）
        miss_latencies: list[float] = []
        for i in range(20):
            inputs_i = {
                "system_prompt": "助手miss。",
                "messages": [{"role": "user", "content": f"缓存未命中测试 {i}"}],
            }
            start = time.perf_counter()
            await forge_with_cache.build(**inputs_i)  # type: ignore[arg-type]
            miss_latencies.append((time.perf_counter() - start) * 1000)

        no_cache_median = statistics.median(no_cache_latencies)
        miss_median = statistics.median(miss_latencies)
        overhead = (miss_median / no_cache_median - 1.0) if no_cache_median > 0 else 0.0

        print(
            f"\n{'='*60}\n"
            f"缓存未命中开销\n"
            f"{'='*60}\n"
            f"  无缓存 P50:      {no_cache_median:.2f} ms\n"
            f"  未命中 P50:      {miss_median:.2f} ms\n"
            f"  额外开销:        {overhead*100:.1f}%\n"
            f"  阈值:            20%\n"
            f"{'='*60}"
        )

        assert overhead < 0.20, (
            f"缓存未命中开销 {overhead*100:.1f}% 超过 20%。"
            f"无缓存: {no_cache_median:.2f}ms, 未命中: {miss_median:.2f}ms"
        )


if __name__ == "__main__":
    async def main() -> None:
        bench = TestCacheLatency()
        await bench.test_cache_hit_reduces_latency_by_60_percent()
        await bench.test_cache_miss_overhead_minimal()

    asyncio.run(main())
