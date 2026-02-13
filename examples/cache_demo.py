"""
缓存模块演示 — 提升上下文组装性能。

→ 6.2.3 缓存架构与复用优化

演示场景：
1. Segment 级缓存：复用清洗后的内容
2. Prefix 级缓存：共享静态 System Segment
3. Context 级缓存：跳过整个 Pipeline
4. L1 + L2 分层缓存：跨进程复用

运行方式：
    python examples/cache_demo.py
"""

import asyncio
import json
import sys
from datetime import datetime

# 修复 Windows 控制台编码
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from context_forge.cache import (
    CacheEntry,
    CacheManager,
    MemoryCache,
    compute_context_key,
    compute_prefix_key,
    compute_segment_key,
    create_default_cache,
)


def print_section(title: str) -> None:
    """打印章节标题。"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


async def demo_segment_cache():
    """演示 Segment 级缓存。"""
    print_section("场景 1：Segment 级缓存 — 复用清洗结果")

    cache = create_default_cache(l1_max_size=1000, default_ttl=3600)

    # 模拟清洗过的 Segment
    original_content = "Python's GIL has been removed in 3.13"
    cleaned_content = "Python's GIL has been removed in 3.13"  # 假设清洗后无变化
    model = "gpt-4o"

    # 计算缓存键
    cache_key = compute_segment_key(original_content, model)
    print(f"缓存键: {cache_key}")

    # 首次访问：未命中，需要清洗
    cached = await cache.get(cache_key)
    if cached is None:
        print("❌ 缓存未命中 — 执行清洗操作...")
        # 模拟清洗过程（实际会调用 Sanitizer）
        await asyncio.sleep(0.1)  # 模拟清洗延迟
        cleaned_data = {
            "content": cleaned_content,
            "sanitized": True,
            "timestamp": datetime.utcnow().isoformat(),
        }
        entry = CacheEntry(value=json.dumps(cleaned_data))
        await cache.set(cache_key, entry, ttl=3600)
        print(f"✓ 清洗完成并缓存: {cleaned_data}")
    else:
        print(f"✓ 缓存命中: {cached.value}")

    # 再次访问：命中缓存
    print("\n第二次访问相同内容...")
    cached = await cache.get(cache_key)
    if cached is not None:
        data = json.loads(cached.value)
        print(f"✓ 缓存命中（命中次数: {cached.hit_count}）")
        print(f"  内容: {data['content']}")
        print(f"  缓存时间: {data['timestamp']}")

    # 统计信息
    stats = await cache.stats()
    print(f"\n缓存统计: 命中率={stats['l1'].hit_rate:.1%}, "
          f"命中={stats['l1'].hits}, 未命中={stats['l1'].misses}")


async def demo_prefix_cache():
    """演示 Prefix 级缓存。"""
    print_section("场景 2：Prefix 级缓存 — 共享静态 System Segment")

    cache = create_default_cache()

    # 静态 System Segment 序列（多轮对话中不变）
    system_segments = [
        "You are a helpful AI assistant.",
        "Always respond in Chinese.",
        "Be concise and accurate.",
    ]
    model = "claude-sonnet-4-5-20250514"
    policy_version = "v1.0"

    # 计算 Prefix 缓存键
    prefix_key = compute_prefix_key(system_segments, model, policy_version)
    print(f"Prefix 缓存键: {prefix_key}")

    # 模拟 KV Cache 前缀
    cached = await cache.get(prefix_key)
    if cached is None:
        print("❌ Prefix 缓存未命中 — 生成 KV Cache...")
        await asyncio.sleep(0.2)  # 模拟 LLM 预填充延迟
        kv_cache_data = {
            "model": model,
            "segments": system_segments,
            "kv_cache_size_mb": 1.2,
            "created_at": datetime.utcnow().isoformat(),
        }
        entry = CacheEntry(value=json.dumps(kv_cache_data))
        await cache.set(prefix_key, entry, ttl=7200)  # System Segment 较长 TTL
        print(f"✓ KV Cache 已生成并缓存: {kv_cache_data['kv_cache_size_mb']} MB")
    else:
        data = json.loads(cached.value)
        print(f"✓ Prefix 缓存命中 — 复用 KV Cache")
        print(f"  System Segments: {data['segments']}")
        print(f"  KV Cache 大小: {data['kv_cache_size_mb']} MB")
        print(f"  创建时间: {data['created_at']}")

    print("\n💡 提示: 在多轮对话中，静态 System Segment 的 Prefix 缓存")
    print("   可以显著减少 LLM 预填充时间（RadixAttention 优化）。")


async def demo_context_cache():
    """演示 Context 级缓存。"""
    print_section("场景 3：Context 级缓存 — 跳过整个 Pipeline")

    cache = create_default_cache()

    # 完整上下文结构（序列化为 dict）
    context_segments = [
        {"type": "system", "content": "You are a math tutor.", "priority": "critical"},
        {"type": "few_shot", "content": "Q: 1+1? A: 2", "priority": "high"},
        {"type": "user", "content": "What is 2+2?", "priority": "high"},
    ]
    model = "gpt-4o-mini"
    policy_version = "v2.0"

    # 计算 Context 缓存键
    context_key = compute_context_key(context_segments, model, policy_version)
    print(f"Context 缓存键: {context_key}")

    # 首次组装
    cached = await cache.get(context_key)
    if cached is None:
        print("❌ Context 缓存未命中 — 执行完整 Pipeline...")
        print("   [Normalize → Sanitize → Rerank → Allocate → Assemble]")
        await asyncio.sleep(0.3)  # 模拟 Pipeline 延迟
        assembled_context = {
            "segments": context_segments,
            "total_tokens": 120,
            "model": model,
            "assembled_at": datetime.utcnow().isoformat(),
        }
        entry = CacheEntry(value=json.dumps(assembled_context))
        await cache.set(context_key, entry, ttl=1800)
        print(f"✓ Pipeline 完成: {assembled_context['total_tokens']} tokens")
    else:
        data = json.loads(cached.value)
        print(f"✓ Context 缓存命中 — 跳过所有 Pipeline 阶段")
        print(f"  总 Token: {data['total_tokens']}")
        print(f"  组装时间: {data['assembled_at']}")
        print(f"  命中次数: {cached.hit_count}")

    print("\n💡 适用场景: 固定的 Few-Shot 示例 + System Prompt（如 JSON 输出格式）")
    print("   不适用场景: 动态 RAG + 多轮对话（每次内容都不同）")


async def demo_l1_l2_cache():
    """演示 L1 + L2 分层缓存。"""
    print_section("场景 4：L1 + L2 分层缓存 — 跨进程复用")

    # 注意：需要 Redis 运行在 localhost:6379
    # 如果 Redis 不可用，会自动降级为仅 L1
    cache = create_default_cache(
        redis_url="redis://localhost:6379/0",
        l1_max_size=100,
        default_ttl=3600,
    )

    if cache.has_l2:
        print("✓ L2 缓存（Redis）已启用")
    else:
        print("⚠️  L2 缓存（Redis）不可用，仅使用 L1（内存）")
        print("   提示：安装 Redis 并运行 `redis-server` 以启用 L2 缓存。")

    # 模拟跨进程场景
    key = "shared_segment"
    content = "This segment is shared across workers."

    # Worker 1 写入
    print(f"\n[Worker 1] 写入缓存: {key}")
    entry = CacheEntry(value=json.dumps({"content": content, "worker": 1}))
    await cache.set(key, entry)
    print("✓ 已写入 L1 和 L2")

    # 模拟 Worker 2 读取（清空 L1，强制从 L2 读取）
    print(f"\n[Worker 2] 读取缓存（L1 已清空，模拟新进程）")
    # 清空 L1（模拟新 worker 的空缓存）
    await cache._l1.clear()

    result = await cache.get(key)
    if result is not None:
        data = json.loads(result.value)
        print(f"✓ 从 L2 读取成功: {data}")
        print(f"  来源: Worker {data['worker']}")
        print("  → L2 命中后自动回填 L1，后续访问更快")
    else:
        print("❌ 缓存未命中（可能 Redis 未运行）")

    # 统计
    stats = await cache.stats()
    print(f"\n缓存统计:")
    print(f"  L1: 命中率={stats['l1'].hit_rate:.1%}, 大小={stats['l1'].current_size}")
    if "l2" in stats:
        print(f"  L2: 命中率={stats['l2'].hit_rate:.1%}, 大小={stats['l2'].current_size}")

    await cache.close()


async def demo_performance_comparison():
    """性能对比：有缓存 vs 无缓存。"""
    print_section("场景 5：性能对比 — 缓存加速效果")

    cache = create_default_cache()
    content = "A" * 10000  # 10KB 内容
    model = "gpt-4o"

    # 无缓存：每次都执行清洗
    print("无缓存场景（10 次清洗）...")
    start = asyncio.get_event_loop().time()
    for i in range(10):
        await asyncio.sleep(0.01)  # 模拟清洗延迟
    no_cache_time = asyncio.get_event_loop().time() - start
    print(f"  总耗时: {no_cache_time*1000:.1f} ms")

    # 有缓存：首次清洗，后续命中
    print("\n有缓存场景（1 次清洗 + 9 次命中）...")
    key = compute_segment_key(content, model)
    start = asyncio.get_event_loop().time()

    # 首次：未命中
    cached = await cache.get(key)
    if cached is None:
        await asyncio.sleep(0.01)  # 模拟清洗
        entry = CacheEntry(value=json.dumps({"content": content}))
        await cache.set(key, entry)

    # 后续：命中
    for i in range(9):
        await cache.get(key)

    cache_time = asyncio.get_event_loop().time() - start
    print(f"  总耗时: {cache_time*1000:.1f} ms")

    # 加速比
    speedup = no_cache_time / cache_time if cache_time > 0 else 0
    print(f"\n📊 加速比: {speedup:.1f}x")
    print(f"   节省时间: {(no_cache_time - cache_time)*1000:.1f} ms")


async def main():
    """运行所有演示。"""
    print("\n" + "="*60)
    print("  Context Forge — 缓存模块演示")
    print("  → 6.2.3 缓存架构与复用优化")
    print("="*60)

    await demo_segment_cache()
    await demo_prefix_cache()
    await demo_context_cache()
    await demo_l1_l2_cache()
    await demo_performance_comparison()

    print("\n" + "="*60)
    print("  演示完成 ✓")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
