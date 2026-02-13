"""
压缩模块演示 — 展示三种压缩策略的使用。

运行方式：
    python examples/compress_demo.py

演示内容：
1. TruncationCompressor：简单截断（三种策略）
2. DedupCompressor：语义去重
3. CompressEngine：饱和度触发的自适应压缩
"""

import asyncio

from context_forge.compress import (
    CompressContext,
    CompressEngine,
    DedupCompressor,
    TruncationCompressor,
    TruncationStrategy,
)
from context_forge.models.control import ControlFlags
from context_forge.models.segment import Priority, Segment, SegmentType


def print_section(title: str):
    """打印分隔符"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


async def demo_truncation_compressor():
    """演示截断压缩器"""
    print_section("1. 截断压缩器演示")

    # 创建测试数据
    segments = [
        Segment(
            type=SegmentType.SYSTEM,
            content="你是一个专业的 AI 助手，擅长回答技术问题。",
            role="system",
            token_count=50,
        ),
        Segment(
            type=SegmentType.USER,
            content="请解释 Python 的 GIL 是什么？",
            role="user",
            token_count=100,
        ),
        Segment(
            type=SegmentType.ASSISTANT,
            content="GIL（全局解释器锁）是 CPython 解释器的一个机制...",
            role="assistant",
            token_count=200,
        ),
        Segment(
            type=SegmentType.USER,
            content="Python 3.13 对 GIL 做了什么改进？",
            role="user",
            token_count=100,
        ),
    ]

    total_tokens = sum(seg.token_count or 0 for seg in segments)
    print(f"原始 Segment 数：{len(segments)}")
    print(f"原始 Token 数：{total_tokens}")

    # 策略 1：Tail（保留头部）
    print("\n策略 1：TAIL（保留头部，适合保留系统提示）")
    compressor_tail = TruncationCompressor(strategy=TruncationStrategy.TAIL)
    context = CompressContext(
        available_tokens=1000,
        target_token_count=200,  # 压缩到 200 tokens
        saturation=0.9,
    )
    result_tail = await compressor_tail.compress(segments, context)
    print(f"  压缩后 Segment 数：{len(result_tail.compressed_segments)}")
    print(f"  压缩后 Token 数：{result_tail.compressed_token_count}")
    print(f"  压缩比例：{result_tail.compression_ratio:.2%}")
    print(f"  节省 Token：{result_tail.tokens_saved}")

    # 策略 2：Head（保留尾部）
    print("\n策略 2：HEAD（保留尾部，适合保留最新消息）")
    compressor_head = TruncationCompressor(strategy=TruncationStrategy.HEAD)
    result_head = await compressor_head.compress(segments, context)
    print(f"  压缩后 Segment 数：{len(result_head.compressed_segments)}")
    print(f"  压缩后 Token 数：{result_head.compressed_token_count}")

    # 策略 3：Middle（保留头尾）
    print("\n策略 3：MIDDLE（保留头尾，适合长文档摘要）")
    compressor_middle = TruncationCompressor(
        strategy=TruncationStrategy.MIDDLE,
        head_ratio=0.6,  # 头部 60%，尾部 40%
    )
    result_middle = await compressor_middle.compress(segments, context)
    print(f"  压缩后 Segment 数：{len(result_middle.compressed_segments)}")
    print(f"  压缩后 Token 数：{result_middle.compressed_token_count}")


async def demo_dedup_compressor():
    """演示去重压缩器"""
    print_section("2. 去重压缩器演示")

    # 创建包含重复内容的测试数据
    segments = [
        Segment(
            type=SegmentType.RAG,
            content="Python 的 GIL 在 3.13 版本中已被移除，支持真正的多线程并行。",
            role="user",
            token_count=100,
            priority=Priority.MEDIUM,
        ),
        Segment(
            type=SegmentType.RAG,
            content="Python 3.13 移除了全局解释器锁 GIL，实现了真正的并行执行。",
            role="user",
            token_count=100,
            priority=Priority.LOW,  # 优先级更低，会被删除
        ),
        Segment(
            type=SegmentType.RAG,
            content="Python 的 GIL 在 3.13 版本中已被移除，支持真正的多线程并行。",
            role="user",
            token_count=100,
            priority=Priority.LOW,  # 完全重复，会被删除
        ),
        Segment(
            type=SegmentType.RAG,
            content="Rust 的所有权系统保证了内存安全，无需垃圾回收器。",
            role="user",
            token_count=100,
            priority=Priority.MEDIUM,
        ),
    ]

    total_tokens = sum(seg.token_count or 0 for seg in segments)
    print(f"原始 Segment 数：{len(segments)}")
    print(f"原始 Token 数：{total_tokens}")

    # 去重压缩
    compressor = DedupCompressor(similarity_threshold=0.85)
    context = CompressContext(
        available_tokens=1000,
        target_token_count=400,
        saturation=0.9,
    )
    result = await compressor.compress(segments, context)

    print(f"\n去重结果：")
    print(f"  压缩后 Segment 数：{len(result.compressed_segments)}")
    print(f"  压缩后 Token 数：{result.compressed_token_count}")
    print(f"  压缩比例：{result.compression_ratio:.2%}")
    print(f"  节省 Token：{result.tokens_saved}")
    print(f"  删除的重复项数：{result.metadata.get('removed_count', 0)}")


async def demo_compress_engine():
    """演示压缩引擎"""
    print_section("3. 压缩引擎演示（饱和度触发 + 优先级保护）")

    # 创建不同优先级的 Segment
    segments = [
        Segment(
            type=SegmentType.SYSTEM,
            content="你是一个专业的 AI 助手" * 10,  # 大系统提示
            role="system",
            token_count=300,
            priority=Priority.CRITICAL,  # 不可压缩
        ),
        Segment(
            type=SegmentType.USER,
            content="当前用户问题：如何优化 Python 性能？",
            role="user",
            token_count=100,
            priority=Priority.HIGH,
            control=ControlFlags(must_keep=True),  # 标记为必须保留
        ),
        # 一堆 LOW 优先级的 RAG 片段（可压缩）
        *[
            Segment(
                type=SegmentType.RAG,
                content=f"RAG 检索片段 {i}：关于 Python 性能优化的内容...",
                role="user",
                token_count=100,
                priority=Priority.LOW,
            )
            for i in range(1, 11)
        ],
    ]

    total_tokens = sum(seg.token_count or 0 for seg in segments)
    print(f"原始 Segment 数：{len(segments)}")
    print(f"原始 Token 数：{total_tokens}")

    # 创建压缩引擎（饱和度阈值 0.6）
    engine = CompressEngine(
        saturation_threshold=0.6,
        preserve_must_keep=True,
    )

    # 场景 1：预算充足，不触发压缩
    print("\n场景 1：预算充足（饱和度 < 0.6）")
    available_tokens = 2000  # 充足的预算
    saturation = total_tokens / available_tokens
    print(f"  可用 Token：{available_tokens}")
    print(f"  饱和度：{saturation:.2%}")

    result1 = await engine.compress(
        segments=segments,
        available_tokens=available_tokens,
        audit_log=[],
    )
    print(f"  压缩后 Segment 数：{len(result1)}（未压缩）")

    # 场景 2：预算紧张，触发压缩
    print("\n场景 2：预算紧张（饱和度 > 0.6）")
    available_tokens = 800  # 紧张的预算
    saturation = total_tokens / available_tokens
    print(f"  可用 Token：{available_tokens}")
    print(f"  饱和度：{saturation:.2%}")

    audit_log = []
    result2 = await engine.compress(
        segments=segments,
        available_tokens=available_tokens,
        audit_log=audit_log,
    )

    compressed_tokens = sum(seg.token_count or 0 for seg in result2)
    print(f"  压缩后 Segment 数：{len(result2)}")
    print(f"  压缩后 Token 数：{compressed_tokens}")
    print(f"  节省 Token：{total_tokens - compressed_tokens}")

    # 验证 CRITICAL 和 must_keep 被保护
    critical_count = sum(1 for seg in result2 if seg.effective_priority == Priority.CRITICAL)
    must_keep_count = sum(1 for seg in result2 if seg.control and seg.control.must_keep)
    print(f"\n保护验证：")
    print(f"  CRITICAL 优先级 Segment 数：{critical_count}（应为 1）")
    print(f"  must_keep 标记 Segment 数：{must_keep_count}（应为 1）")


async def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("  Context Forge 压缩模块演示")
    print("=" * 60)

    await demo_truncation_compressor()
    await demo_dedup_compressor()
    await demo_compress_engine()

    print("\n" + "=" * 60)
    print("  演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
