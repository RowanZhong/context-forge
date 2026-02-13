"""
Context Forge 快速上手示例。

演示最基本的用法：3 行代码完成上下文组装。

运行方式：
    python examples/quickstart.py

无需 API Key，无需任何配置文件。
"""

import asyncio


async def main() -> None:
    from context_forge import ContextForge

    # ===== 场景 1：最简用法 =====
    print("=" * 60)
    print("场景 1：最简用法（3 行代码）")
    print("=" * 60)

    forge = ContextForge(model="gpt-4o")
    context = await forge.build(
        system_prompt="你是一个有用的助手。",
        messages=[{"role": "user", "content": "你好，请介绍一下你自己。"}],
    )

    print(f"\n组装结果：")
    print(f"  Segment 数量：{len(context.segments)}")
    print(f"  总 Token：{context.token_usage.total_tokens:,}")
    print(f"  组装耗时：{context.assembly_duration_ms:.1f}ms")
    print(f"\n消息格式：")
    for msg in context.to_messages():
        print(f"  [{msg['role']}] {msg['content'][:80]}...")

    # ===== 场景 2：带 RAG 片段 =====
    print("\n" + "=" * 60)
    print("场景 2：带 RAG 检索片段")
    print("=" * 60)

    context = await forge.build(
        system_prompt="你是一个客服助手，根据知识库回答用户问题。",
        messages=[
            {"role": "user", "content": "你们的退货政策是什么？"},
        ],
        rag_chunks=[
            {
                "content": "退货政策：自收货之日起 7 天内，商品未拆封可无理由退货。"
                           "已拆封商品如有质量问题，15 天内可申请换货。",
                "score": 0.95,
                "source_id": "policy_doc_001",
            },
            {
                "content": "退款流程：提交退货申请后，客服将在 24 小时内审核。"
                           "审核通过后，退款将在 3-5 个工作日内到账。",
                "score": 0.87,
                "source_id": "policy_doc_002",
            },
            {
                "content": "公司简介：我们是一家成立于 2020 年的电商平台...",
                "score": 0.32,
                "source_id": "about_doc_001",
            },
        ],
    )

    print(f"\n组装结果：")
    print(f"  Segment 数量：{len(context.segments)}")
    print(f"  总 Token：{context.token_usage.total_tokens:,}")

    # 查看预算使用
    if context.budget_allocation:
        print(f"\n预算使用：")
        print(f"  饱和度：{context.budget_allocation.saturation_rate:.1%}")

    # 查看审计日志
    if context.dropped_segments:
        print(f"\n被丢弃的 Segment：")
        for entry in context.dropped_segments:
            print(f"  - {entry.segment_id}: {entry.reason_code.value}")
            if entry.reason_detail:
                print(f"    原因：{entry.reason_detail}")

    if context.warnings:
        print(f"\n警告：")
        for w in context.warnings:
            print(f"  ⚠ {w}")

    # ===== 场景 3：多轮对话 =====
    print("\n" + "=" * 60)
    print("场景 3：多轮对话历史")
    print("=" * 60)

    history = [
        {"role": "user", "content": "我想买一台笔记本电脑"},
        {"role": "assistant", "content": "您的预算大概是多少？主要用途是什么？"},
        {"role": "user", "content": "预算 8000 左右，主要写代码和跑模型"},
        {"role": "assistant", "content": "推荐考虑 ThinkPad X1 Carbon 或 MacBook Air M3..."},
        {"role": "user", "content": "MacBook 能跑 PyTorch 吗？"},
    ]

    context = await forge.build(
        system_prompt="你是一个电子产品导购助手。",
        messages=history,
        current_turn=len(history) // 2,
    )

    print(f"\n组装结果：")
    print(f"  消息数：{len(context.to_messages())}")
    print(f"  总 Token：{context.token_usage.total_tokens:,}")
    print(f"  组装耗时：{context.assembly_duration_ms:.1f}ms")

    # 打印完整摘要
    print(f"\n{context.summary()}")


if __name__ == "__main__":
    asyncio.run(main())
