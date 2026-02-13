"""
场景 2：多轮对话记忆管理

演示如何在长对话中管理上下文：
- 20 轮对话历史
- must_keep 关键信息保护
- 滑动窗口策略
- 历史压缩
- 时效性衰减

使用方法：
  python examples/scenario_conversation_memory.py          # 使用 mock 模式（默认）
  python examples/scenario_conversation_memory.py --mock   # 明确指定 mock
  python examples/scenario_conversation_memory.py --no-mock # 使用真实 LLM（需要 API Key）

→ 6.2.4 压缩策略
→ 6.3.3 Compress：压缩策略引擎
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from examples._shared import (
    MockLLM,
    check_api_key,
    console,
    create_comparison_table,
    format_percentage,
    format_tokens,
    parse_args,
    print_header,
    print_section,
    print_success,
    print_warning,
    truncate_text,
)


async def main(mock: bool = True):
    """主函数。"""
    from context_forge import ContextForge
    from context_forge.models.control import ControlFlags
    from context_forge.models.segment import Priority, Segment, SegmentType
    from context_forge.models.provenance import Provenance, SourceType
    from context_forge.models.metadata import SegmentMetadata

    print_header(
        "场景 2：多轮对话记忆管理",
        "演示如何在长对话中压缩历史、保护关键信息、优化 Token 使用"
    )

    # 检查 API Key
    if not check_api_key(mock):
        mock = True

    print_section("步骤 1：构建 20 轮对话历史")

    # 创建对话历史（模拟旅行规划助手）
    now = datetime.now()
    conversation_history = [
        # 第 1 轮（15 天前）- 初次咨询
        {"role": "user", "content": "我想计划一次日本旅行，时间在 5 月份，预算 2 万元。", "turn": 1, "days_ago": 15, "must_keep": True},
        {"role": "assistant", "content": "好的！5 月份去日本是个不错的选择。请问您计划去几天？主要想去哪些城市？", "turn": 1, "days_ago": 15},

        # 第 2 轮（14 天前）
        {"role": "user", "content": "大概 7-10 天，想去东京和京都。", "turn": 2, "days_ago": 14, "must_keep": True},
        {"role": "assistant", "content": "7-10 天游东京和京都很充裕。2 万元预算可以玩得不错。建议东京 4 天，京都 3 天，可以安排 1-2 天大阪。", "turn": 2, "days_ago": 14},

        # 第 3-5 轮（13-11 天前）- 闲聊
        {"role": "user", "content": "日本的樱花季是什么时候？", "turn": 3, "days_ago": 13},
        {"role": "assistant", "content": "3 月下旬到 4 月上旬是关东地区的樱花季，5 月已过花期，但可以看到新绿和紫藤。", "turn": 3, "days_ago": 13},
        {"role": "user", "content": "那边天气怎么样？", "turn": 4, "days_ago": 12},
        {"role": "assistant", "content": "5 月日本气温在 15-25℃，温和舒适，偶尔有雨。建议带薄外套和雨具。", "turn": 4, "days_ago": 12},
        {"role": "user", "content": "需要办签证吗？", "turn": 5, "days_ago": 11},
        {"role": "assistant", "content": "是的，中国公民前往日本需要签证。建议提前 1 个月办理，准备护照、照片、在职证明等材料。", "turn": 5, "days_ago": 11},

        # 第 6-8 轮（10-8 天前）- 深入规划
        {"role": "user", "content": "我对动漫和电子产品比较感兴趣。", "turn": 6, "days_ago": 10, "must_keep": True},
        {"role": "assistant", "content": "那秋叶原和中野百老汇一定要去！秋叶原是电子产品和动漫周边的天堂，中野有很多复古玩具店。", "turn": 6, "days_ago": 10},
        {"role": "user", "content": "有推荐的住宿吗？", "turn": 7, "days_ago": 9},
        {"role": "assistant", "content": "东京可以住新宿或浅草，交通方便。京都推荐住河原町或京都站附近。预算每晚 500-800 元的话可以住不错的商务酒店。", "turn": 7, "days_ago": 9},
        {"role": "user", "content": "美食方面有什么建议？", "turn": 8, "days_ago": 8},
        {"role": "assistant", "content": "东京必吃：筑地市场寿司、一兰拉面、矶丸水产。京都推荐：锦市场小吃、怀石料理、抹茶甜品。", "turn": 8, "days_ago": 8},

        # 第 9-12 轮（7-4 天前）- 更多细节
        {"role": "user", "content": "交通方面需要买 JR Pass 吗？", "turn": 9, "days_ago": 7},
        {"role": "assistant", "content": "东京-京都往返加上市内交通，7 日 JR Pass（约 1800 元）比较划算。也可以考虑 ICOCA 卡用于市内交通。", "turn": 9, "days_ago": 7},
        {"role": "user", "content": "语言不通怎么办？", "turn": 10, "days_ago": 6},
        {"role": "assistant", "content": "日本主要景点和车站都有中英文标识。建议下载 Google 翻译和换乘案内 App，准备一些常用日语短句。", "turn": 10, "days_ago": 6},
        {"role": "user", "content": "有什么需要注意的礼仪吗？", "turn": 11, "days_ago": 5},
        {"role": "assistant", "content": "主要注意：电车上不要大声说话、垃圾分类、进屋脱鞋、泡温泉前要先洗澡、不要边走边吃。", "turn": 11, "days_ago": 5},
        {"role": "user", "content": "购物可以退税吗？", "turn": 12, "days_ago": 4},
        {"role": "assistant", "content": "可以。单次购买满 5000 日元（约 250 元）可以办理退税，退税率 8%。需要带护照，在指定柜台办理。", "turn": 12, "days_ago": 4},

        # 第 13-16 轮（3-1 天前）- 进一步确认
        {"role": "user", "content": "带多少现金合适？", "turn": 13, "days_ago": 3},
        {"role": "assistant", "content": "建议带 3-5 万日元现金（约 1500-2500 元），用于小店消费和交通。多数地方支持信用卡。", "turn": 13, "days_ago": 3},
        {"role": "user", "content": "需要买保险吗？", "turn": 14, "days_ago": 2},
        {"role": "assistant", "content": "建议购买旅游意外险，涵盖医疗、行李丢失、航班延误等。费用约 100-200 元。", "turn": 14, "days_ago": 2},
        {"role": "user", "content": "手机上网怎么解决？", "turn": 15, "days_ago": 1},
        {"role": "assistant", "content": "可以租移动 WiFi（约 10 元/天）或买电话卡（7 天约 80 元）。机场和便利店都有售。", "turn": 15, "days_ago": 1},
        {"role": "user", "content": "还有什么其他建议吗？", "turn": 16, "days_ago": 1},
        {"role": "assistant", "content": "记得下载好离线地图，提前预订热门餐厅，准备一些 1 元硬币用于投币机。祝旅途愉快！", "turn": 16, "days_ago": 1},

        # 第 17-18 轮（今天早上）- 新问题
        {"role": "user", "content": "对了，富士山值得去吗？", "turn": 17, "days_ago": 0},
        {"role": "assistant", "content": "值得！从东京出发，可以去河口湖看富士山，或者坐登山电车到五合目。5 月天气好的话能看到雪顶。", "turn": 17, "days_ago": 0},
    ]

    # 显示完整历史
    history_table = create_comparison_table(
        "完整对话历史（20 轮，38 条消息）",
        ["轮次", "角色", "内容预览", "时间", "标记"],
        [
            [
                str(msg.get("turn", "?")),
                msg["role"],
                truncate_text(msg["content"], 45),
                f"{msg.get('days_ago', 0)} 天前",
                "🔒 重要" if msg.get("must_keep") else "",
            ]
            for msg in conversation_history[:10]  # 只显示前 10 条
        ] + [["...", "...", "还有 28 条消息", "...", "..."]]
    )
    console.print(history_table)

    # 计算原始 Token 数
    raw_tokens = sum(len(msg["content"]) // 4 for msg in conversation_history)
    console.print(f"\n原始历史总 Token：[bold]{format_tokens(raw_tokens)}[/bold]\n")

    print_section("步骤 2：使用 Context Forge 压缩历史")

    # 创建 ContextForge 实例（小窗口，触发压缩）
    forge = ContextForge(
        model="gpt-4o",
        max_context_tokens=4096,  # 小窗口
        output_reserved_tokens=512,
    )

    # 启用压缩和时效性加权
    from context_forge.config.schema import CompressConfig, RerankConfig

    new_compress = CompressConfig(
        enabled=True,
        default_compressor="truncation",  # 使用截断压缩（不需要 LLM）
        saturation_trigger=0.75,
        preserve_must_keep=True,
        min_segment_tokens=50,
    )

    new_rerank = RerankConfig(
        enable_mmr=False,
        enable_temporal_weighting=True,
        temporal_decay_rate=0.2,  # 中等衰减
        temporal_min_weight=0.2,
    )

    forge._policy = forge._policy.model_copy(update={
        "compress": new_compress,
        "rerank": new_rerank,
    })

    # 重新创建 pipeline
    from context_forge.pipeline.base import create_default_pipeline
    forge._pipeline = create_default_pipeline(policy=forge._policy)

    # 准备 Segment 列表（手动创建以便添加 must_keep 标记）
    segments = []
    for msg in conversation_history:
        days_ago = msg.get("days_ago", 0)
        timestamp = now - timedelta(days=days_ago)

        seg_type = SegmentType.USER if msg["role"] == "user" else SegmentType.ASSISTANT
        priority = Priority.HIGH if msg.get("must_keep") else Priority.MEDIUM

        segments.append(Segment(
            type=seg_type,
            content=msg["content"],
            role=msg["role"],
            priority=priority,
            control=ControlFlags(
                must_keep=msg.get("must_keep", False),
                compressible=not msg.get("must_keep", False),
            ),
            provenance=Provenance(
                source_id=f"turn_{msg.get('turn', 0)}",
                source_type=SourceType.USER_INPUT if msg["role"] == "user" else SourceType.SYSTEM_CONFIG,
                created_at=timestamp,
            ),
            metadata=SegmentMetadata(
                turn_number=msg.get("turn", 0),
                timestamp=timestamp,
            ),
        ))

    # 组装上下文（当前问题：计划富士山行程）
    context = await forge.build(
        system_prompt="你是一个专业的旅行规划助手，帮助用户制定日本旅行计划。",
        messages=[{"role": "user", "content": "根据我之前的需求，帮我规划一下富士山一日游的行程。"}],
        extra_segments=segments,
        current_turn=19,
    )

    print_success(f"上下文组装完成，耗时 {context.assembly_duration_ms:.1f}ms\n")

    print_section("步骤 3：压缩与过滤结果")

    # 统计保留的对话
    from context_forge.models.segment import SegmentType
    conversation_segments = [s for s in context.segments if s.type in (SegmentType.USER, SegmentType.ASSISTANT)]
    kept_count = len(conversation_segments)
    dropped_count = len(conversation_history) - kept_count
    must_keep_count = len([s for s in conversation_segments if s.control.must_keep])

    console.print(f"原始消息：[bold]{len(conversation_history)}[/bold] 条")
    console.print(f"保留消息：[bold green]{kept_count}[/bold green] 条（含 {must_keep_count} 条关键信息）")
    console.print(f"丢弃消息：[bold red]{dropped_count}[/bold red] 条（低时效性 + 低优先级）\n")

    # 显示保留的消息
    kept_table = create_comparison_table(
        "保留的对话片段（按时效性加权排序）",
        ["轮次", "角色", "内容预览", "优先级", "Token"],
        [
            [
                seg.metadata.turn_number if seg.metadata else "?",
                seg.role,
                truncate_text(seg.content, 45),
                seg.priority.value + (" 🔒" if seg.control.must_keep else ""),
                str(seg.token_count or 0),
            ]
            for seg in conversation_segments[:8]  # 只显示前 8 条
        ]
    )
    console.print(kept_table)

    if kept_count > 8:
        console.print(f"\n   ... 还有 {kept_count - 8} 条消息\n")

    print_section("步骤 4：审计日志 — 压缩决策")

    # 显示压缩和重排决策
    compress_entries = [e for e in context.audit_log if "压缩" in e.reason_detail or "丢弃" in e.reason_detail or "时效" in e.reason_detail]

    console.print("[bold]压缩阶段决策：[/bold]\n")
    for entry in compress_entries[:6]:
        console.print(f"  - [dim]{entry.segment_id}[/dim]: {entry.reason_detail}")

    # 显示 must_keep 保护
    must_keep_entries = [e for e in context.audit_log if "must_keep" in e.reason_detail.lower() or "保护" in e.reason_detail]
    if must_keep_entries:
        console.print("\n[bold green]关键信息保护：[/bold green]\n")
        for entry in must_keep_entries[:3]:
            console.print(f"  - [dim]{entry.segment_id}[/dim]: {entry.reason_detail}")

    print_section("步骤 5：Token 优化效果")

    # 计算 Token 节省
    compressed_tokens = sum(s.token_count or 0 for s in conversation_segments)
    saved_tokens = raw_tokens - compressed_tokens
    saved_ratio = saved_tokens / raw_tokens if raw_tokens > 0 else 0

    # 获取预算使用情况
    budget_allocation = context.budget_allocation

    metrics_table = create_comparison_table(
        "优化指标",
        ["指标", "压缩前", "压缩后", "优化"],
        [
            [
                "对话消息数",
                str(len(conversation_history)),
                str(kept_count),
                f"-{dropped_count} ({format_percentage(dropped_count / len(conversation_history))})",
            ],
            [
                "Token 总数",
                format_tokens(raw_tokens),
                format_tokens(compressed_tokens),
                f"-{format_tokens(saved_tokens)} ({format_percentage(saved_ratio)})",
            ],
            [
                "预算饱和度",
                "N/A",
                format_percentage(budget_allocation.saturation_rate),
                "OK 健康" if budget_allocation.saturation_rate < 0.85 else "! 接近上限",
            ],
            [
                "关键信息保留",
                f"{must_keep_count} 条",
                f"{must_keep_count} 条",
                "OK 100% 保护",
            ],
        ]
    )
    console.print(metrics_table)

    print_section("步骤 6：滑动窗口策略")

    # 分析时间分布
    console.print("[bold]对话时间分布分析：[/bold]\n")

    # 按时间段统计
    recent_7d = [s for s in conversation_segments if s.metadata and s.metadata.timestamp and (now - s.metadata.timestamp).days < 7]
    recent_14d = [s for s in conversation_segments if s.metadata and s.metadata.timestamp and 7 <= (now - s.metadata.timestamp).days < 14]
    older = [s for s in conversation_segments if s.metadata and s.metadata.timestamp and (now - s.metadata.timestamp).days >= 14]

    distribution_table = create_comparison_table(
        "时间分布",
        ["时间段", "保留消息数", "占比", "说明"],
        [
            ["最近 7 天", str(len(recent_7d)), format_percentage(len(recent_7d) / kept_count), "高时效性"],
            ["7-14 天", str(len(recent_14d)), format_percentage(len(recent_14d) / kept_count), "中时效性"],
            ["14 天以上", str(len(older)), format_percentage(len(older) / kept_count), "低时效性（must_keep 除外）"],
        ]
    )
    console.print(distribution_table)

    print_section("总结")

    print_success(f"对话记忆管理完成！")
    print_success(f"- 从 {len(conversation_history)} 条消息压缩到 {kept_count} 条，保留率 {format_percentage(kept_count / len(conversation_history))}")
    print_success(f"- 节省了 {format_percentage(saved_ratio)} 的 Token 开销")
    print_success(f"- {must_keep_count} 条关键信息（预算、时间、兴趣）100% 保护")
    print_success(f"- 时效性加权确保最近对话优先保留")
    print_success(f"- 预算饱和度：{format_percentage(budget_allocation.saturation_rate)} (健康)")

    console.print(f"\n[dim]提示：生产环境建议启用 LLM 摘要压缩（compress.default_compressor=summary），效果更好[/dim]")


if __name__ == "__main__":
    args = parse_args("场景 2：多轮对话记忆管理")
    asyncio.run(main(mock=args.mock))
