"""
Assemble 阶段 — 最终组装与快照生成。

→ 6.1.2.1 Pipeline 第五阶段（最后阶段）
→ 6.5.1 Context Snapshot

这是流水线的最后一个阶段，负责：
1. 按照 LLM API 要求的格式组织 Segment 顺序
2. 合并同角色的相邻消息（部分 API 不允许连续同角色消息）
3. 为每个保留的 Segment 记录 KEEP 审计条目
4. 生成最终的决策摘要
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from context_forge.models.audit import AuditEntry, DecisionType, ReasonCode
from context_forge.models.segment import Segment, SegmentType

if TYPE_CHECKING:
    from context_forge.pipeline.base import PipelineContext

logger = logging.getLogger(__name__)

# Segment 类型到组装顺序的映射
# [Design Decision] 系统提示永远在最前面，工具定义紧随其后，
# 然后是对话历史（按时间顺序），最后是状态锚点和当前用户消息。
_TYPE_ORDER: dict[SegmentType, int] = {
    SegmentType.SYSTEM: 0,
    SegmentType.SCHEMA: 1,
    SegmentType.TOOL_DEFINITION: 2,
    SegmentType.FEW_SHOT: 3,
    SegmentType.SUMMARY: 4,
    SegmentType.STATE: 5,
    # 对话历史按时间顺序排列，不在此处固定
    SegmentType.USER: 10,
    SegmentType.ASSISTANT: 10,
    SegmentType.RAG: 10,
    SegmentType.TOOL_CALL: 10,
    SegmentType.TOOL_RESULT: 10,
}


class AssembleStage:
    """
    最终组装阶段。

    → 6.1.2.1 Pipeline 第五阶段

    按 LLM API 规范组织 Segment 顺序并生成组装快照。
    """

    def __init__(self, merge_adjacent: bool = False) -> None:
        """
        参数:
            merge_adjacent: 是否合并相邻的同角色消息
        """
        self._merge_adjacent = merge_adjacent

    @property
    def name(self) -> str:
        return "assemble"

    async def process(
        self,
        segments: list[Segment],
        context: PipelineContext,
    ) -> list[Segment]:
        """执行最终组装。"""
        # 1. 分离固定位置和动态位置的 Segment
        fixed: list[Segment] = []
        dynamic: list[Segment] = []

        for seg in segments:
            if seg.control.lock_position or seg.type in (
                SegmentType.SYSTEM,
                SegmentType.SCHEMA,
                SegmentType.TOOL_DEFINITION,
                SegmentType.FEW_SHOT,
            ):
                fixed.append(seg)
            else:
                dynamic.append(seg)

        # 2. 固定位置 Segment 按类型顺序排列
        fixed.sort(key=lambda s: _TYPE_ORDER.get(s.type, 99))

        # 3. 动态 Segment 保持输入顺序（通常是时间顺序）
        # 已经由 Rerank 阶段排好了

        # 4. 合并
        assembled = fixed + dynamic

        # 5. 可选：合并相邻同角色消息
        if self._merge_adjacent:
            assembled = _merge_adjacent_messages(assembled)

        # 6. 为每个保留的 Segment 记录审计
        for seg in assembled:
            context.audit_log.append(AuditEntry(
                segment_id=seg.id,
                decision=DecisionType.KEEP,
                reason_code=ReasonCode.BUDGET_EXCEEDED,  # 复用：表示成功通过预算检查
                reason_detail=f"最终保留：{seg.token_count or 0:,} tokens，类型 {seg.type.value}。",
                pipeline_stage=self.name,
                token_impact=seg.token_count or 0,
            ))

        if context.debug:
            total_tokens = sum(s.token_count or 0 for s in assembled)
            type_counts = {}
            for s in assembled:
                type_counts[s.type.value] = type_counts.get(s.type.value, 0) + 1
            logger.debug(
                "[assemble] 最终组装 %d 个 Segment，%d tokens。类型分布：%s",
                len(assembled),
                total_tokens,
                type_counts,
            )

        return assembled


def _merge_adjacent_messages(segments: list[Segment]) -> list[Segment]:
    """
    合并相邻的同角色消息。

    某些 LLM API（如 OpenAI）不允许连续出现同角色消息。
    此函数将相邻的同角色 Segment 合并为一个。
    """
    if not segments:
        return []

    result: list[Segment] = [segments[0]]

    for seg in segments[1:]:
        last = result[-1]
        if seg.role == last.role and not last.control.lock_position:
            # 合并内容
            merged_content = last.content + "\n\n" + seg.content
            merged_tokens = (last.token_count or 0) + (seg.token_count or 0)
            merged = last.with_content(merged_content).with_token_count(merged_tokens)
            result[-1] = merged
        else:
            result.append(seg)

    return result
