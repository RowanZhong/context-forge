"""
Normalize 阶段 — 格式归一化与 Token 计数填充。

→ 6.1.2.1 Pipeline 第一阶段
→ 6.4.1.1 格式归一化

这是流水线的第一个阶段，负责：
1. 对所有 Segment 的内容做 Unicode 归一化（NFC）
2. 移除不可见的控制字符和零宽字符
3. 为每个 Segment 填充 token_count（如果尚未填充）
4. 移除空内容的 Segment

这个阶段看似简单，但在生产环境中至关重要。
Unicode 的同形字符攻击（如用西里尔字母的 "а" 代替拉丁字母 "a"）
可以绕过关键词匹配的安全检查。NFC 归一化是第一道防线。
"""

from __future__ import annotations

import logging
import re
import unicodedata

from context_forge.models.audit import AuditEntry, DecisionType, ReasonCode
from context_forge.models.segment import Segment
from context_forge.pipeline.base import PipelineContext
from context_forge.tokenizer.registry import get_tokenizer

logger = logging.getLogger(__name__)

# 不可见字符和零宽字符的匹配模式
_INVISIBLE_CHARS = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f"  # C0 控制字符（保留 \t\n\r）
    r"\u200b-\u200f"  # 零宽字符
    r"\u2028-\u2029"  # 行/段分隔符
    r"\u202a-\u202e"  # 双向文本控制
    r"\u2060-\u2064"  # 不可见格式化
    r"\ufeff"  # BOM / 零宽不间断空格
    r"\ufff9-\ufffb"  # 交互格式化
    r"]"
)


class NormalizeStage:
    """
    格式归一化阶段。

    处理内容：
    - Unicode NFC 归一化
    - 不可见字符移除
    - Token 计数填充
    - 空 Segment 过滤

    → 6.1.2.1 Pipeline 第一阶段
    """

    @property
    def name(self) -> str:
        return "normalize"

    async def process(
        self,
        segments: list[Segment],
        context: PipelineContext,
    ) -> list[Segment]:
        """归一化所有 Segment 的内容并填充 Token 计数。"""
        counter = get_tokenizer(context.model)
        result: list[Segment] = []

        for seg in segments:
            # 1. Unicode NFC 归一化
            normalized_content = unicodedata.normalize("NFC", seg.content)

            # 2. 移除不可见字符
            cleaned_content = _INVISIBLE_CHARS.sub("", normalized_content)

            # 记录内容是否被修改
            if cleaned_content != seg.content:
                context.audit_log.append(AuditEntry(
                    segment_id=seg.id,
                    decision=DecisionType.SANITIZE,
                    reason_code=ReasonCode.SANITIZE_UNICODE_NORMALIZED,
                    reason_detail=(
                        f"Unicode 归一化并移除了 "
                        f"{len(seg.content) - len(cleaned_content)} 个不可见字符。"
                    ),
                    pipeline_stage=self.name,
                ))

            # 3. 过滤空 Segment
            stripped = cleaned_content.strip()
            if not stripped:
                context.audit_log.append(AuditEntry(
                    segment_id=seg.id,
                    decision=DecisionType.DROP,
                    reason_code=ReasonCode.SELECT_LOW_RELEVANCE,
                    reason_detail="Segment 内容为空（归一化后仅含空白字符）。",
                    pipeline_stage=self.name,
                ))
                continue

            # 4. 填充 Token 计数
            new_seg = seg.with_content(cleaned_content)
            token_count = counter.count(cleaned_content)
            new_seg = new_seg.with_token_count(token_count)

            result.append(new_seg)

        if context.debug:
            total_tokens = sum(s.token_count or 0 for s in result)
            logger.debug(
                "[normalize] %d → %d Segment，总计 %d tokens",
                len(segments),
                len(result),
                total_tokens,
            )

        return result
