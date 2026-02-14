"""
Sanitize 阶段 — 零信任清洗。

→ 6.1.2.1 Pipeline 第二阶段
→ 6.4 上下文清洗与零信任安全

零信任原则：**所有来自外部的内容都视为不可信**，
包括用户输入、RAG 检索结果、工具返回值。

本阶段委托给 sanitize/ 模块中的责任链编排器，支持可插拔的清洗器。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from context_forge.models.audit import AuditEntry, DecisionType, ReasonCode

if TYPE_CHECKING:
    from context_forge.models.segment import Segment
    from context_forge.pipeline.base import PipelineContext
    from context_forge.sanitize.base import Sanitizer, SanitizeResult, SanitizerChain

logger = logging.getLogger(__name__)


class SanitizeStage:
    """
    零信任清洗阶段。

    → 6.4 上下文清洗与零信任安全

    第二轮增强版：集成 SanitizerChain，委托给可插拔的清洗器。
    同时保持向后兼容，支持旧版简单参数（自动映射到对应插件）。

    高级用法（自定义清洗链）::

        from context_forge.sanitize.base import SanitizerChain
        from context_forge.sanitize.unicode_normalizer import UnicodeNormalizer
        from context_forge.sanitize.html_stripper import HTMLStripper
        from context_forge.sanitize.injection_detector import HeuristicInjectionDetector

        chain = SanitizerChain([
            UnicodeNormalizer(),
            HTMLStripper(),
            HeuristicInjectionDetector(on_detected="warn_and_remove"),
        ])
        stage = SanitizeStage(sanitizer_chain=chain)

    向后兼容用法（简单参数）::

        # 旧版参数仍然工作，内部自动构建清洗链
        stage = SanitizeStage(
            strip_html=True,
            detect_injection=True,
            on_injection="warn_and_remove",
        )
    """

    def __init__(
        self,
        sanitizer_chain: SanitizerChain | None = None,
        # 向后兼容参数（仅在 sanitizer_chain=None 时生效）
        max_segment_chars: int = 50_000,
        strip_html: bool = True,
        detect_injection: bool = True,
        on_injection: str = "warn_and_remove",
        injection_level: str = "heuristic",
        injection_confidence_threshold: float = 0.7,
        pii_redaction: bool = False,
        pii_patterns: list[str] | None = None,
        max_repeat_chars: int = 100,
    ) -> None:
        """
        初始化 Sanitize 阶段。

        参数:
            sanitizer_chain: 自定义清洗链（高级用法）
            max_segment_chars: 单个 Segment 最大字符数
            strip_html: 是否剥离 HTML 标签（向后兼容）
            detect_injection: 是否启用 Injection 检测（向后兼容）
            on_injection: Injection 处理策略（向后兼容）
            injection_level: Injection 检测级别：heuristic / classifier
            injection_confidence_threshold: 检测置信度阈值（classifier 模式）
            pii_redaction: 是否启用 PII 脱敏（向后兼容）
            pii_patterns: PII 类型列表（向后兼容）
            max_repeat_chars: 最大重复字符数（向后兼容）
        """
        self._max_chars = max_segment_chars
        self._max_repeat_chars = max_repeat_chars

        # [DX Decision] 如果用户提供了 sanitizer_chain，直接使用
        if sanitizer_chain is not None:
            self._chain = sanitizer_chain
        else:
            # 否则根据向后兼容参数构建默认清洗链
            self._chain = self._build_default_chain(
                strip_html=strip_html,
                detect_injection=detect_injection,
                on_injection=on_injection,
                injection_level=injection_level,
                injection_confidence_threshold=injection_confidence_threshold,
                pii_redaction=pii_redaction,
                pii_patterns=pii_patterns or ["phone", "email", "id_card"],
            )

    def _build_default_chain(
        self,
        strip_html: bool,
        detect_injection: bool,
        on_injection: str,
        injection_level: str,
        injection_confidence_threshold: float,
        pii_redaction: bool,
        pii_patterns: list[str],
    ) -> SanitizerChain:
        """
        根据简单参数构建默认清洗链。

        → 6.4.1 零信任清洗管道架构

        推荐顺序：
        1. Unicode 归一化（预处理）
        2. HTML/Markdown 剥离（结构清理）
        3. PII 脱敏（隐私保护）
        4. Injection 检测（安全检查）
        5. 长度防御（资源保护）
        """
        from context_forge.sanitize.base import SanitizerChain
        from context_forge.sanitize.html_stripper import HTMLStripper
        from context_forge.sanitize.injection_detector import DetectionLevel, InjectionDetector
        from context_forge.sanitize.length_guard import LengthGuard
        from context_forge.sanitize.pii_redactor import PIIRedactor
        from context_forge.sanitize.unicode_normalizer import UnicodeNormalizer

        sanitizers: list[Sanitizer] = []

        # 1. Unicode 归一化（始终启用）
        sanitizers.append(UnicodeNormalizer())

        # 2. HTML 剥离
        if strip_html:
            sanitizers.append(HTMLStripper())

        # 3. PII 脱敏
        if pii_redaction:
            # 将 pattern 名称转换为 PIIType
            from context_forge.sanitize.pii_redactor import PIIType
            _pii_name_map = {
                "phone": PIIType.PHONE,
                "email": PIIType.EMAIL,
                "id_card": PIIType.ID_CARD,
                "bank_card": PIIType.BANK_CARD,
                "ip_address": PIIType.IP_ADDRESS,
                "url": PIIType.URL,
            }
            enabled_types = set()
            for pattern in pii_patterns:
                if pattern in _pii_name_map:
                    enabled_types.add(_pii_name_map[pattern])
                else:
                    logger.warning(
                        "未知的 PII 模式 '%s'，已忽略。支持的模式：%s",
                        pattern,
                        ", ".join(sorted(_pii_name_map.keys())),
                    )
            sanitizers.append(PIIRedactor(enabled_types=enabled_types if enabled_types else None))

        # 4. Injection 检测
        if detect_injection:
            # [DX Decision] on_injection 参数映射到 block_on_detection
            block_on_detection = on_injection in ("warn_and_remove", "error")

            if injection_level == "heuristic":
                sanitizers.append(InjectionDetector(
                    level=DetectionLevel.STANDARD,
                    block_on_detection=block_on_detection,
                ))
            elif injection_level == "classifier":
                # → 6.4.3 基于 LLM 的高级检测（可选）
                # 🏭 生产提示：需要接入 LLM Provider
                logger.warning(
                    "injection_level='classifier' 需要 LLM 支持，"
                    "当前降级为 heuristic 模式。"
                )
                sanitizers.append(InjectionDetector(
                    level=DetectionLevel.STANDARD,
                    block_on_detection=block_on_detection,
                ))

        # 5. 长度防御
        # [Design Decision] LengthGuard 使用 max_repetition_ratio 而非 max_repeat_chars
        # 将 max_repeat_chars 转换为合理的 repetition_ratio
        sanitizers.append(LengthGuard(
            max_chars=self._max_chars,
            truncate_on_overflow=True,  # 截断而非拒绝，保持向后兼容
        ))

        return SanitizerChain(sanitizers)

    @property
    def name(self) -> str:
        return "sanitize"

    async def process(
        self,
        segments: list[Segment],
        context: PipelineContext,
    ) -> list[Segment]:
        """
        对所有 Segment 执行零信任清洗。

        → 6.4.1 零信任清洗管道

        流程：
        1. 遍历每个 Segment
        2. 通过 SanitizerChain 处理内容
        3. 根据清洗结果决定保留/移除/修改
        4. 记录审计日志
        """
        result: list[Segment] = []

        for seg in segments:
            # 调用清洗链处理内容
            try:
                sanitize_result: SanitizeResult = await self._chain.process(seg.content)
            except Exception as e:
                # 清洗失败，记录警告并跳过该 Segment
                logger.error(
                    "[sanitize] 清洗 Segment %s 时失败：%s",
                    seg.id,
                    e,
                )
                context.warnings.append(
                    f"[清洗] Segment {seg.id} 清洗失败，已移除：{e}"
                )
                context.audit_log.append(AuditEntry(
                    segment_id=seg.id,
                    decision=DecisionType.DROP,
                    reason_code=ReasonCode.SANITIZE_FAILED,
                    reason_detail=f"清洗过程中发生错误：{e}",
                    pipeline_stage=self.name,
                ))
                continue

            # 检查清洗结果
            if not sanitize_result.passed:
                # 未通过清洗（检测到威胁），移除该 Segment
                logger.warning(
                    "[sanitize] Segment %s 未通过清洗检查，已移除",
                    seg.id,
                )
                context.warnings.append(
                    f"[安全] Segment {seg.id} 未通过清洗检查：{sanitize_result.warning}"
                )
                context.audit_log.append(AuditEntry(
                    segment_id=seg.id,
                    decision=DecisionType.DROP,
                    reason_code=ReasonCode.SANITIZE_INJECTION_DETECTED,
                    reason_detail=sanitize_result.warning or "检测到安全威胁。",
                    pipeline_stage=self.name,
                ))
                continue

            # 内容已修改，更新 Segment
            if sanitize_result.content != seg.content:
                seg = seg.with_content(sanitize_result.content)
                context.audit_log.append(AuditEntry(
                    segment_id=seg.id,
                    decision=DecisionType.SANITIZE,
                    reason_code=ReasonCode.SANITIZE_HTML_STRIPPED,
                    reason_detail=sanitize_result.warning or "内容已清洗。",
                    pipeline_stage=self.name,
                ))

            # 保留该 Segment
            result.append(seg)

        if context.debug:
            logger.debug(
                "[sanitize] %d → %d Segment（移除 %d 个）",
                len(segments),
                len(result),
                len(segments) - len(result),
            )

        return result
