"""零信任清洗层。

本模块提供完整的清洗插件体系，用于处理不可信的用户输入。

→ 6.4 零信任清洗管道

核心组件：
- SanitizerChain: 清洗器责任链编排器
- UnicodeNormalizer: Unicode 归一化
- HTMLStripper: HTML 标签剥离
- MarkdownStripper: Markdown 格式剥离
- PIIRedactor: PII 脱敏
- InjectionDetector: Prompt Injection 检测
- LengthGuard: 长度攻击防御

快速开始：
    >>> from context_forge.sanitize import create_default_chain
    >>> chain = create_default_chain()
    >>> result = await chain.process("用户输入内容")
    >>> if not result.passed:
    ...     raise SanitizationError(result.warning)
"""

from context_forge.sanitize.base import Sanitizer, SanitizerChain, SanitizeResult
from context_forge.sanitize.html_stripper import HTMLStripper, MarkdownStripper
from context_forge.sanitize.injection_detector import DetectionLevel, InjectionDetector
from context_forge.sanitize.length_guard import LengthGuard
from context_forge.sanitize.pii_redactor import PIIRedactor, PIIType
from context_forge.sanitize.unicode_normalizer import UnicodeNormalizer

__all__ = [
    # 基础协议
    "Sanitizer",
    "SanitizeResult",
    "SanitizerChain",
    # 清洗插件
    "UnicodeNormalizer",
    "HTMLStripper",
    "MarkdownStripper",
    "PIIRedactor",
    "PIIType",
    "InjectionDetector",
    "DetectionLevel",
    "LengthGuard",
    # 工厂函数
    "create_default_chain",
]


def create_default_chain(
    *,
    enable_pii_redaction: bool = True,
    enable_injection_detection: bool = True,
    injection_level: DetectionLevel = DetectionLevel.STANDARD,
    max_chars: int = 100_000,
) -> SanitizerChain:
    """创建默认清洗链。

    按推荐顺序组合所有清洗器，适用于大多数场景。

    Args:
        enable_pii_redaction: 是否启用 PII 脱敏
        enable_injection_detection: 是否启用 Injection 检测
        injection_level: Injection 检测级别
        max_chars: 最大字符数限制

    Returns:
        SanitizerChain: 配置好的清洗链

    # [Design Decision] 推荐的清洗顺序：
    # 1. UnicodeNormalizer: 预处理，统一编码
    # 2. LengthGuard: 早期拒绝超长内容，节省后续开销
    # 3. HTMLStripper: 移除结构标记
    # 4. PIIRedactor: 脱敏敏感信息（可选）
    # 5. InjectionDetector: 最后检测攻击模式（可选）
    #
    # 为什么 Injection 检测放最后：
    # - 攻击者可能利用 HTML/Unicode 编码隐藏攻击模式
    # - 先清洗再检测，避免绕过

    Examples:
        >>> # 最小清洗（仅归一化 + 长度限制）
        >>> chain = create_default_chain(
        ...     enable_pii_redaction=False,
        ...     enable_injection_detection=False,
        ... )

        >>> # 严格清洗（启用所有检测）
        >>> chain = create_default_chain(
        ...     injection_level=DetectionLevel.STRICT,
        ... )
    """
    sanitizers: list[Sanitizer] = []

    # 1. Unicode 归一化（必选）
    sanitizers.append(UnicodeNormalizer(
        form="NFC",
        strip_control_chars=True,
        strip_zero_width=True,
    ))

    # 2. 长度防御（必选）
    sanitizers.append(LengthGuard(
        max_chars=max_chars,
        max_lines=10_000,
        max_line_length=10_000,
        max_repetition_ratio=0.8,
        truncate_on_overflow=False,  # 默认拒绝而非截断
    ))

    # 3. HTML 剥离（必选）
    sanitizers.append(HTMLStripper(
        mode="strip",
        preserve_whitespace=True,
    ))

    # 4. PII 脱敏（可选）
    if enable_pii_redaction:
        sanitizers.append(PIIRedactor(
            enabled_types=None,  # 全部启用
            redaction_char="*",
        ))

    # 5. Injection 检测（可选）
    if enable_injection_detection:
        sanitizers.append(InjectionDetector(
            level=injection_level,
            block_on_detection=True,
        ))

    return SanitizerChain(sanitizers)


# 🏭 生产提示：
# 1. 场景化预设：
#    - create_rag_chain(): RAG 场景（严格清洗 + PII 脱敏）
#    - create_chat_chain(): 聊天场景（宽松清洗 + 保留格式）
#    - create_code_chain(): 代码场景（保留特殊字符 + 语法高亮）
#
# 2. 动态配置：
#    - 从 PolicyConfig 读取清洗策略
#    - 支持按 SegmentType 选择不同的清洗链
#
# 3. 插件注册表：
#    - 支持用户注册自定义 Sanitizer
#    - 通过配置文件声明插件顺序
#
# 4. 性能监控：
#    - 记录每个清洗器的执行时间
#    - 识别性能瓶颈
#
# 5. A/B 测试：
#    - 支持多个清洗链并行执行
#    - 比较不同策略的效果
