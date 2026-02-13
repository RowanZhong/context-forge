"""PII（个人身份信息）脱敏清洗插件。

检测并脱敏敏感个人信息，防止隐私泄露。

→ 6.4.3.3 PII 脱敏
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from context_forge.sanitize.base import SanitizeResult


class PIIType(Enum):
    """PII 类型枚举。"""
    PHONE = "phone"  # 手机号
    EMAIL = "email"  # 邮箱
    ID_CARD = "id_card"  # 身份证号
    BANK_CARD = "bank_card"  # 银行卡号
    IP_ADDRESS = "ip_address"  # IP 地址
    URL = "url"  # URL（可能包含敏感信息）


@dataclass(frozen=True)
class PIIMatch:
    """PII 匹配结果。"""
    pii_type: PIIType
    original: str  # 原始文本
    start: int  # 起始位置
    end: int  # 结束位置


class PIIRedactor:
    """PII 脱敏清洗器。

    检测并替换文本中的个人身份信息，支持中国大陆常见的 PII 格式。

    → 6.4.3.3 PII 脱敏

    Examples:
        >>> redactor = PIIRedactor()
        >>> result = await redactor.sanitize("我的手机是 13800138000")
        >>> assert "138****8000" in result.content

        >>> result = await redactor.sanitize("身份证号：110101199001011234")
        >>> assert "110101********1234" in result.content
    """

    def __init__(
        self,
        enabled_types: set[PIIType] | None = None,
        redaction_char: str = "*",
    ) -> None:
        """初始化 PII 脱敏器。

        Args:
            enabled_types: 启用的 PII 类型集合，None 表示全部启用
            redaction_char: 用于替换的字符

        # [Design Decision] 默认启用所有 PII 类型：
        # 遵循零信任原则，宁可过度脱敏也不遗漏敏感信息
        """
        self._enabled_types = enabled_types or set(PIIType)
        self._redaction_char = redaction_char

        # 预编译正则表达式（性能优化）
        self._patterns = self._build_patterns()

    @property
    def name(self) -> str:
        """清洗器名称。"""
        return "PIIRedactor"

    def _build_patterns(self) -> dict[PIIType, re.Pattern]:
        """构建 PII 检测正则表达式。

        # [Design Decision] 正则表达式设计原则：
        # 1. 使用负向前瞻/后顾避免误匹配（如电话号码不应该是纯数字序列的一部分）
        # 2. 考虑中国大陆特定格式（手机号、身份证、银行卡）
        # 3. 宽松匹配（高召回率）优先于精确匹配，减少漏报
        """
        patterns = {}

        # 中国大陆手机号：1[3-9]\d{9}
        # 负向前瞻/后顾：确保不是更长数字序列的一部分
        if PIIType.PHONE in self._enabled_types:
            patterns[PIIType.PHONE] = re.compile(
                r"(?<!\d)"  # 前面不能是数字
                r"1[3-9]\d{9}"  # 1开头，第二位3-9，后面9位数字
                r"(?!\d)"  # 后面不能是数字
            )

        # 邮箱地址：标准 RFC 5322 简化版
        if PIIType.EMAIL in self._enabled_types:
            patterns[PIIType.EMAIL] = re.compile(
                r"\b"  # 单词边界
                r"[a-zA-Z0-9._%+-]+"  # 用户名部分
                r"@"
                r"[a-zA-Z0-9.-]+"  # 域名部分
                r"\.[a-zA-Z]{2,}"  # 顶级域名
                r"\b"
            )

        # 中国大陆身份证号：18位（或15位旧版）
        # 格式：6位地区码 + 8位生日 + 3位顺序码 + 1位校验码
        if PIIType.ID_CARD in self._enabled_types:
            patterns[PIIType.ID_CARD] = re.compile(
                r"(?<!\d)"
                r"[1-9]\d{5}"  # 地区码（不以0开头）
                r"(?:19|20)\d{2}"  # 年份（1900-2099）
                r"(?:0[1-9]|1[0-2])"  # 月份（01-12）
                r"(?:0[1-9]|[12]\d|3[01])"  # 日期（01-31）
                r"\d{3}"  # 顺序码
                r"[\dXx]"  # 校验码（数字或X）
                r"(?!\d)"
            )

        # 银行卡号：13-19位数字（符合国际标准）
        # 使用 Luhn 算法验证会更准确，但正则足够处理大多数场景
        if PIIType.BANK_CARD in self._enabled_types:
            patterns[PIIType.BANK_CARD] = re.compile(
                r"(?<!\d)"
                r"\d{13,19}"  # 13-19位数字
                r"(?!\d)"
            )

        # IP 地址：IPv4（简化版，不做严格范围校验）
        if PIIType.IP_ADDRESS in self._enabled_types:
            patterns[PIIType.IP_ADDRESS] = re.compile(
                r"\b"
                r"(?:\d{1,3}\.){3}\d{1,3}"  # 四组数字用点分隔
                r"\b"
            )

        # URL：http/https 开头
        if PIIType.URL in self._enabled_types:
            patterns[PIIType.URL] = re.compile(
                r"\b"
                r"(?:https?://)"  # 协议
                r"(?:[a-zA-Z0-9-]+\.)*"  # 子域名（可选）
                r"[a-zA-Z0-9-]+"  # 域名
                r"(?:\.[a-zA-Z]{2,})?"  # 顶级域名（可选，用于 localhost）
                r"(?::\d+)?"  # 端口（可选）
                r"(?:/[^\s]*)?"  # 路径（可选）
                r"\b"
            )

        return patterns

    async def sanitize(self, content: str) -> SanitizeResult:
        """检测并脱敏 PII。

        Args:
            content: 待处理的文本

        Returns:
            SanitizeResult: 脱敏后的文本，passed 始终为 True

        # [DX Decision] 脱敏不拒绝内容：
        # 与 Injection 检测不同，PII 脱敏是转换而非拒绝，因此 passed 始终为 True
        """
        if not content:
            return SanitizeResult(content="", passed=True)

        # 收集所有匹配项
        matches: list[PIIMatch] = []
        for pii_type, pattern in self._patterns.items():
            for match in pattern.finditer(content):
                matches.append(PIIMatch(
                    pii_type=pii_type,
                    original=match.group(),
                    start=match.start(),
                    end=match.end(),
                ))

        # 按位置排序（从后往前替换，避免位置偏移）
        matches.sort(key=lambda m: m.start, reverse=True)

        # 执行脱敏
        redacted = content
        redaction_stats: dict[str, int] = {}

        for match in matches:
            redacted_text = self._redact(match)
            redacted = (
                redacted[:match.start]
                + redacted_text
                + redacted[match.end:]
            )
            # 统计
            pii_type_name = match.pii_type.value
            redaction_stats[pii_type_name] = redaction_stats.get(pii_type_name, 0) + 1

        metadata = {
            "total_redactions": len(matches),
            "redactions_by_type": redaction_stats,
        }

        warning = None
        if matches:
            warning = f"脱敏了 {len(matches)} 处 PII 信息：{redaction_stats}"

        return SanitizeResult(
            content=redacted,
            passed=True,
            warning=warning,
            metadata=metadata,
        )

    def _redact(self, match: PIIMatch) -> str:
        """根据 PII 类型生成脱敏文本。

        # [Design Decision] 保留部分信息用于识别：
        # - 手机号：保留前3位和后4位（138****8000）
        # - 邮箱：保留首尾字符和域名（a***b@example.com）
        # - 身份证：保留前6位（地区）和后4位（110101********1234）
        # - 银行卡：保留前6位（BIN）和后4位（6225********1234）
        # - IP：保留首段（192.*.*.*）
        # - URL：替换为通用占位符
        """
        original = match.original

        if match.pii_type == PIIType.PHONE:
            # 138****8000
            return original[:3] + self._redaction_char * 4 + original[-4:]

        elif match.pii_type == PIIType.EMAIL:
            # a***b@example.com
            local, domain = original.split("@", 1)
            if len(local) <= 2:
                redacted_local = self._redaction_char * len(local)
            else:
                redacted_local = local[0] + self._redaction_char * (len(local) - 2) + local[-1]
            return f"{redacted_local}@{domain}"

        elif match.pii_type == PIIType.ID_CARD:
            # 110101********1234
            return original[:6] + self._redaction_char * 8 + original[-4:]

        elif match.pii_type == PIIType.BANK_CARD:
            # 6225********1234
            if len(original) <= 10:
                # 短卡号（如13位），保留前4后4
                return original[:4] + self._redaction_char * (len(original) - 8) + original[-4:]
            else:
                # 长卡号（如16-19位），保留前6后4
                return original[:6] + self._redaction_char * (len(original) - 10) + original[-4:]

        elif match.pii_type == PIIType.IP_ADDRESS:
            # 192.*.*.*
            parts = original.split(".")
            return parts[0] + "." + ".".join(self._redaction_char for _ in parts[1:])

        elif match.pii_type == PIIType.URL:
            # [REDACTED_URL]
            return "[REDACTED_URL]"

        else:
            # 默认：完全替换
            return self._redaction_char * len(original)


# 🏭 生产提示：
# 1. 身份证号校验：添加 Luhn 算法验证校验码，减少误报
# 2. 银行卡号校验：使用标准的 Luhn 算法验证，提高准确率
# 3. 扩展 PII 类型：
#    - 护照号码（如中国护照：E/G/P + 8位数字）
#    - 车牌号码（如京A12345）
#    - 社保号码
#    - 驾驶证号码
# 4. 国际化支持：
#    - 美国 SSN（xxx-xx-xxxx）
#    - 欧盟 GDPR 相关 PII
#    - 其他国家/地区的身份证号码格式
# 5. 高级检测：
#    - 使用 NER 模型（如 spaCy）检测姓名、地址等非结构化 PII
#    - 使用 ML 模型检测上下文中的敏感信息
# 6. 可逆脱敏：
#    - 对于需要恢复原始数据的场景，使用加密而非简单替换
#    - 建立 Token 映射表（原始值 → Token ID）
# 7. 审计日志：
#    - 记录脱敏操作的详细信息（时间、类型、位置）
#    - 用于合规审计和事后分析
