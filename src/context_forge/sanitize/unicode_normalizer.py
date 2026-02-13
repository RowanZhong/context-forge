"""Unicode 归一化清洗插件。

统一 Unicode 表示形式，防止利用不同编码绕过检测。

→ 6.4.3.1 Unicode 归一化
"""

from __future__ import annotations

import unicodedata

from context_forge.sanitize.base import SanitizeResult


class UnicodeNormalizer:
    """Unicode 归一化清洗器。

    将文本统一转换为 NFC 形式（Canonical Decomposition + Canonical Composition），
    防止攻击者利用组合字符、零宽字符等手段绕过后续检测。

    → 6.4.3.1 Unicode 归一化

    Examples:
        >>> normalizer = UnicodeNormalizer()
        >>> result = await normalizer.sanitize("café")  # 可能是 NFC 或 NFD 编码
        >>> assert result.content == "café"  # 统一为 NFC
        >>> assert result.passed is True
    """

    def __init__(
        self,
        form: str = "NFC",
        strip_control_chars: bool = True,
        strip_zero_width: bool = True,
    ) -> None:
        """初始化 Unicode 归一化器。

        Args:
            form: Unicode 归一化形式，可选 NFC/NFD/NFKC/NFKD
            strip_control_chars: 是否剥离控制字符（C0/C1 控制字符）
            strip_zero_width: 是否剥离零宽字符（ZWSP/ZWNJ/ZWJ/ZWNBSP）

        # [Design Decision] 默认使用 NFC：
        # - NFC 是 W3C 推荐的 Web 标准形式
        # - 大多数编程语言和数据库的默认编码
        # - 对中文/日文/韩文友好（不会过度分解）
        #
        # NFKC 虽然兼容性更强，但会将全角字符转半角，可能改变语义
        # （如中文标点「」会被转为 <>，不适合 LLM 上下文）
        """
        if form not in ("NFC", "NFD", "NFKC", "NFKD"):
            raise ValueError(f"不支持的归一化形式: {form}，仅支持 NFC/NFD/NFKC/NFKD")

        self._form = form
        self._strip_control_chars = strip_control_chars
        self._strip_zero_width = strip_zero_width

    @property
    def name(self) -> str:
        """清洗器名称。"""
        return f"UnicodeNormalizer({self._form})"

    async def sanitize(self, content: str) -> SanitizeResult:
        """执行 Unicode 归一化。

        Args:
            content: 待归一化的文本

        Returns:
            SanitizeResult: 归一化后的文本，passed 始终为 True

        # [DX Decision] 此清洗器不会拒绝内容，只做规范化转换
        """
        if not content:
            return SanitizeResult(content="", passed=True)

        # 执行 Unicode 归一化
        normalized = unicodedata.normalize(self._form, content)

        # 剥离控制字符
        if self._strip_control_chars:
            normalized = self._remove_control_chars(normalized)

        # 剥离零宽字符
        if self._strip_zero_width:
            normalized = self._remove_zero_width_chars(normalized)

        # 统计修改
        changes = len(content) - len(normalized)
        metadata = {
            "original_length": len(content),
            "normalized_length": len(normalized),
            "form": self._form,
        }

        warning = None
        if changes != 0:
            warning = f"Unicode 归一化移除了 {abs(changes)} 个字符"
            metadata["characters_removed"] = abs(changes)

        return SanitizeResult(
            content=normalized,
            passed=True,
            warning=warning,
            metadata=metadata,
        )

    @staticmethod
    def _remove_control_chars(text: str) -> str:
        """移除 C0/C1 控制字符（保留换行/制表符）。

        # [Design Decision] 保留常用空白字符：
        # - \n (U+000A) 换行符
        # - \r (U+000D) 回车符
        # - \t (U+0009) 制表符
        #
        # 移除其他控制字符（U+0000-U+001F, U+0080-U+009F），防止：
        # - 终端转义序列注入
        # - 不可见字符干扰
        """
        allowed_control = {"\n", "\r", "\t"}
        return "".join(
            char for char in text
            if char in allowed_control or unicodedata.category(char) != "Cc"
        )

    @staticmethod
    def _remove_zero_width_chars(text: str) -> str:
        """移除零宽字符。

        移除以下零宽字符：
        - U+200B ZERO WIDTH SPACE (ZWSP)
        - U+200C ZERO WIDTH NON-JOINER (ZWNJ)
        - U+200D ZERO WIDTH JOINER (ZWJ)
        - U+FEFF ZERO WIDTH NO-BREAK SPACE (BOM/ZWNBSP)

        # [Design Decision] 为什么要移除零宽字符：
        # 攻击者可以利用零宽字符：
        # 1. 绕过关键词检测（在 "password" 中插入 ZWSP 变成 "pass​word"）
        # 2. 隐藏恶意指令（在正常文本中嵌入不可见的 Prompt Injection）
        # 3. 制造视觉欺骗（显示 URL 与实际 URL 不一致）
        """
        zero_width_chars = {"\u200B", "\u200C", "\u200D", "\uFEFF"}
        return "".join(char for char in text if char not in zero_width_chars)


# 🏭 生产提示：
# 1. 对于处理国际化文本（多语言混合），考虑添加：
#    - Bidi（双向文本）攻击检测（RTL Override U+202E）
#    - 同形异义字检测（如西里尔字母 'а' vs 拉丁字母 'a'）
# 2. 对于特定领域（如代码），考虑保留更多控制字符（如 ANSI 转义序列）
# 3. 性能优化：对于大文本（>100KB），考虑分块处理或使用 C 扩展库
# 4. 添加归一化冲突检测（记录被归一化的字符及其原始形式）
