"""HTML/Markdown 剥离清洗插件。

移除或转义 HTML 标签和 Markdown 语法，防止注入和格式混淆。

→ 6.4.3.2 HTML/Markdown 清理
"""

from __future__ import annotations

import html
import re

from context_forge.sanitize.base import SanitizeResult


class HTMLStripper:
    """HTML 标签剥离器。

    移除 HTML 标签，保留纯文本内容。可选择完全剥离或转义为安全文本。

    → 6.4.3.2 HTML/Markdown 清理

    Examples:
        >>> stripper = HTMLStripper(mode="strip")
        >>> result = await stripper.sanitize("<script>alert('xss')</script>Hello")
        >>> assert result.content == "Hello"

        >>> stripper = HTMLStripper(mode="escape")
        >>> result = await stripper.sanitize("<b>Bold</b>")
        >>> assert result.content == "&lt;b&gt;Bold&lt;/b&gt;"
    """

    def __init__(
        self,
        mode: str = "strip",
        preserve_whitespace: bool = True,
    ) -> None:
        """初始化 HTML 剥离器。

        Args:
            mode: 处理模式，可选 "strip"（删除标签）或 "escape"（转义标签）
            preserve_whitespace: 剥离标签后是否保留空格（避免单词粘连）

        # [Design Decision] 默认使用 strip 模式：
        # - LLM 上下文通常不需要 HTML 标记
        # - 剥离可以减少 Token 消耗
        #
        # escape 模式适用于需要保留原始内容用于审计的场景
        """
        if mode not in ("strip", "escape"):
            raise ValueError(f"不支持的模式: {mode}，仅支持 'strip' 或 'escape'")

        self._mode = mode
        self._preserve_whitespace = preserve_whitespace

        # 预编译正则表达式（性能优化）
        # [Design Decision] 使用正则而非 HTML 解析器：
        # - 更轻量，无需依赖 lxml/beautifulsoup4
        # - 对畸形 HTML 有更好的容错性
        # - 足够处理大多数清洗场景
        self._tag_pattern = re.compile(r"<[^>]+>")
        self._script_style_pattern = re.compile(
            r"<(script|style)[^>]*>.*?</\1>",
            re.IGNORECASE | re.DOTALL,
        )
        self._comment_pattern = re.compile(r"<!--.*?-->", re.DOTALL)

    @property
    def name(self) -> str:
        """清洗器名称。"""
        return f"HTMLStripper({self._mode})"

    async def sanitize(self, content: str) -> SanitizeResult:
        """剥离或转义 HTML 标签。

        Args:
            content: 待处理的文本

        Returns:
            SanitizeResult: 处理后的文本，passed 始终为 True
        """
        if not content:
            return SanitizeResult(content="", passed=True)

        original_length = len(content)

        if self._mode == "escape":
            # 转义所有 HTML 特殊字符
            cleaned = html.escape(content)
        else:
            # 剥离 HTML 标签
            cleaned = self._strip_html(content)

        cleaned_length = len(cleaned)
        metadata = {
            "original_length": original_length,
            "cleaned_length": cleaned_length,
            "mode": self._mode,
        }

        warning = None
        if original_length != cleaned_length:
            removed = original_length - cleaned_length
            warning = f"移除了 {removed} 个字符的 HTML 内容"
            metadata["characters_removed"] = removed

        return SanitizeResult(
            content=cleaned,
            passed=True,
            warning=warning,
            metadata=metadata,
        )

    def _strip_html(self, content: str) -> str:
        """剥离 HTML 标签（保留文本内容）。"""
        # 1. 移除 <script> 和 <style> 标签及其内容
        cleaned = self._script_style_pattern.sub("", content)

        # 2. 移除 HTML 注释
        cleaned = self._comment_pattern.sub("", cleaned)

        # 3. 移除其他 HTML 标签
        if self._preserve_whitespace:
            # 将标签替换为空格（避免单词粘连）
            cleaned = self._tag_pattern.sub(" ", cleaned)
        else:
            cleaned = self._tag_pattern.sub("", cleaned)

        # 4. 解码 HTML 实体（如 &nbsp; → 空格）
        cleaned = html.unescape(cleaned)

        # 5. 规范化空白字符（多个空格/换行合并为一个）
        cleaned = re.sub(r"\s+", " ", cleaned)

        return cleaned.strip()


class MarkdownStripper:
    """Markdown 语法剥离器。

    移除 Markdown 格式标记，保留纯文本内容。

    → 6.4.3.2 HTML/Markdown 清理

    Examples:
        >>> stripper = MarkdownStripper()
        >>> result = await stripper.sanitize("# Title\\n**Bold** _italic_")
        >>> assert result.content == "Title Bold italic"
    """

    def __init__(self, preserve_code: bool = True) -> None:
        """初始化 Markdown 剥离器。

        Args:
            preserve_code: 是否保留代码块内容（去除 ``` 标记但保留代码）

        # [Design Decision] 默认保留代码内容：
        # - 代码块通常包含重要上下文（如用户的代码片段）
        # - 只移除 Markdown 标记而非代码本身
        """
        self._preserve_code = preserve_code

        # 预编译 Markdown 模式
        self._patterns = {
            # 标题标记（# ## ###）
            "heading": re.compile(r"^#+\s*", re.MULTILINE),
            # 粗体/斜体（**bold** *italic* __bold__ _italic_）
            "emphasis": re.compile(r"(\*\*|__)(.*?)\1|\*|_"),
            # 链接（[text](url)）
            "link": re.compile(r"\[([^\]]+)\]\([^)]+\)"),
            # 图片（![alt](url)）
            "image": re.compile(r"!\[([^\]]*)\]\([^)]+\)"),
            # 代码块（```code```）
            "code_block": re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL),
            # 行内代码（`code`）
            "inline_code": re.compile(r"`([^`]+)`"),
            # 引用（> quote）
            "blockquote": re.compile(r"^>\s*", re.MULTILINE),
            # 列表标记（- * + 1.）
            "list": re.compile(r"^[\s]*[-*+]\s+|^\s*\d+\.\s+", re.MULTILINE),
            # 水平线（--- *** ___）
            "hr": re.compile(r"^(\-{3,}|\*{3,}|_{3,})$", re.MULTILINE),
        }

    @property
    def name(self) -> str:
        """清洗器名称。"""
        return "MarkdownStripper"

    async def sanitize(self, content: str) -> SanitizeResult:
        """剥离 Markdown 格式标记。

        Args:
            content: 待处理的文本

        Returns:
            SanitizeResult: 处理后的文本，passed 始终为 True
        """
        if not content:
            return SanitizeResult(content="", passed=True)

        original_length = len(content)
        cleaned = content

        # 按顺序应用剥离规则
        # 1. 代码块（需要先处理，避免代码内的 Markdown 语法被误处理）
        if self._preserve_code:
            cleaned = self._patterns["code_block"].sub(r"\1", cleaned)
        else:
            cleaned = self._patterns["code_block"].sub("", cleaned)

        # 2. 链接（保留链接文本）
        cleaned = self._patterns["link"].sub(r"\1", cleaned)

        # 3. 图片（保留 alt 文本）
        cleaned = self._patterns["image"].sub(r"\1", cleaned)

        # 4. 标题标记
        cleaned = self._patterns["heading"].sub("", cleaned)

        # 5. 粗体/斜体（保留文本）
        cleaned = self._patterns["emphasis"].sub(r"\2", cleaned)

        # 6. 行内代码（保留代码）
        cleaned = self._patterns["inline_code"].sub(r"\1", cleaned)

        # 7. 引用标记
        cleaned = self._patterns["blockquote"].sub("", cleaned)

        # 8. 列表标记
        cleaned = self._patterns["list"].sub("", cleaned)

        # 9. 水平线
        cleaned = self._patterns["hr"].sub("", cleaned)

        # 10. 规范化空白
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)  # 多个空行合并为两个
        cleaned = re.sub(r" {2,}", " ", cleaned)  # 多个空格合并为一个

        cleaned_length = len(cleaned.strip())
        metadata = {
            "original_length": original_length,
            "cleaned_length": cleaned_length,
        }

        warning = None
        if original_length != cleaned_length:
            removed = original_length - cleaned_length
            warning = f"移除了 {removed} 个字符的 Markdown 标记"
            metadata["characters_removed"] = removed

        return SanitizeResult(
            content=cleaned.strip(),
            passed=True,
            warning=warning,
            metadata=metadata,
        )


# 🏭 生产提示：
# 1. 对于复杂 HTML（嵌套表格、SVG），考虑使用 html5lib 或 lxml 进行完整解析
# 2. 对于需要保留部分格式的场景（如保留列表结构），考虑添加白名单模式
# 3. 对于 Markdown 扩展语法（如 GFM 的表格、任务列表），需要额外的处理规则
# 4. 性能优化：对于大量文本，考虑使用 C 扩展的正则引擎（regex 库而非 re）
# 5. 添加 HTML/Markdown 检测器，自动识别输入格式并选择合适的剥离器
