"""长度攻击防御清洗插件。

防止通过超长输入耗尽资源或绕过 Token 限制。

→ 6.4.5 长度攻击防御
"""

from __future__ import annotations

from typing import Any

from context_forge.sanitize.base import SanitizeResult


class LengthGuard:
    """长度防御清洗器。

    限制输入文本的长度和复杂度，防止资源耗尽攻击（DoS）。

    → 6.4.5 长度攻击防御

    # [Design Decision] 多维度防御：
    # 1. 字符数限制：防止超长文本
    # 2. 行数限制：防止大量换行符攻击
    # 3. 单行长度限制：防止无换行超长行
    # 4. 重复度检测：防止通过重复内容绕过限制

    Examples:
        >>> guard = LengthGuard(max_chars=1000)
        >>> result = await guard.sanitize("a" * 2000)
        >>> assert result.passed is False  # 超长被拒绝

        >>> guard = LengthGuard(max_chars=1000, truncate_on_overflow=True)
        >>> result = await guard.sanitize("a" * 2000)
        >>> assert result.passed is True
        >>> assert len(result.content) == 1000  # 截断到限制
    """

    def __init__(
        self,
        max_chars: int = 100_000,
        max_lines: int = 10_000,
        max_line_length: int = 10_000,
        max_repetition_ratio: float = 0.8,
        truncate_on_overflow: bool = False,
    ) -> None:
        """初始化长度防御器。

        Args:
            max_chars: 最大字符数
            max_lines: 最大行数
            max_line_length: 单行最大长度
            max_repetition_ratio: 最大重复比例（0.0-1.0）
            truncate_on_overflow: 超长时是否截断（False=拒绝，True=截断）

        # [Design Decision] 默认限制值：
        # - max_chars=100K: 约 25K tokens（GPT-4 的 1/8），足够大多数场景
        # - max_lines=10K: 防止换行符炸弹（\\n * 1M）
        # - max_line_length=10K: 防止单行超长（影响正则性能）
        # - max_repetition_ratio=0.8: 允许一定重复（如日志），但拒绝明显攻击

        # [DX Decision] 默认拒绝而非截断：
        # 截断可能破坏语义，导致难以调试的问题。默认拒绝更安全。
        # 如需截断，显式设置 truncate_on_overflow=True
        """
        if max_chars <= 0 or max_lines <= 0 or max_line_length <= 0:
            raise ValueError("长度限制必须大于 0")
        if not 0.0 <= max_repetition_ratio <= 1.0:
            raise ValueError("重复比例必须在 0.0-1.0 之间")

        self._max_chars = max_chars
        self._max_lines = max_lines
        self._max_line_length = max_line_length
        self._max_repetition_ratio = max_repetition_ratio
        self._truncate_on_overflow = truncate_on_overflow

    @property
    def name(self) -> str:
        """清洗器名称。"""
        return "LengthGuard"

    async def sanitize(self, content: str) -> SanitizeResult:
        """检查文本长度和复杂度。

        Args:
            content: 待检查的文本

        Returns:
            SanitizeResult: 检查结果
                - passed=False: 超出限制且未设置截断
                - passed=True: 未超限制或已截断

        Raises:
            SanitizationError: 内部错误（不应发生）
        """
        if not content:
            return SanitizeResult(content="", passed=True)

        violations: list[str] = []
        metadata: dict[str, Any] = {}

        # 1. 检查总字符数
        char_count = len(content)
        metadata["char_count"] = char_count
        if char_count > self._max_chars:
            violations.append(
                f"字符数超限：{char_count} > {self._max_chars}"
            )

        # 2. 检查行数
        lines = content.split("\n")
        line_count = len(lines)
        metadata["line_count"] = line_count
        if line_count > self._max_lines:
            violations.append(
                f"行数超限：{line_count} > {self._max_lines}"
            )

        # 3. 检查单行长度
        max_line_len = max((len(line) for line in lines), default=0)
        metadata["max_line_length"] = max_line_len
        if max_line_len > self._max_line_length:
            violations.append(
                f"单行长度超限：{max_line_len} > {self._max_line_length}"
            )

        # 4. 检查重复度（仅对足够长的文本）
        if char_count >= 100:  # 短文本跳过重复检测
            repetition_ratio = self._calculate_repetition_ratio(content)
            metadata["repetition_ratio"] = round(repetition_ratio, 3)
            if repetition_ratio > self._max_repetition_ratio:
                violations.append(
                    f"重复度过高：{repetition_ratio:.1%} > {self._max_repetition_ratio:.1%}"
                )

        # 处理违规
        if violations:
            if self._truncate_on_overflow:
                # 截断模式：修复违规并通过
                truncated = self._truncate(content)
                warning = f"内容超限已截断：{'; '.join(violations)}"
                return SanitizeResult(
                    content=truncated,
                    passed=True,
                    warning=warning,
                    metadata=metadata,
                )
            else:
                # 拒绝模式：返回失败
                warning = f"长度攻击检测：{'; '.join(violations)}"
                return SanitizeResult(
                    content=content,
                    passed=False,
                    warning=warning,
                    metadata=metadata,
                )

        # 未超限
        return SanitizeResult(
            content=content,
            passed=True,
            metadata=metadata,
        )

    def _truncate(self, content: str) -> str:
        """截断内容到安全限制。

        # [Design Decision] 截断策略：
        # 1. 优先保留前面的内容（通常更重要）
        # 2. 按行截断（避免截断到单词中间）
        # 3. 遵守所有限制（字符数、行数、单行长度）
        """
        lines = content.split("\n")

        # 截断行数
        if len(lines) > self._max_lines:
            lines = lines[:self._max_lines]

        # 截断单行长度
        lines = [line[:self._max_line_length] for line in lines]

        # 拼接并截断总字符数
        truncated = "\n".join(lines)
        if len(truncated) > self._max_chars:
            truncated = truncated[:self._max_chars]

        return truncated

    @staticmethod
    def _calculate_repetition_ratio(content: str) -> float:
        """计算内容重复比例。

        使用滑动窗口检测重复片段。

        Returns:
            重复比例（0.0-1.0），值越大表示重复越多

        # [Design Decision] 简化的重复检测算法：
        # 使用固定窗口大小（50 字符）检测重复出现的片段。
        # 更精确的算法可以使用后缀数组或 Rabin-Karp，但开销更大。
        """
        if len(content) < 100:
            return 0.0

        window_size = 50
        if len(content) < window_size:
            window_size = len(content) // 2

        # 收集所有窗口
        windows: dict[str, int] = {}
        for i in range(len(content) - window_size + 1):
            window = content[i:i + window_size]
            windows[window] = windows.get(window, 0) + 1

        # 计算重复字符数
        total_repeated_chars = 0
        for window, count in windows.items():
            if count > 1:
                # (count - 1) 是重复次数
                total_repeated_chars += len(window) * (count - 1)

        return total_repeated_chars / len(content)


# 🏭 生产提示：
# 1. 动态限制：
#    - 根据用户等级/配额动态调整限制
#    - VIP 用户允许更大的输入
#
# 2. 智能截断：
#    - 在语义边界截断（句子、段落）而非硬截断
#    - 使用 NLP 工具（如 spaCy）识别句子边界
#
# 3. 压缩检测：
#    - 检测 gzip/bzip2 炸弹（压缩后小但解压后巨大）
#    - 限制解压后的大小
#
# 4. 嵌套结构检测：
#    - 检测过深的 JSON/XML 嵌套（递归炸弹）
#    - 限制嵌套深度
#
# 5. 正则性能保护：
#    - 检测可能导致正则引擎回溯爆炸的模式
#    - 使用线性时间正则引擎（如 re2）
#
# 6. 资源监控：
#    - 限制清洗过程的 CPU 时间和内存
#    - 使用 timeout 装饰器
#
# 7. 分段处理：
#    - 对超大文本分段处理（避免一次性加载到内存）
#    - 流式处理（Generator/AsyncGenerator）
#
# 8. 统计分析：
#    - 记录输入长度分布
#    - 识别异常模式（突然大量超长请求 = 攻击迹象）
