"""清洗层基础协议与编排器。

本模块定义了清洗插件的统一接口和责任链编排器。

→ 6.4.1 零信任清洗管道架构
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from context_forge.errors import SanitizationError


# [Design Decision] 使用 tuple 而非 list 保证不可变性
@dataclass(frozen=True)
class SanitizeResult:
    """清洗结果数据类。

    Attributes:
        content: 清洗后的文本内容
        passed: 是否通过清洗检查（False 表示检测到威胁需要拒绝）
        warning: 警告信息（可选）
        metadata: 附加元数据（如替换计数、检测到的模式等）
    """
    content: str
    passed: bool = True
    warning: str | None = None
    metadata: dict[str, Any] | None = None


# [Design Decision] 使用 Protocol 而非 ABC 提供鸭子类型的灵活性
@runtime_checkable
class Sanitizer(Protocol):
    """清洗器协议接口。

    所有清洗插件必须实现此协议。遵循单一职责原则，每个插件只做一种清洗。

    → 6.4.2 清洗插件协议设计
    """

    @property
    def name(self) -> str:
        """清洗器名称，用于审计日志。"""
        ...

    async def sanitize(self, content: str) -> SanitizeResult:
        """执行清洗操作。

        Args:
            content: 待清洗的文本内容

        Returns:
            SanitizeResult: 清洗结果，包含处理后内容和检查状态

        Raises:
            SanitizationError: 清洗过程中发生不可恢复的错误
        """
        ...


class SanitizerChain:
    """清洗器责任链编排器。

    按顺序执行多个清洗插件，支持短路机制（检测到威胁时立即停止）。

    → 6.4.1 零信任清洗管道架构

    Examples:
        >>> chain = SanitizerChain([normalizer, html_stripper, pii_redactor])
        >>> result = await chain.process("用户输入内容")
        >>> if not result.passed:
        ...     raise SanitizationError(result.warning)
    """

    def __init__(self, sanitizers: list[Sanitizer]) -> None:
        """初始化清洗链。

        Args:
            sanitizers: 清洗插件列表，按执行顺序排列

        # [Design Decision] 推荐顺序：
        # 1. Unicode 归一化（预处理）
        # 2. HTML/Markdown 剥离（结构清理）
        # 3. PII 脱敏（隐私保护）
        # 4. Injection 检测（安全检查）
        # 5. 长度防御（资源保护）
        """
        if not sanitizers:
            warnings.warn(
                "创建了空的 SanitizerChain，所有内容将不经处理直接通过",
                UserWarning,
                stacklevel=2,
            )
        self._sanitizers = tuple(sanitizers)  # 不可变

    async def process(self, content: str) -> SanitizeResult:
        """执行完整清洗流程。

        Args:
            content: 待清洗的原始内容

        Returns:
            SanitizeResult: 最终清洗结果

        Raises:
            SanitizationError: 任一清洗器抛出不可恢复错误

        # [Design Decision] 短路机制：
        # 当任一清洗器返回 passed=False 时立即停止，不再执行后续清洗器。
        # 这样可以避免浪费计算资源处理已确认的恶意内容。
        """
        current_content = content
        all_warnings: list[str] = []
        all_metadata: dict[str, Any] = {}

        for sanitizer in self._sanitizers:
            try:
                result = await sanitizer.sanitize(current_content)
            except Exception as e:
                # 将插件异常包装为 SanitizationError
                raise SanitizationError(
                    f"清洗器 '{sanitizer.name}' 执行失败: {e}",
                    details={
                        "sanitizer": sanitizer.name,
                        "original_error": str(e),
                        "content_length": len(current_content),
                    }
                ) from e

            # 累积元数据
            if result.metadata:
                all_metadata[sanitizer.name] = result.metadata

            # 累积警告
            if result.warning:
                all_warnings.append(f"[{sanitizer.name}] {result.warning}")

            # [Design Decision] 短路：检测到威胁立即停止
            if not result.passed:
                return SanitizeResult(
                    content=result.content,
                    passed=False,
                    warning=result.warning,
                    metadata=all_metadata,
                )

            # 继续下一个清洗器
            current_content = result.content

        # 所有清洗器都通过
        combined_warning = "; ".join(all_warnings) if all_warnings else None
        return SanitizeResult(
            content=current_content,
            passed=True,
            warning=combined_warning,
            metadata=all_metadata if all_metadata else None,
        )

    @property
    def sanitizers(self) -> tuple[Sanitizer, ...]:
        """获取清洗器列表（只读）。"""
        return self._sanitizers

    def __len__(self) -> int:
        """返回清洗器数量。"""
        return len(self._sanitizers)

    def __repr__(self) -> str:
        names = [s.name for s in self._sanitizers]
        return f"SanitizerChain({names})"


# 🏭 生产提示：
# 1. 考虑添加 ConcurrentSanitizerChain 支持独立清洗器的并行执行（如 PII 脱敏和长度检查可以并行）
# 2. 添加清洗器缓存机制（如 LRU Cache）避免重复处理相同内容
# 3. 支持清洗器热插拔（动态加载/卸载插件）
# 4. 添加清洗器性能监控（执行时间、通过率统计）
# 5. 支持条件清洗（根据 SegmentType 或 Provenance 选择性应用清洗器）
