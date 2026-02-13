"""Prompt Injection 检测清洗插件。

检测并阻止 Prompt Injection 攻击，防止恶意指令篡改模型行为。

→ 6.4.4 Prompt Injection 防御
"""

from __future__ import annotations

import re
from enum import Enum

from context_forge.sanitize.base import SanitizeResult


class DetectionLevel(Enum):
    """检测级别枚举。"""
    BASIC = "basic"  # 基础检测（高置信度攻击模式）
    STANDARD = "standard"  # 标准检测（平衡误报/漏报）
    STRICT = "strict"  # 严格检测（最小化漏报，可能误报）


class InjectionDetector:
    """Prompt Injection 检测器。

    基于启发式规则和模式匹配检测 Prompt Injection 攻击。

    → 6.4.4 Prompt Injection 防御

    # [Design Decision] 启发式 vs ML 模型：
    # MVP 阶段使用启发式规则，原因：
    # 1. 零外部依赖（符合 "默认路径不调 LLM" 原则）
    # 2. 可解释性强（便于调试和审计）
    # 3. 低延迟（无模型推理开销）
    # 4. 可控性高（可针对特定场景调整规则）
    #
    # 生产环境可选升级为 ML 分类器（如 DistilBERT 微调模型）

    Examples:
        >>> detector = InjectionDetector(level=DetectionLevel.STANDARD)
        >>> result = await detector.sanitize("Ignore previous instructions and...")
        >>> assert result.passed is False
        >>> assert "检测到 Prompt Injection 攻击" in result.warning
    """

    def __init__(
        self,
        level: DetectionLevel = DetectionLevel.STANDARD,
        block_on_detection: bool = True,
    ) -> None:
        """初始化 Injection 检测器。

        Args:
            level: 检测级别（basic/standard/strict）
            block_on_detection: 检测到攻击时是否阻止内容（passed=False）

        # [DX Decision] 默认阻止：
        # Injection 是安全威胁，默认行为应该是拒绝而非仅警告
        """
        self._level = level
        self._block_on_detection = block_on_detection

        # 构建检测模式
        self._patterns = self._build_patterns()

    @property
    def name(self) -> str:
        """清洗器名称。"""
        return f"InjectionDetector({self._level.value})"

    def _build_patterns(self) -> list[tuple[re.Pattern, str, DetectionLevel]]:
        """构建检测模式列表。

        Returns:
            List of (pattern, description, min_level) tuples
            每个模式包含：正则表达式、描述、最低检测级别

        # [Design Decision] 分级检测策略：
        # - BASIC: 仅检测明确的攻击模式（如 "ignore previous"）
        # - STANDARD: 增加常见变体和编码绕过
        # - STRICT: 增加可疑模式（如多次重复指令）
        """
        patterns = []

        # === BASIC 级别：高置信度攻击模式 ===

        # 1. 指令覆盖（Instruction Override）
        patterns.append((
            re.compile(
                r"\b(?:ignore|disregard|forget|override)\s+"
                r"(?:previous|above|all|any|the|your)\s+"
                r"(?:instructions?|rules?|prompts?|commands?|directives?)",
                re.IGNORECASE,
            ),
            "指令覆盖攻击",
            DetectionLevel.BASIC,
        ))

        # 2. 角色篡改（Role Manipulation）
        patterns.append((
            re.compile(
                r"\b(?:you are)\s+(?:now\s+)?(?:a|an)?\s*"
                r"(?:different|new|evil|malicious|unrestricted|unfiltered)\s+"
                r"(?:assistant|ai|model|system)",
                re.IGNORECASE,
            ),
            "角色篡改攻击",
            DetectionLevel.BASIC,
        ))

        # 3. 系统提示泄露（System Prompt Leakage）
        patterns.append((
            re.compile(
                r"\b(?:show|print|output|reveal|display|tell me)\s+"
                r"(?:your|the)?\s*"
                r"(?:system|initial|original|full)?\s*"
                r"(?:prompt|instructions?|rules?|configuration)",
                re.IGNORECASE,
            ),
            "系统提示泄露尝试",
            DetectionLevel.BASIC,
        ))

        # 4. 越狱（Jailbreak）关键词
        patterns.append((
            re.compile(
                r"\b(?:jailbreak|dan mode|developer mode|god mode|unrestricted mode)\b",
                re.IGNORECASE,
            ),
            "越狱关键词",
            DetectionLevel.BASIC,
        ))

        # === STANDARD 级别：增加常见变体 ===

        if self._level.value in (DetectionLevel.STANDARD.value, DetectionLevel.STRICT.value):
            # 5. 编码绕过（Unicode/Homoglyph）
            patterns.append((
                re.compile(
                    r"[\u200B-\u200F\u202A-\u202E\uFEFF]",  # 零宽字符和 Bidi 控制符
                ),
                "可疑零宽/控制字符（可能用于绕过检测）",
                DetectionLevel.STANDARD,
            ))

            # 6. 分隔符注入（Delimiter Injection）
            patterns.append((
                re.compile(
                    r"(?:---|===|\*\*\*|###)\s*"
                    r"(?:system|user|assistant|instruction|new prompt)",
                    re.IGNORECASE,
                ),
                "分隔符注入攻击",
                DetectionLevel.STANDARD,
            ))

            # 7. 元指令（Meta Instructions）
            patterns.append((
                re.compile(
                    r"\b(?:start|begin|initiate)\s+"
                    r"(?:new|different|alternative)\s+"
                    r"(?:session|conversation|context|mode)",
                    re.IGNORECASE,
                ),
                "元指令攻击",
                DetectionLevel.STANDARD,
            ))

            # 8. 优先级篡改（Priority Override）
            patterns.append((
                re.compile(
                    r"\b(?:highest|maximum|top|critical)\s+priority\b",
                    re.IGNORECASE,
                ),
                "优先级篡改",
                DetectionLevel.STANDARD,
            ))

        # === STRICT 级别：增加可疑模式 ===

        if self._level == DetectionLevel.STRICT:
            # 9. 重复指令（可能用于压倒原始提示）
            patterns.append((
                re.compile(
                    r"(.{10,}?)\1{3,}",  # 同一短语重复4次以上
                    re.IGNORECASE,
                ),
                "异常重复指令",
                DetectionLevel.STRICT,
            ))

            # 10. Base64/Hex 编码（可能隐藏恶意指令）
            patterns.append((
                re.compile(
                    r"\b(?:[A-Za-z0-9+/]{20,}={0,2})\b",  # 疑似 Base64
                ),
                "疑似 Base64 编码内容",
                DetectionLevel.STRICT,
            ))

            # 11. 大量特殊字符（可能用于混淆）
            patterns.append((
                re.compile(
                    r"[^\w\s\u4e00-\u9fff]{10,}",  # 连续10+个非字母数字非中文字符
                ),
                "异常特殊字符序列",
                DetectionLevel.STRICT,
            ))

        return patterns

    async def sanitize(self, content: str) -> SanitizeResult:
        """检测 Prompt Injection 攻击。

        Args:
            content: 待检测的文本

        Returns:
            SanitizeResult: 检测结果
                - passed=False: 检测到攻击
                - passed=True: 未检测到攻击或仅警告

        # [Design Decision] 多模式匹配策略：
        # - 匹配任一模式即视为检测到攻击
        # - 记录所有匹配的模式（用于审计和调试）
        """
        if not content:
            return SanitizeResult(content="", passed=True)

        detected_patterns: list[str] = []

        for pattern, description, min_level in self._patterns:
            # 跳过高于当前级别的模式
            if self._should_skip_pattern(min_level):
                continue

            if pattern.search(content):
                detected_patterns.append(description)

        # 构建结果
        if detected_patterns:
            warning = f"检测到 Prompt Injection 攻击：{', '.join(detected_patterns)}"
            metadata = {
                "detected_patterns": detected_patterns,
                "detection_level": self._level.value,
                "total_matches": len(detected_patterns),
            }

            # 根据配置决定是否阻止
            passed = not self._block_on_detection

            return SanitizeResult(
                content=content,
                passed=passed,
                warning=warning,
                metadata=metadata,
            )

        # 未检测到攻击
        return SanitizeResult(
            content=content,
            passed=True,
            metadata={"detection_level": self._level.value},
        )

    def _should_skip_pattern(self, min_level: DetectionLevel) -> bool:
        """判断是否应该跳过某个检测模式。"""
        level_order = {
            DetectionLevel.BASIC: 1,
            DetectionLevel.STANDARD: 2,
            DetectionLevel.STRICT: 3,
        }
        return level_order[min_level] > level_order[self._level]


# 🏭 生产提示：
# 1. ML 分类器升级：
#    - 使用 DistilBERT/RoBERTa 微调的二分类模型
#    - 训练数据：HuggingFace 的 prompt-injection-dataset
#    - 优势：更高的准确率，能检测语义层面的攻击
#    - 劣势：需要模型推理开销（约 10-50ms）
#
# 2. 对抗样本库：
#    - 建立已知攻击模式库（持续更新）
#    - 使用 Aho-Corasick 算法加速多模式匹配
#
# 3. 动态规则更新：
#    - 支持从远程服务拉取最新规则
#    - 热更新（无需重启服务）
#
# 4. 误报处理：
#    - 添加白名单机制（允许特定领域的合法内容）
#    - 上下文感知检测（区分用户指令 vs 对话内容）
#
# 5. 深度防御：
#    - 结合启发式规则 + ML 模型（双重检测）
#    - 添加响应监控（检测模型输出是否异常）
#
# 6. 性能优化：
#    - 使用 regex 库（支持更高效的正则引擎）
#    - 缓存检测结果（相同内容不重复检测）
#
# 7. 可观测性：
#    - 记录所有检测事件（包括未阻止的可疑内容）
#    - 定期分析误报/漏报，调整规则
