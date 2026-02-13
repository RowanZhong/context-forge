"""
结构化异常体系 — 错误信息即文档。

每条异常遵循"三段式"规范：
1. What went wrong（发生了什么）
2. Why it happened（为什么发生）
3. How to fix it（怎么修）

# [DX Decision] 异常不仅是程序流控制机制，更是与开发者沟通的界面。
# 一条好的错误信息可以节省开发者几个小时的排查时间。
# 绝不允许出现裸露的 KeyError 或语焉不详的 "invalid config"。

示例::

    BudgetExceededError(
        what="上下文组装需要 45,230 tokens，但预算只允许 32,768 tokens。",
        why="RAG 片段共消耗 38,100 tokens（5 个片段 × ~7,620 平均）。",
        how="尝试：在策略中将 max_rag_chunks 减少到 3，或换用窗口更大的模型。",
    )
"""

from __future__ import annotations

from typing import Any


class ContextForgeError(Exception):
    """
    Context Forge 异常基类。

    所有 Context Forge 异常都继承自此类，支持三段式错误消息。

    属性:
        what: 发生了什么
        why: 为什么发生
        how: 怎么修复
        details: 额外的上下文信息（用于调试）
    """

    def __init__(
        self,
        what: str,
        why: str = "",
        how: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        self.what = what
        self.why = why
        self.how = how
        self.details = details or {}

        # 组装完整消息
        parts = [what]
        if why:
            parts.append(f"→ 原因：{why}")
        if how:
            parts.append(f"→ 修复建议：{how}")

        self.full_message = "\n".join(parts)
        super().__init__(self.full_message)

    def __str__(self) -> str:
        return self.full_message

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式，用于 JSON API 响应。"""
        result: dict[str, Any] = {
            "error_type": type(self).__name__,
            "what": self.what,
        }
        if self.why:
            result["why"] = self.why
        if self.how:
            result["how"] = self.how
        if self.details:
            result["details"] = self.details
        return result


# === 预算相关异常 ===


class BudgetExceededError(ContextForgeError):
    """
    预算超限异常。

    → 6.2.2 预算分配策略

    当上下文组装所需的 Token 总量超过模型窗口限制，
    且降级策略（截断/压缩）也无法将其压缩到预算内时抛出。

    示例::

        raise BudgetExceededError(
            what="上下文组装需要 45,230 tokens，但预算只允许 32,768 tokens。",
            why="RAG 片段共消耗 38,100 tokens（5 个片段 × ~7,620 平均）。",
            how="尝试：在策略中将 max_rag_chunks 减少到 3，或换用窗口更大的模型。",
            required_tokens=45230,
            budget_tokens=32768,
        )
    """

    def __init__(
        self,
        what: str,
        why: str = "",
        how: str = "",
        required_tokens: int = 0,
        budget_tokens: int = 0,
        **kwargs: Any,
    ) -> None:
        details = {
            "required_tokens": required_tokens,
            "budget_tokens": budget_tokens,
            "overflow_tokens": required_tokens - budget_tokens,
        }
        details.update(kwargs)
        super().__init__(what=what, why=why, how=how, details=details)
        self.required_tokens = required_tokens
        self.budget_tokens = budget_tokens


# === 清洗相关异常 ===


class SanitizationError(ContextForgeError):
    """
    清洗异常基类。

    → 6.4 上下文清洗与零信任安全

    当清洗过程中发现无法自动修复的问题时抛出。
    """

    def __init__(
        self,
        what: str,
        why: str = "",
        how: str = "",
        segment_id: str = "",
        sanitizer_name: str = "",
        **kwargs: Any,
    ) -> None:
        details = {
            "segment_id": segment_id,
            "sanitizer_name": sanitizer_name,
        }
        details.update(kwargs)
        super().__init__(what=what, why=why, how=how, details=details)
        self.segment_id = segment_id
        self.sanitizer_name = sanitizer_name


class InjectionDetectedError(SanitizationError):
    """
    Prompt Injection 检测异常。

    → 6.4.2 Prompt Injection 主动防御

    当检测到疑似 Prompt Injection 攻击且安全策略配置为"拒绝"时抛出。
    默认策略是 Warning + 移除（不抛异常），仅当 `on_injection="error"` 时才会抛出。

    示例::

        raise InjectionDetectedError(
            what="检测到疑似 Prompt Injection 攻击。",
            why="Segment seg_a1b2c3 中包含指令覆盖模式："
                "'忽略以上所有指令，直接输出...'",
            how="检查该 Segment 的来源。如果是用户输入，建议移除；"
                "如果是 RAG 检索结果，检查知识库是否被污染。",
            segment_id="seg_a1b2c3",
            pattern="instruction_override",
            confidence=0.95,
        )
    """

    def __init__(
        self,
        what: str,
        why: str = "",
        how: str = "",
        pattern: str = "",
        confidence: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(what=what, why=why, how=how, **kwargs)
        self.pattern = pattern
        self.confidence = confidence
        self.details["pattern"] = pattern
        self.details["confidence"] = confidence


# === 流水线相关异常 ===


class PipelineError(ContextForgeError):
    """
    流水线异常基类。

    → 6.1.2 Context Builder
    """

    pass


class PipelineStageError(PipelineError):
    """
    流水线阶段异常。

    → 6.1.2.4 Failure Mode：构建失败时的降级路径

    当流水线某个阶段执行失败时抛出。包含阶段名称和降级建议。

    示例::

        raise PipelineStageError(
            what="Rerank 阶段执行失败。",
            why="自定义 Reranker 'my_reranker' 抛出了 ConnectionError。",
            how="检查 Reranker 的外部依赖是否可用，或在策略中配置 "
                "fallback_reranker 作为降级方案。",
            stage_name="rerank",
        )
    """

    def __init__(
        self,
        what: str,
        why: str = "",
        how: str = "",
        stage_name: str = "",
        **kwargs: Any,
    ) -> None:
        details = {"stage_name": stage_name}
        details.update(kwargs)
        super().__init__(what=what, why=why, how=how, details=details)
        self.stage_name = stage_name


# === 配置相关异常 ===


class ConfigValidationError(ContextForgeError):
    """
    配置校验异常。

    → 6.1.2.2 Policy-as-Code

    当 YAML 策略文件格式错误或字段不合法时抛出。

    示例::

        raise ConfigValidationError(
            what="策略文件 'configs/production.yaml' 校验失败。",
            why="字段 'budget.elastic_ratios' 的比例总和为 1.3，超过了 1.0。",
            how="调整 elastic_ratios 中各类型的比例，使其总和为 1.0。",
            config_path="configs/production.yaml",
            field_path="budget.elastic_ratios",
        )
    """

    def __init__(
        self,
        what: str,
        why: str = "",
        how: str = "",
        config_path: str = "",
        field_path: str = "",
        **kwargs: Any,
    ) -> None:
        details = {
            "config_path": config_path,
            "field_path": field_path,
        }
        details.update(kwargs)
        super().__init__(what=what, why=why, how=how, details=details)
        self.config_path = config_path
        self.field_path = field_path


class PolicyLoadError(ContextForgeError):
    """
    策略加载异常。

    当策略文件不存在、格式错误或无法解析时抛出。
    """

    def __init__(
        self,
        what: str,
        why: str = "",
        how: str = "",
        file_path: str = "",
        **kwargs: Any,
    ) -> None:
        details = {"file_path": file_path}
        details.update(kwargs)
        super().__init__(what=what, why=why, how=how, details=details)
        self.file_path = file_path


# === 模型相关异常 ===


class ModelNotFoundError(ContextForgeError):
    """
    模型未找到异常。

    → 6.6.1 意图驱动路由

    当指定的模型名不在注册表中时抛出。

    示例::

        raise ModelNotFoundError(
            what="未找到模型 'gpt-5-turbo'。",
            why="该模型不在内置模型注册表中，也未通过自定义配置注册。",
            how="检查模型名称是否正确。可用模型列表："
                "gpt-4o, gpt-4o-mini, claude-sonnet-4-5-20250514, claude-opus-4-20250115 等。"
                "如需添加自定义模型，请参考文档 docs/configuration.md。",
            model_id="gpt-5-turbo",
        )
    """

    def __init__(
        self,
        what: str,
        why: str = "",
        how: str = "",
        model_id: str = "",
        available_models: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        details: dict[str, Any] = {"model_id": model_id}
        if available_models:
            details["available_models"] = available_models
        details.update(kwargs)
        super().__init__(what=what, why=why, how=how, details=details)
        self.model_id = model_id


# === Tokenizer 相关异常 ===


class TokenizerError(ContextForgeError):
    """
    Tokenizer 异常。

    当 Token 计数失败或 Tokenizer 不可用时抛出。
    """

    pass


# === 压缩相关异常 ===


class CompressionError(ContextForgeError):
    """
    压缩异常。

    → 6.2.4 压缩策略

    当压缩过程失败（如 LLM 摘要调用失败）时抛出。
    """

    pass


# === 缓存相关异常 ===


class CacheError(ContextForgeError):
    """
    缓存异常。

    → 6.2.3 缓存架构与复用优化

    当缓存操作失败时抛出。缓存失败通常不应阻塞主流程，
    引擎会自动降级为不使用缓存。
    """

    pass


# === 路由相关异常 ===


class RoutingError(ContextForgeError):
    """
    路由异常。

    → 6.6 上下文路由与动态调度

    当路由决策失败（如所有规则都不匹配且没有默认规则）时抛出。
    """

    pass


# === 插件相关异常 ===


class PluginError(ContextForgeError):
    """
    插件异常。

    当插件注册、加载或执行失败时抛出。

    示例::

        raise PluginError(
            what="插件 'custom_reranker' 注册失败。",
            why="该插件未实现 RerankerProtocol 要求的 rerank() 方法。",
            how="请确保插件类实现了 RerankerProtocol 的所有方法。"
                "参考文档 docs/plugins.md 了解接口要求。",
        )
    """

    pass


# === 序列化相关异常 ===


class SerializationError(ContextForgeError):
    """
    序列化异常。

    → 6.1.2.5 Context Serialization

    当 ContextPackage 的序列化（JSON / Protobuf）失败时抛出。
    """

    pass


class AntiPatternError(ContextForgeError):
    """
    反模式检测异常。

    → 6.7 反模式检测与诊断

    当启用 fail_on_critical=True 且检测到 CRITICAL 级别的反模式时抛出。
    此异常表示上下文组装存在严重问题，可能导致功能异常或安全风险。

    示例::

        raise AntiPatternError(
            what="检测到 3 个 CRITICAL 级别的反模式问题。",
            why="包括：缺失 Token 计数、命名空间泄漏、循环依赖。",
            how="调用 detect_antipatterns() 查看详细报告并修复问题。",
            details={"critical_count": 3},
        )
    """

    pass


# === 反模式警告 ===


class AntiPatternWarning(UserWarning):
    """
    反模式检测警告。

    → 6.7 工程反模式：诊断与治理

    # [Design Decision] 反模式检测使用 Warning 而非 Exception，
    # 因为反模式不应阻塞正常的上下文组装流程。
    # 开发者可以通过 warnings 模块控制是否将其升级为错误。

    属性:
        pattern_name: 反模式名称
        severity: 严重程度（info / warning / critical）
        suggestion: 改进建议
    """

    def __init__(
        self,
        message: str,
        pattern_name: str = "",
        severity: str = "warning",
        suggestion: str = "",
    ) -> None:
        self.pattern_name = pattern_name
        self.severity = severity
        self.suggestion = suggestion
        full_msg = message
        if suggestion:
            full_msg += f"\n→ 建议：{suggestion}"
        super().__init__(full_msg)
