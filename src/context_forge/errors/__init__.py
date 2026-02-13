"""
Context Forge 结构化异常体系。

每一条用户可见的错误都是产品界面的一部分。
所有异常遵循"三段式"规范：What / Why / How to fix。
"""

from context_forge.errors.exceptions import (
    AntiPatternError,
    AntiPatternWarning,
    BudgetExceededError,
    CacheError,
    CompressionError,
    ConfigValidationError,
    ContextForgeError,
    InjectionDetectedError,
    ModelNotFoundError,
    PipelineError,
    PipelineStageError,
    PluginError,
    PolicyLoadError,
    RoutingError,
    SanitizationError,
    SerializationError,
    TokenizerError,
)

__all__ = [
    "AntiPatternError",
    "AntiPatternWarning",
    "BudgetExceededError",
    "CacheError",
    "CompressionError",
    "ConfigValidationError",
    "ContextForgeError",
    "InjectionDetectedError",
    "ModelNotFoundError",
    "PipelineError",
    "PipelineStageError",
    "PluginError",
    "PolicyLoadError",
    "RoutingError",
    "SanitizationError",
    "SerializationError",
    "TokenizerError",
]
