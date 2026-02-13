"""
Context Forge 配置模块。

提供 YAML 策略加载、模型注册表和默认配置。
"""

from context_forge.config.defaults import (
    MODEL_ALIASES,
    MODEL_REGISTRY,
    list_models,
    register_model,
    resolve_model,
)
from context_forge.config.loader import load_policy, validate_policy_file
from context_forge.config.schema import PolicyConfig

__all__ = [
    "MODEL_ALIASES",
    "MODEL_REGISTRY",
    "PolicyConfig",
    "list_models",
    "load_policy",
    "register_model",
    "resolve_model",
    "validate_policy_file",
]
