"""
默认配置与模型注册表。

# [DX Decision] 内置常见模型的完整配置信息，
# 让用户传入模型名即可自动匹配窗口大小、tokenizer、成本等，
# 无需手动查阅各厂商文档。这是"零配置识别"的数据基础。

→ 6.6.1 意图驱动路由
"""

from __future__ import annotations

from context_forge.models.routing import ModelConfig

# ============================================================
# 模型注册表
# 包含主流 LLM 的配置信息。数据来源为各厂商官方文档（截止 2025 年）。
#
# 🏭 生产提示：模型信息会随时间更新（新模型发布、价格调整）。
# 建议通过 YAML 策略文件覆盖默认值，或定期更新此注册表。
# ============================================================

MODEL_REGISTRY: dict[str, ModelConfig] = {
    # --- OpenAI ---
    "gpt-4o": ModelConfig(
        model_id="gpt-4o",
        provider="openai",
        max_context_tokens=128_000,
        max_output_tokens=16_384,
        tokenizer_name="o200k_base",
        supports_thinking=False,
        supports_tool_use=True,
        supports_vision=True,
        cost_per_million_input=2.50,
        cost_per_million_output=10.00,
    ),
    "gpt-4o-mini": ModelConfig(
        model_id="gpt-4o-mini",
        provider="openai",
        max_context_tokens=128_000,
        max_output_tokens=16_384,
        tokenizer_name="o200k_base",
        supports_thinking=False,
        supports_tool_use=True,
        supports_vision=True,
        cost_per_million_input=0.15,
        cost_per_million_output=0.60,
    ),
    "gpt-4-turbo": ModelConfig(
        model_id="gpt-4-turbo",
        provider="openai",
        max_context_tokens=128_000,
        max_output_tokens=4_096,
        tokenizer_name="cl100k_base",
        supports_thinking=False,
        supports_tool_use=True,
        supports_vision=True,
        cost_per_million_input=10.00,
        cost_per_million_output=30.00,
    ),
    "o1": ModelConfig(
        model_id="o1",
        provider="openai",
        max_context_tokens=200_000,
        max_output_tokens=100_000,
        tokenizer_name="o200k_base",
        supports_thinking=True,
        supports_tool_use=True,
        supports_vision=True,
        cost_per_million_input=15.00,
        cost_per_million_output=60.00,
    ),
    "o3-mini": ModelConfig(
        model_id="o3-mini",
        provider="openai",
        max_context_tokens=200_000,
        max_output_tokens=100_000,
        tokenizer_name="o200k_base",
        supports_thinking=True,
        supports_tool_use=True,
        supports_vision=False,
        cost_per_million_input=1.10,
        cost_per_million_output=4.40,
    ),
    # --- Anthropic ---
    "claude-opus-4-20250115": ModelConfig(
        model_id="claude-opus-4-20250115",
        provider="anthropic",
        max_context_tokens=200_000,
        max_output_tokens=32_000,
        tokenizer_name="cl100k_base",  # 近似
        supports_thinking=True,
        supports_tool_use=True,
        supports_vision=True,
        cost_per_million_input=15.00,
        cost_per_million_output=75.00,
    ),
    "claude-sonnet-4-5-20250514": ModelConfig(
        model_id="claude-sonnet-4-5-20250514",
        provider="anthropic",
        max_context_tokens=200_000,
        max_output_tokens=16_000,
        tokenizer_name="cl100k_base",  # 近似
        supports_thinking=True,
        supports_tool_use=True,
        supports_vision=True,
        cost_per_million_input=3.00,
        cost_per_million_output=15.00,
    ),
    "claude-haiku-3-5-20241022": ModelConfig(
        model_id="claude-haiku-3-5-20241022",
        provider="anthropic",
        max_context_tokens=200_000,
        max_output_tokens=8_192,
        tokenizer_name="cl100k_base",  # 近似
        supports_thinking=False,
        supports_tool_use=True,
        supports_vision=True,
        cost_per_million_input=0.80,
        cost_per_million_output=4.00,
    ),
    # --- Google ---
    "gemini-2.0-flash": ModelConfig(
        model_id="gemini-2.0-flash",
        provider="google",
        max_context_tokens=1_048_576,
        max_output_tokens=8_192,
        tokenizer_name="cl100k_base",  # 近似
        supports_thinking=True,
        supports_tool_use=True,
        supports_vision=True,
        cost_per_million_input=0.10,
        cost_per_million_output=0.40,
    ),
    "gemini-2.5-pro": ModelConfig(
        model_id="gemini-2.5-pro",
        provider="google",
        max_context_tokens=1_048_576,
        max_output_tokens=65_536,
        tokenizer_name="cl100k_base",  # 近似
        supports_thinking=True,
        supports_tool_use=True,
        supports_vision=True,
        cost_per_million_input=1.25,
        cost_per_million_output=10.00,
    ),
    # --- 本地/开源模型 ---
    "llama-3.1-70b": ModelConfig(
        model_id="llama-3.1-70b",
        provider="local",
        max_context_tokens=128_000,
        max_output_tokens=4_096,
        tokenizer_name="cl100k_base",  # 近似
        supports_thinking=False,
        supports_tool_use=True,
        supports_vision=False,
        cost_per_million_input=0.0,
        cost_per_million_output=0.0,
    ),
    "deepseek-v3": ModelConfig(
        model_id="deepseek-v3",
        provider="deepseek",
        max_context_tokens=128_000,
        max_output_tokens=8_192,
        tokenizer_name="cl100k_base",  # 近似
        supports_thinking=False,
        supports_tool_use=True,
        supports_vision=False,
        cost_per_million_input=0.27,
        cost_per_million_output=1.10,
    ),
    "qwen-2.5-72b": ModelConfig(
        model_id="qwen-2.5-72b",
        provider="local",
        max_context_tokens=131_072,
        max_output_tokens=8_192,
        tokenizer_name="cl100k_base",  # 近似
        supports_thinking=False,
        supports_tool_use=True,
        supports_vision=False,
        cost_per_million_input=0.0,
        cost_per_million_output=0.0,
    ),
}

# 常用模型别名映射
# [DX Decision] 让用户可以用简短别名而非完整模型 ID
MODEL_ALIASES: dict[str, str] = {
    "gpt4o": "gpt-4o",
    "gpt4o-mini": "gpt-4o-mini",
    "4o": "gpt-4o",
    "4o-mini": "gpt-4o-mini",
    "claude-opus": "claude-opus-4-20250115",
    "claude-sonnet": "claude-sonnet-4-5-20250514",
    "claude-haiku": "claude-haiku-3-5-20241022",
    "opus": "claude-opus-4-20250115",
    "sonnet": "claude-sonnet-4-5-20250514",
    "haiku": "claude-haiku-3-5-20241022",
    "gemini-flash": "gemini-2.0-flash",
    "gemini-pro": "gemini-2.5-pro",
    "llama": "llama-3.1-70b",
    "deepseek": "deepseek-v3",
    "qwen": "qwen-2.5-72b",
}


def resolve_model(model_id: str) -> ModelConfig:
    """
    解析模型名称并返回完整的模型配置。

    支持：
    1. 精确匹配模型 ID
    2. 别名匹配
    3. 前缀匹配（如 "claude-sonnet" 匹配 "claude-sonnet-4-5-20250514"）

    参数:
        model_id: 模型名称或别名

    返回:
        ModelConfig 实例

    异常:
        ModelNotFoundError: 未找到匹配的模型
    """
    from context_forge.errors import ModelNotFoundError

    # 1. 精确匹配
    if model_id in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_id]

    # 2. 别名匹配
    resolved = MODEL_ALIASES.get(model_id)
    if resolved and resolved in MODEL_REGISTRY:
        return MODEL_REGISTRY[resolved]

    # 3. 前缀匹配（从最长前缀开始）
    model_lower = model_id.lower()
    for registry_id in sorted(MODEL_REGISTRY.keys(), key=len, reverse=True):
        if model_lower.startswith(registry_id.lower()):
            return MODEL_REGISTRY[registry_id]
        if registry_id.lower().startswith(model_lower):
            return MODEL_REGISTRY[registry_id]

    available = list(MODEL_REGISTRY.keys()) + list(MODEL_ALIASES.keys())
    raise ModelNotFoundError(
        what=f"未找到模型 '{model_id}'。",
        why="该模型不在内置模型注册表中，也未通过自定义配置注册。",
        how=f"检查模型名称是否正确。可用的模型和别名：{', '.join(sorted(available)[:10])} 等。"
            f"如需添加自定义模型，使用 register_model() 方法。",
        model_id=model_id,
        available_models=sorted(available),
    )


def register_model(model_id: str, config: ModelConfig) -> None:
    """
    注册自定义模型配置。

    参数:
        model_id: 模型 ID
        config: 模型配置

    示例::

        register_model("my-local-model", ModelConfig(
            model_id="my-local-model",
            provider="local",
            max_context_tokens=32768,
            max_output_tokens=4096,
        ))
    """
    MODEL_REGISTRY[model_id] = config


def list_models() -> list[str]:
    """返回所有已注册模型的 ID 列表。"""
    return sorted(MODEL_REGISTRY.keys())
