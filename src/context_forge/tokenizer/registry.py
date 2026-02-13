"""
Tokenizer 注册表 — 根据模型名自动选择最佳 Tokenizer。

# [DX Decision] 用户只需要传入模型名（如 "gpt-4o"），
# 引擎自动选择正确的 Tokenizer。这是"零配置识别"的核心组件——
# 开发者不需要知道 GPT-4o 用的是 o200k_base 编码方案。

→ 6.2.1 物理约束与性能模型
"""

from __future__ import annotations

import logging

from context_forge.tokenizer.fallback import CharBasedCounter
from context_forge.tokenizer.protocol import TokenCounter
from context_forge.tokenizer.tiktoken_counter import TiktokenCounter

logger = logging.getLogger(__name__)

# 模型名前缀到 tiktoken 编码方案的映射
# [Design Decision] 使用前缀匹配而非精确匹配，
# 因为模型名经常包含日期后缀（如 claude-sonnet-4-5-20250514），
# 前缀匹配更具鲁棒性。
_MODEL_TO_ENCODING: dict[str, str] = {
    # OpenAI — GPT-4o 系列使用 o200k_base
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "chatgpt-4o": "o200k_base",
    # OpenAI — GPT-4 / GPT-3.5 使用 cl100k_base
    "gpt-4-turbo": "cl100k_base",
    "gpt-4-": "cl100k_base",  # gpt-4-0125-preview 等
    "gpt-4": "cl100k_base",
    "gpt-3.5": "cl100k_base",
    # OpenAI — Reasoning Models
    "o1": "o200k_base",
    "o3": "o200k_base",
    "o4-mini": "o200k_base",
    # Anthropic — 使用 cl100k_base 作为近似
    # 🏭 生产提示：Anthropic 没有公开的 Python tokenizer，
    # 使用 cl100k_base 近似计数，误差通常在 5% 以内。
    # 如需精确计数，可调用 Anthropic 的 count_tokens API。
    "claude": "cl100k_base",
    # Google — 使用 cl100k_base 作为近似
    # 🏭 生产提示：Gemini 使用 SentencePiece tokenizer，
    # 如需精确计数，使用 google-generativeai 的 count_tokens() 方法。
    "gemini": "cl100k_base",
    # 本地模型 — 使用 cl100k_base 作为近似
    "llama": "cl100k_base",
    "mistral": "cl100k_base",
    "qwen": "cl100k_base",
    "deepseek": "cl100k_base",
}

# Tokenizer 实例缓存（避免重复创建）
_counter_cache: dict[str, TokenCounter] = {}

# 用户注册的自定义 Tokenizer
_custom_counters: dict[str, TokenCounter] = {}


def get_tokenizer(model: str) -> TokenCounter:
    """
    根据模型名获取最佳 Token 计数器。

    查找优先级：
    1. 用户注册的自定义 Tokenizer
    2. 基于模型名前缀匹配的 tiktoken 编码方案
    3. CharBasedCounter fallback

    参数:
        model: 模型名称（如 "gpt-4o"、"claude-sonnet-4-5-20250514"）

    返回:
        TokenCounter 实例

    示例::

        counter = get_tokenizer("gpt-4o")
        tokens = counter.count("Hello, world!")
    """
    # 1. 检查自定义注册
    if model in _custom_counters:
        return _custom_counters[model]

    # 2. 检查缓存
    if model in _counter_cache:
        return _counter_cache[model]

    # 3. 前缀匹配
    encoding_name = _find_encoding(model)

    if encoding_name:
        try:
            counter: TokenCounter = TiktokenCounter(encoding_name)
            _counter_cache[model] = counter
            return counter
        except Exception as e:
            logger.warning(
                "为模型 '%s' 创建 tiktoken 计数器失败（编码：%s），"
                "回退到字符计数器。错误：%s",
                model,
                encoding_name,
                e,
            )

    # 4. Fallback
    logger.info(
        "模型 '%s' 未找到专用 Tokenizer，使用字符计数器（近似值）。",
        model,
    )
    counter = CharBasedCounter()
    _counter_cache[model] = counter
    return counter


def _find_encoding(model: str) -> str | None:
    """通过前缀匹配找到编码方案。"""
    model_lower = model.lower()

    # 按前缀长度降序排列，优先匹配更具体的前缀
    for prefix in sorted(_MODEL_TO_ENCODING.keys(), key=len, reverse=True):
        if model_lower.startswith(prefix):
            return _MODEL_TO_ENCODING[prefix]

    return None


def register_tokenizer(model: str, counter: TokenCounter) -> None:
    """
    注册自定义 Token 计数器。

    注册后，该模型将优先使用自定义计数器而非内置的 tiktoken。
    适用于需要精确计数的场景（如使用 Anthropic API 的 count_tokens）。

    参数:
        model: 模型名称
        counter: TokenCounter 实例

    示例::

        class AnthropicCounter:
            def count(self, text: str) -> int:
                # 调用 Anthropic API 的 count_tokens
                return anthropic_client.count_tokens(text)

            def count_messages(self, messages: list[dict[str, str]]) -> int:
                return sum(self.count(m["content"]) for m in messages)

            @property
            def name(self) -> str:
                return "anthropic_api"

        register_tokenizer("claude-sonnet-4-5-20250514", AnthropicCounter())
    """
    if not isinstance(counter, TokenCounter):
        raise TypeError(
            f"counter 必须实现 TokenCounter 协议，"
            f"但 {type(counter).__name__} 缺少必要的方法。"
            f"需要实现：count(text) -> int, count_messages(messages) -> int, name -> str"
        )
    _custom_counters[model] = counter
    logger.info("已为模型 '%s' 注册自定义 Tokenizer: %s", model, counter.name)


def clear_cache() -> None:
    """清除 Tokenizer 缓存。通常仅在测试中使用。"""
    _counter_cache.clear()
