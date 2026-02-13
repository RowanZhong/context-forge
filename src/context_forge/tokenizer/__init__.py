"""
Context Forge Token 计数模块。

提供可插拔的 Token 计数能力，根据模型名自动选择最佳 Tokenizer。
"""

from context_forge.tokenizer.fallback import CharBasedCounter
from context_forge.tokenizer.protocol import TokenCounter
from context_forge.tokenizer.registry import (
    clear_cache,
    get_tokenizer,
    register_tokenizer,
)
from context_forge.tokenizer.tiktoken_counter import TiktokenCounter

__all__ = [
    "CharBasedCounter",
    "TiktokenCounter",
    "TokenCounter",
    "clear_cache",
    "get_tokenizer",
    "register_tokenizer",
]
