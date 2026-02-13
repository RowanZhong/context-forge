"""
TokenCounter 协议定义。

Token 计数是上下文工程的基础能力——不知道每段内容有多少 Token，
就无法做预算分配。但不同模型使用不同的 Tokenizer，
同一段文本在 GPT-4o 和 Claude 中的 Token 数量可能完全不同。

# [Design Decision] 使用 Protocol（结构化子类型）而非 ABC（名义子类型），
# 让任何实现了 count() 方法的对象都可以作为 TokenCounter 使用，
# 无需显式继承。这降低了自定义 Tokenizer 的接入门槛。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class TokenCounter(Protocol):
    """
    Token 计数器协议。

    所有 Token 计数器必须实现此协议。Context Forge 内置了三种实现：
    - TiktokenCounter：基于 tiktoken，适用于 OpenAI 模型（精确）
    - CharBasedCounter：基于字符数除以 4 的粗估（零依赖 fallback）
    - 用户可以通过实现此协议注入自定义 Tokenizer

    最小实现示例::

        class MyTokenizer:
            def count(self, text: str) -> int:
                return len(my_custom_tokenize(text))

            def count_messages(self, messages: list[dict[str, str]]) -> int:
                return sum(self.count(m["content"]) for m in messages)

            @property
            def name(self) -> str:
                return "my_tokenizer"
    """

    def count(self, text: str) -> int:
        """
        计算文本的 Token 数量。

        参数:
            text: 待计数的文本

        返回:
            Token 数量
        """
        ...

    def count_messages(self, messages: list[dict[str, str]]) -> int:
        """
        计算消息列表的 Token 总数。

        # [Design Decision] 单独提供消息计数方法，因为不同模型的消息格式
        # 会引入额外的控制 Token（如 <|im_start|>、<|im_end|>），
        # 简单地逐条拼接文本计数会低估实际消耗。

        参数:
            messages: 消息列表，每条消息为 {"role": "...", "content": "..."} 格式

        返回:
            Token 总数（含消息格式开销）
        """
        ...

    @property
    def name(self) -> str:
        """Tokenizer 名称标识。"""
        ...
