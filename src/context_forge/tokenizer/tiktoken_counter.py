"""
基于 tiktoken 的 Token 计数器。

tiktoken 是 OpenAI 官方的 tokenizer 库，支持 GPT-4o、GPT-4、GPT-3.5 等模型
使用的 BPE 编码方案。对于 OpenAI 系列模型，它提供精确的 Token 计数。

对于非 OpenAI 模型（如 Claude、Gemini），tiktoken 的计数结果是近似值，
误差通常在 5% 以内。在大多数预算管理场景下，这个精度是够用的。

→ 6.2.1 物理约束与性能模型
"""

from __future__ import annotations

import logging

import tiktoken

logger = logging.getLogger(__name__)

# 每条消息的格式开销（OpenAI 的消息格式会引入额外的控制 Token）
# 参考：https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken
_MESSAGE_OVERHEAD = 4  # 每条消息额外 4 token（<|im_start|>role\n...content<|im_end|>\n）
_REPLY_OVERHEAD = 3    # 回复消息额外 3 token


class TiktokenCounter:
    """
    基于 tiktoken 的 Token 计数器。

    这是 Context Forge 的默认计数器。对 OpenAI 模型精确，
    对其他模型提供合理的近似值。

    用法::

        counter = TiktokenCounter()  # 默认使用 cl100k_base
        count = counter.count("Hello, world!")

        # 指定编码方案
        counter = TiktokenCounter(encoding_name="o200k_base")  # GPT-4o

    属性:
        encoding_name: tiktoken 编码方案名称
    """

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        """
        初始化 TiktokenCounter。

        参数:
            encoding_name: tiktoken 编码方案名称。
                常用值：
                - "cl100k_base"：GPT-4、GPT-3.5-turbo
                - "o200k_base"：GPT-4o、GPT-4o-mini
        """
        self._encoding_name = encoding_name
        try:
            self._encoding = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            logger.warning(
                "tiktoken 编码方案 '%s' 加载失败，回退到 cl100k_base。错误：%s",
                encoding_name,
                e,
            )
            self._encoding_name = "cl100k_base"
            self._encoding = tiktoken.get_encoding("cl100k_base")

    def count(self, text: str) -> int:
        """
        计算文本的 Token 数量。

        参数:
            text: 待计数的文本

        返回:
            Token 数量
        """
        if not text:
            return 0
        return len(self._encoding.encode(text))

    def count_messages(self, messages: list[dict[str, str]]) -> int:
        """
        计算消息列表的 Token 总数（含消息格式开销）。

        # [Design Decision] 消息格式开销是 OpenAI 特有的。
        # 对于 Claude 等模型，这个开销估算可能不完全准确，
        # 但误差很小（每条消息仅差几个 Token），不影响预算分配。

        参数:
            messages: 消息列表

        返回:
            Token 总数
        """
        total = 0
        for message in messages:
            total += _MESSAGE_OVERHEAD
            for key, value in message.items():
                total += self.count(value)
                if key == "name":
                    total += -1  # name 字段会减少 1 token
        total += _REPLY_OVERHEAD
        return total

    @property
    def name(self) -> str:
        """Tokenizer 名称标识。"""
        return f"tiktoken:{self._encoding_name}"

    def encode(self, text: str) -> list[int]:
        """
        将文本编码为 Token ID 列表。

        这个方法不是 TokenCounter 协议要求的，
        但在需要精确截断（按 Token 而非按字符）时很有用。

        参数:
            text: 待编码的文本

        返回:
            Token ID 列表
        """
        return self._encoding.encode(text)

    def decode(self, tokens: list[int]) -> str:
        """
        将 Token ID 列表解码为文本。

        参数:
            tokens: Token ID 列表

        返回:
            解码后的文本
        """
        return self._encoding.decode(tokens)

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        将文本截断到指定的 Token 数量。

        # [DX Decision] 提供基于 Token 的精确截断，
        # 比基于字符数的截断更准确——避免在 Token 边界处截断导致乱码。

        参数:
            text: 待截断的文本
            max_tokens: 最大 Token 数

        返回:
            截断后的文本
        """
        if max_tokens <= 0:
            return ""
        tokens = self._encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self._encoding.decode(tokens[:max_tokens])
