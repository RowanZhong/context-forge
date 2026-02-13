"""
Tokenizer 模块单元测试 — 测试 Token 计数器。

覆盖范围:
- tokenizer/protocol.py: TokenCounter Protocol
- tokenizer/tiktoken_counter.py: TiktokenCounter
- tokenizer/fallback.py: CharBasedCounter
- tokenizer/registry.py: get_tokenizer()
"""

from __future__ import annotations

import pytest

from context_forge.tokenizer.fallback import CharBasedCounter
from context_forge.tokenizer.protocol import TokenCounter
from context_forge.tokenizer.registry import get_tokenizer
from context_forge.tokenizer.tiktoken_counter import TiktokenCounter


# === TiktokenCounter 测试（~6 tests）===


class TestTiktokenCounter:
    """TiktokenCounter 测试（精确计数）。"""

    def test_create_tiktoken_counter(self) -> None:
        """测试创建 Tiktoken 计数器（默认 cl100k_base 编码）。"""
        counter = TiktokenCounter()
        assert isinstance(counter, TokenCounter)

    def test_create_tiktoken_counter_with_encoding(self) -> None:
        """测试指定编码方案创建 Tiktoken 计数器。"""
        counter = TiktokenCounter(encoding_name="o200k_base")
        assert isinstance(counter, TokenCounter)
        assert counter.name == "tiktoken:o200k_base"

    def test_tiktoken_count_english(self) -> None:
        """测试计数英文文本。"""
        counter = TiktokenCounter(encoding_name="o200k_base")
        text = "Hello, world! This is a test."
        count = counter.count(text)
        assert count > 0
        assert count < 20  # 应该在合理范围内

    def test_tiktoken_count_chinese(self) -> None:
        """测试计数中文文本。"""
        counter = TiktokenCounter(encoding_name="o200k_base")
        text = "你好，世界！这是一个测试。"
        count = counter.count(text)
        assert count > 0

    def test_tiktoken_count_empty(self) -> None:
        """测试计数空字符串。"""
        counter = TiktokenCounter()
        assert counter.count("") == 0

    def test_tiktoken_count_mixed_language(self) -> None:
        """测试计数中英文混合文本。"""
        counter = TiktokenCounter(encoding_name="o200k_base")
        text = "Hello 你好 World 世界"
        count = counter.count(text)
        assert count > 0

    def test_tiktoken_different_encodings(self) -> None:
        """测试不同编码方案的计数可能不同。"""
        counter_o200k = TiktokenCounter(encoding_name="o200k_base")
        counter_cl100k = TiktokenCounter(encoding_name="cl100k_base")

        text = "This is a test sentence."

        count1 = counter_o200k.count(text)
        count2 = counter_cl100k.count(text)

        # 可能相同也可能不同，但都应该大于 0
        assert count1 > 0
        assert count2 > 0

    def test_tiktoken_name_property(self) -> None:
        """测试 name 属性格式。"""
        counter = TiktokenCounter()
        assert counter.name == "tiktoken:cl100k_base"

    def test_tiktoken_count_messages(self) -> None:
        """测试消息列表 Token 计数（含格式开销）。"""
        counter = TiktokenCounter()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        count = counter.count_messages(messages)
        # 应该包含消息内容 + 格式开销
        assert count > counter.count("Hello") + counter.count("Hi there!")

    def test_tiktoken_encode_decode(self) -> None:
        """测试 encode/decode 辅助方法。"""
        counter = TiktokenCounter()
        text = "Hello, world!"
        tokens = counter.encode(text)
        assert len(tokens) == counter.count(text)
        decoded = counter.decode(tokens)
        assert decoded == text

    def test_tiktoken_truncate_to_tokens(self) -> None:
        """测试按 Token 精确截断。"""
        counter = TiktokenCounter()
        text = "This is a longer test sentence with many words."
        full_count = counter.count(text)
        max_tokens = 3
        truncated = counter.truncate_to_tokens(text, max_tokens)
        assert counter.count(truncated) <= max_tokens
        assert counter.count(truncated) > 0

    def test_tiktoken_truncate_empty(self) -> None:
        """测试截断到 0 个 Token。"""
        counter = TiktokenCounter()
        assert counter.truncate_to_tokens("Hello", 0) == ""

    def test_tiktoken_invalid_encoding_fallback(self) -> None:
        """测试无效编码方案时回退到 cl100k_base。"""
        counter = TiktokenCounter(encoding_name="nonexistent_encoding")
        # 应该回退到 cl100k_base，仍然能正常计数
        assert counter.count("Hello") > 0
        assert counter.name == "tiktoken:cl100k_base"


# === CharBasedCounter 测试（~5 tests）===


class TestCharBasedCounter:
    """CharBasedCounter 测试（粗估 fallback）。"""

    def test_create_char_based_counter(self) -> None:
        """测试创建字符计数器。"""
        counter = CharBasedCounter()
        assert isinstance(counter, TokenCounter)

    def test_char_based_count_english(self) -> None:
        """测试计数英文（字符数 / 4）。"""
        counter = CharBasedCounter()
        text = "Hello world"  # 11 个字符（不含空格为 10）
        count = counter.count(text)

        # 英文按 chars/4 估算
        expected = len(text) // 4
        assert count == expected or count == expected + 1

    def test_char_based_count_chinese(self) -> None:
        """测试计数中文（字符数 / 2）。"""
        counter = CharBasedCounter()
        text = "你好世界"  # 4 个中文字符
        count = counter.count(text)

        # 中文按 chars/2 估算
        expected = len(text) // 2
        assert count == expected

    def test_char_based_count_mixed(self) -> None:
        """测试计数中英文混合。"""
        counter = CharBasedCounter()
        text = "Hello 你好 123"
        count = counter.count(text)

        # 应该有合理的估算值
        assert count > 0
        assert count < len(text)

    def test_char_based_count_empty(self) -> None:
        """测试计数空字符串。"""
        counter = CharBasedCounter()
        assert counter.count("") == 0

    # === 新增测试：fixed_ratio 模式 ===

    def test_fixed_ratio_english(self) -> None:
        """测试固定比率模式（英文）。"""
        counter = CharBasedCounter(chars_per_token=4.0)
        text = "Hello world test"  # 16 个字符
        count = counter.count(text)
        assert count == 4  # 16 / 4 = 4

    def test_fixed_ratio_chinese(self) -> None:
        """测试固定比率模式（中文）。"""
        counter = CharBasedCounter(chars_per_token=2.0)
        text = "你好世界测试文本"  # 8 个字符
        count = counter.count(text)
        assert count == 4  # 8 / 2 = 4

    def test_fixed_ratio_custom(self) -> None:
        """测试自定义固定比率。"""
        counter = CharBasedCounter(chars_per_token=3.0)
        text = "123456789"  # 9 个字符
        count = counter.count(text)
        assert count == 3  # 9 / 3 = 3

    # === 新增测试：中文检测边界条件 ===

    def test_cjk_threshold_pure_english(self) -> None:
        """测试纯英文（CJK 比率 0%）。"""
        counter = CharBasedCounter()
        text = "This is a test sentence with only English characters."
        count = counter.count(text)
        # 纯英文：ratio = 4.0
        expected = len(text) / 4.0
        assert abs(count - expected) <= 1

    def test_cjk_threshold_pure_chinese(self) -> None:
        """测试纯中文（CJK 比率 100%）。"""
        counter = CharBasedCounter()
        text = "这是一个完全由中文字符组成的测试句子"
        count = counter.count(text)
        # 纯中文：ratio = 4.0 - (1.0 * 2.5) = 1.5
        expected = len(text) / 1.5
        assert abs(count - expected) <= 1

    def test_cjk_threshold_30_percent(self) -> None:
        """测试 CJK 比率接近 30% 临界值。"""
        counter = CharBasedCounter()
        # 构造约 30% 中文的文本
        text = "Hello你好World世界Test测试"  # 6 中文 + 14 英文 = 30% CJK
        count = counter.count(text)
        assert count > 0
        # 30% CJK: ratio = 4.0 - (0.3 * 2.5) = 3.25
        expected = len(text) / 3.25
        assert abs(count - expected) <= 2

    def test_cjk_threshold_50_percent(self) -> None:
        """测试 CJK 比率 50% 混合文本。"""
        counter = CharBasedCounter()
        text = "Hello你好World世界Test测试Text文本"  # 8 中文 + 8 英文
        count = counter.count(text)
        # 50% CJK: ratio = 4.0 - (0.5 * 2.5) = 2.75
        expected = len(text) / 2.75
        assert abs(count - expected) <= 2

    # === 新增测试：特殊字符处理 ===

    def test_count_emoji(self) -> None:
        """测试包含 Emoji 的文本。"""
        counter = CharBasedCounter()
        text = "Hello 😀 World 🌍 Test 🚀"
        count = counter.count(text)
        assert count > 0
        # Emoji 按英文字符处理
        assert count < len(text)

    def test_count_symbols(self) -> None:
        """测试包含特殊符号的文本。"""
        counter = CharBasedCounter()
        text = "!@#$%^&*()_+-=[]{}|;:',.<>?/~`"
        count = counter.count(text)
        assert count > 0
        # 符号按英文字符处理（chars / 4）
        expected = len(text) / 4.0
        assert abs(count - expected) <= 1

    def test_count_unicode_combining_characters(self) -> None:
        """测试 Unicode 组合字符。"""
        counter = CharBasedCounter()
        # é = e + 组合重音符号
        text = "café"  # 可能是 4 或 5 个 code points
        count = counter.count(text)
        assert count >= 1  # 至少应该有 1 个 token

    def test_count_japanese_hiragana(self) -> None:
        """测试日文平假名（属于 CJK 范围）。"""
        counter = CharBasedCounter()
        text = "こんにちは世界"  # 平假名 + 汉字
        count = counter.count(text)
        # 日文平假名在 CJK 范围内，按中文处理
        # 实际上平假名不在 _CJK_PATTERN 中，所以按英文处理
        assert count > 0
        assert count <= len(text)

    def test_count_korean_hangul(self) -> None:
        """测试韩文（属于 CJK 范围）。"""
        counter = CharBasedCounter()
        text = "안녕하세요"  # 韩文
        count = counter.count(text)
        # 韩文在 CJK 范围内
        assert count > 0

    # === 新增测试：边界条件 ===

    def test_count_single_character(self) -> None:
        """测试单个字符（确保 max(1, ...) 生效）。"""
        counter = CharBasedCounter()
        assert counter.count("a") == 1
        assert counter.count("中") == 1

    def test_count_very_long_text(self) -> None:
        """测试超长文本（> 100K 字符）。"""
        counter = CharBasedCounter()
        text = "a" * 100_000  # 100K 英文字符
        count = counter.count(text)
        # 100K / 4 = 25K tokens
        assert count == 25_000

    def test_count_whitespace_only(self) -> None:
        """测试纯空白字符。"""
        counter = CharBasedCounter()
        text = "   \t\n\r   "
        count = counter.count(text)
        # 空白字符按英文处理
        assert count >= 1

    # === 新增测试：count_messages() 方法 ===

    def test_count_messages_single_message(self) -> None:
        """测试单条消息的 Token 计数。"""
        counter = CharBasedCounter()
        messages = [{"role": "user", "content": "Hello"}]
        count = counter.count_messages(messages)
        # 4 (消息格式开销) + 所有字段值 + 3 (回复开销)
        expected = 4 + counter.count("user") + counter.count("Hello") + 3
        assert count == expected

    def test_count_messages_multiple_messages(self) -> None:
        """测试多条消息的 Token 计数。"""
        counter = CharBasedCounter()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        count = counter.count_messages(messages)
        # 每条消息 4 tokens 格式开销 + 所有字段值 + 3 tokens 回复开销
        expected = (
            4 + counter.count("user") + counter.count("Hello") +
            4 + counter.count("assistant") + counter.count("Hi there!") +
            4 + counter.count("user") + counter.count("How are you?") +
            3
        )
        assert count == expected

    def test_count_messages_empty_content(self) -> None:
        """测试空内容消息的 Token 计数。"""
        counter = CharBasedCounter()
        messages = [{"role": "user", "content": ""}]
        count = counter.count_messages(messages)
        # 4 (格式) + counter.count("user") + 0 (空内容) + 3 (回复)
        expected = 4 + counter.count("user") + 0 + 3
        assert count == expected

    def test_count_messages_chinese_content(self) -> None:
        """测试中文消息的 Token 计数。"""
        counter = CharBasedCounter()
        messages = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好世界"},
        ]
        count = counter.count_messages(messages)
        expected = (
            4 + counter.count("user") + counter.count("你好") +
            4 + counter.count("assistant") + counter.count("你好世界") +
            3
        )
        assert count == expected

    def test_count_messages_multiple_fields(self) -> None:
        """测试多字段消息的 Token 计数。"""
        counter = CharBasedCounter()
        messages = [
            {"role": "user", "content": "Hello", "name": "Alice"},
        ]
        count = counter.count_messages(messages)
        # 应该计数所有字段的值
        expected = 4 + counter.count("user") + counter.count("Hello") + counter.count("Alice") + 3
        assert count == expected

    # === 新增测试：name 属性 ===

    def test_name_property_auto_mode(self) -> None:
        """测试自动检测模式的 name 属性。"""
        counter = CharBasedCounter()
        assert counter.name == "char_based:auto"

    def test_name_property_fixed_ratio(self) -> None:
        """测试固定比率模式的 name 属性。"""
        counter = CharBasedCounter(chars_per_token=3.5)
        assert counter.name == "char_based:3.5"

    def test_name_property_integer_ratio(self) -> None:
        """测试整数比率的 name 属性。"""
        counter = CharBasedCounter(chars_per_token=2.0)
        assert counter.name == "char_based:2.0"

    # === 新增测试：内部方法 _estimate_ratio() ===

    def test_estimate_ratio_empty_string(self) -> None:
        """测试空字符串的比率估算（覆盖第 64 行）。"""
        counter = CharBasedCounter()
        ratio = counter._estimate_ratio("")
        assert ratio == 4.0  # 空文本默认返回 4.0

    def test_estimate_ratio_fixed_mode(self) -> None:
        """测试固定比率模式不受文本影响。"""
        counter = CharBasedCounter(chars_per_token=3.0)
        # 固定比率模式应该直接返回 _fixed_ratio
        assert counter._estimate_ratio("Hello") == 3.0
        assert counter._estimate_ratio("你好") == 3.0
        assert counter._estimate_ratio("") == 3.0

    # === 新增测试：极端边界条件 ===

    def test_count_zero_length_after_strip(self) -> None:
        """测试仅包含不可见字符的特殊情况（间接测试 total_chars == 0 分支）。"""
        counter = CharBasedCounter()
        # 虽然无法直接构造 len(text) != 0 但 total_chars == 0 的情况
        # 但我们可以验证空字符串的稳健性
        assert counter.count("") == 0
        # 以及测试其他边界情况的正确性
        assert counter.count("\u200b") >= 1  # 零宽空格


# === get_tokenizer() 测试（~4 tests）===


class TestGetTokenizer:
    """get_tokenizer() 工厂函数测试。"""

    def test_get_tokenizer_for_gpt(self) -> None:
        """测试为 GPT 模型获取 tokenizer。"""
        counter = get_tokenizer("gpt-4o")
        assert isinstance(counter, TiktokenCounter)

    def test_get_tokenizer_for_claude(self) -> None:
        """测试为 Claude 模型获取 tokenizer。"""
        counter = get_tokenizer("claude-sonnet-4-5-20250514")
        # Claude 也使用 tiktoken（cl100k_base）
        assert isinstance(counter, (TiktokenCounter, CharBasedCounter))

    def test_get_tokenizer_fallback(self) -> None:
        """测试未知模型使用 fallback。"""
        counter = get_tokenizer("unknown-model")
        # 应该返回 CharBasedCounter 作为 fallback
        assert isinstance(counter, CharBasedCounter)

    def test_get_tokenizer_consistent_counts(self) -> None:
        """测试同一模型的计数器结果一致。"""
        counter1 = get_tokenizer("gpt-4o")
        counter2 = get_tokenizer("gpt-4o")

        text = "This is a test."
        assert counter1.count(text) == counter2.count(text)


# === TokenCounter Protocol 测试（~2 tests）===


class TestTokenCounterProtocol:
    """TokenCounter Protocol 测试。"""

    def test_protocol_compliance(self) -> None:
        """测试 TiktokenCounter 符合 Protocol。"""
        counter = TiktokenCounter()
        assert isinstance(counter, TokenCounter)

    def test_char_based_protocol_compliance(self) -> None:
        """测试 CharBasedCounter 符合 Protocol。"""
        counter = CharBasedCounter()
        assert isinstance(counter, TokenCounter)

    def test_custom_counter_implementation(self) -> None:
        """测试自定义 TokenCounter 实现。"""

        class CustomCounter:
            """自定义计数器（简单实现）。"""

            def count(self, text: str) -> int:
                return len(text.split())

            def count_messages(self, messages: list[dict[str, str]]) -> int:
                return sum(self.count(m.get("content", "")) for m in messages)

            @property
            def name(self) -> str:
                return "custom_word_counter"

        counter = CustomCounter()
        # 应该符合 Protocol
        assert isinstance(counter, TokenCounter)
        assert counter.count("hello world") == 2


# === 精度对比测试（~2 tests）===


class TestTokenizerAccuracy:
    """Tokenizer 精度对比测试。"""

    def test_tiktoken_vs_char_based_english(self) -> None:
        """测试英文文本的精度差异。"""
        text = "This is a test sentence with multiple words."

        tiktoken_counter = TiktokenCounter(encoding_name="o200k_base")
        char_counter = CharBasedCounter()

        tiktoken_count = tiktoken_counter.count(text)
        char_count = char_counter.count(text)

        # Tiktoken 应该更精确，但差异应该在合理范围内（< 50%）
        diff_ratio = abs(tiktoken_count - char_count) / tiktoken_count
        assert diff_ratio < 0.5

    def test_tiktoken_vs_char_based_chinese(self) -> None:
        """测试中文文本的精度差异。"""
        text = "这是一个测试句子，包含多个中文字符。"

        tiktoken_counter = TiktokenCounter(encoding_name="o200k_base")
        char_counter = CharBasedCounter()

        tiktoken_count = tiktoken_counter.count(text)
        char_count = char_counter.count(text)

        # 中文的估算误差可能更大
        diff_ratio = abs(tiktoken_count - char_count) / tiktoken_count
        assert diff_ratio < 1.0  # 允许更大的误差
