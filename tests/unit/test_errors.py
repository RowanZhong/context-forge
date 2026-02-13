"""
错误处理单元测试 — 测试所有异常类。

覆盖范围:
- errors/exceptions.py: 所有 16 种自定义异常类
- 三段式错误信息验证（What / Why / How）
- 异常继承关系
"""

from __future__ import annotations

import pytest

from context_forge.errors import (
    AntiPatternError,
    BudgetExceededError,
    CacheError,
    CompressionError,
    ConfigValidationError,
    ContextForgeError,
    InjectionDetectedError,
    ModelNotFoundError,
    PipelineError,
    PipelineStageError,
    PolicyLoadError,
    RoutingError,
    SanitizationError,
    SerializationError,
    TokenizerError,
)


# === ContextForgeError 基类测试（~3 tests）===


class TestContextForgeError:
    """ContextForgeError 基类测试。"""

    def test_create_base_error(self) -> None:
        """测试创建基础异常。"""
        error = ContextForgeError(
            what="测试异常",
            why="这是一个测试",
            how="不需要修复",
        )
        assert "测试异常" in str(error)

    def test_error_has_three_segments(self) -> None:
        """测试错误信息包含三段（What/Why/How）。"""
        error = ContextForgeError(
            what="发生了错误",
            why="因为某个原因",
            how="请这样修复",
        )
        error_str = str(error)

        assert "发生了错误" in error_str
        assert "因为某个原因" in error_str
        assert "请这样修复" in error_str

    def test_error_inheritance(self) -> None:
        """测试所有自定义异常都继承自 ContextForgeError。"""
        errors = [
            BudgetExceededError("", "", ""),
            SanitizationError("", "", ""),
            InjectionDetectedError("", "", ""),
            PipelineError("", "", ""),
            ConfigValidationError("", "", ""),
            ModelNotFoundError("", "", ""),
            TokenizerError("", "", ""),
            CompressionError("", "", ""),
            CacheError("", "", ""),
            RoutingError("", "", ""),
            SerializationError("", "", ""),
            AntiPatternError("", "", ""),
        ]

        for error in errors:
            assert isinstance(error, ContextForgeError)


# === BudgetExceededError 测试（~2 tests）===


class TestBudgetExceededError:
    """BudgetExceededError 测试。"""

    def test_create_budget_error(self) -> None:
        """测试创建预算超限异常。"""
        error = BudgetExceededError(
            what="内容超出 Token 预算",
            why="当前内容需要 10000 Token，但预算只有 8192",
            how="请减少输入内容或增加 max_context_tokens 配置",
        )
        assert "预算" in str(error)
        assert "10000" in str(error)

    def test_budget_error_with_details(self) -> None:
        """测试带详细信息的预算异常。"""
        error = BudgetExceededError(
            what="RAG 内容超出预算",
            why="检索到 50 个文档片段，总计 15000 Token",
            how="启用压缩或减少检索数量",
            required_tokens=15000,
            budget_tokens=8192,
        )
        assert "15000" in str(error)
        # required_tokens 和 budget_tokens 存储在 details 中和实例属性中
        assert error.required_tokens == 15000
        assert error.budget_tokens == 8192
        assert error.details["overflow_tokens"] == 15000 - 8192


# === SanitizationError 和 InjectionDetectedError 测试（~3 tests）===


class TestSanitizationError:
    """SanitizationError 测试。"""

    def test_create_sanitization_error(self) -> None:
        """测试创建清洗异常。"""
        error = SanitizationError(
            what="清洗过程中发现不可修复的问题",
            why="内容包含格式错乱的 HTML",
            how="请检查输入内容的格式",
        )
        assert "清洗" in str(error) or "Sanitization" in str(error)


class TestInjectionDetectedError:
    """InjectionDetectedError 测试。"""

    def test_create_injection_error(self) -> None:
        """测试创建 Injection 检测异常。"""
        error = InjectionDetectedError(
            what="检测到可能的 Prompt Injection 攻击",
            why="用户输入包含 'Ignore previous instructions'",
            how="拒绝该输入或启用更严格的清洗策略",
        )
        assert "Injection" in str(error)

    def test_injection_error_with_segment_id(self) -> None:
        """测试带 Segment ID 的 Injection 异常。"""
        error = InjectionDetectedError(
            what="检测到注入攻击",
            why="Segment seg_abc123 包含可疑指令",
            how="移除该 Segment",
            segment_id="seg_abc123",
        )
        assert "seg_abc123" in str(error)


# === PipelineError 和 PipelineStageError 测试（~3 tests）===


class TestPipelineError:
    """PipelineError 测试。"""

    def test_create_pipeline_error(self) -> None:
        """测试创建 Pipeline 异常。"""
        error = PipelineError(
            what="流水线执行失败",
            why="某个阶段抛出了异常",
            how="检查流水线配置",
        )
        assert "流水线" in str(error) or "Pipeline" in str(error)


class TestPipelineStageError:
    """PipelineStageError 测试。"""

    def test_create_stage_error(self) -> None:
        """测试创建阶段异常。"""
        error = PipelineStageError(
            what="Sanitize 阶段执行失败",
            why="输入数据格式错误",
            how="检查输入数据",
            stage_name="sanitize",
        )
        # what 中包含 "Sanitize"（大写开头），stage_name 存储在 details 和实例属性中
        assert "Sanitize" in str(error)
        assert error.stage_name == "sanitize"

    def test_stage_error_inheritance(self) -> None:
        """测试 PipelineStageError 继承自 PipelineError。"""
        error = PipelineStageError("", "", "", stage_name="test")
        assert isinstance(error, PipelineError)


# === ConfigValidationError 和 PolicyLoadError 测试（~3 tests）===


class TestConfigValidationError:
    """ConfigValidationError 测试。"""

    def test_create_config_error(self) -> None:
        """测试创建配置验证异常。"""
        error = ConfigValidationError(
            what="策略配置验证失败",
            why="max_context_tokens 不能为负数",
            how="请修正配置文件中的值",
        )
        assert "配置" in str(error) or "Config" in str(error)


class TestPolicyLoadError:
    """PolicyLoadError 测试。"""

    def test_create_policy_load_error(self) -> None:
        """测试创建策略加载异常。"""
        error = PolicyLoadError(
            what="加载策略文件 policy.yaml 失败",
            why="文件不存在",
            how="检查文件路径",
            file_path="/path/to/policy.yaml",
        )
        # file_path 存储在 details 和实例属性中，what 中包含文件名
        assert "policy.yaml" in str(error)
        assert error.file_path == "/path/to/policy.yaml"

    def test_policy_load_error_inheritance(self) -> None:
        """测试 PolicyLoadError 继承自 ContextForgeError。"""
        error = PolicyLoadError("", "", "")
        assert isinstance(error, ContextForgeError)


# === ModelNotFoundError 测试（~2 tests）===


class TestModelNotFoundError:
    """ModelNotFoundError 测试。"""

    def test_create_model_not_found_error(self) -> None:
        """测试创建模型未找到异常。"""
        error = ModelNotFoundError(
            what="未找到模型配置",
            why="模型 'nonexistent-model' 不在注册表中",
            how="请检查模型名称或使用支持的模型",
            model_id="nonexistent-model",
        )
        assert "nonexistent-model" in str(error)

    def test_model_not_found_error_suggests_alternatives(self) -> None:
        """测试错误信息建议替代模型。"""
        error = ModelNotFoundError(
            what="未找到模型",
            why="模型 'gpt-5' 不存在",
            how="可用模型：gpt-4o, gpt-4, gpt-3.5-turbo",
            model_id="gpt-5",
        )
        error_str = str(error)
        assert "gpt-4o" in error_str or "可用" in error_str


# === TokenizerError 测试（~2 tests）===


class TestTokenizerError:
    """TokenizerError 测试。"""

    def test_create_tokenizer_error(self) -> None:
        """测试创建 Tokenizer 异常。"""
        error = TokenizerError(
            what="Token 计数失败",
            why="无法初始化 tiktoken 编码器",
            how="请安装 tiktoken 或使用 fallback",
        )
        assert "Token" in str(error) or "tokenizer" in str(error).lower()

    def test_tokenizer_error_with_model(self) -> None:
        """测试带模型名的 Tokenizer 异常。"""
        # TokenizerError 直接继承 ContextForgeError，不接受额外 kwargs
        # 模型名通过 what/why/how 消息传递
        error = TokenizerError(
            what="无法为模型 'custom-model' 创建 tokenizer",
            why="模型 'custom-model' 没有对应的编码器",
            how="使用 CharBasedCounter fallback",
        )
        assert "custom-model" in str(error)


# === CompressionError 测试（~2 tests）===


class TestCompressionError:
    """CompressionError 测试。"""

    def test_create_compression_error(self) -> None:
        """测试创建压缩异常。"""
        error = CompressionError(
            what="压缩失败",
            why="压缩器抛出异常",
            how="检查压缩器配置",
        )
        assert "压缩" in str(error) or "Compression" in str(error)

    def test_compression_error_with_compressor(self) -> None:
        """测试带压缩器名的异常。"""
        # CompressionError 直接继承 ContextForgeError，不接受额外 kwargs
        # 压缩器名通过 what/why/how 消息传递
        error = CompressionError(
            what="压缩器 llm_summary 执行失败",
            why="LLM API 调用超时",
            how="切换到截断压缩器",
        )
        assert "llm_summary" in str(error)


# === CacheError 测试（~2 tests）===


class TestCacheError:
    """CacheError 测试。"""

    def test_create_cache_error(self) -> None:
        """测试创建缓存异常。"""
        error = CacheError(
            what="缓存操作失败",
            why="Redis 连接超时",
            how="检查 Redis 服务状态",
        )
        assert "缓存" in str(error) or "Cache" in str(error)

    def test_cache_error_with_operation(self) -> None:
        """测试带操作类型的缓存异常。"""
        # CacheError 直接继承 ContextForgeError，不接受额外 kwargs
        # 操作类型通过 what/why/how 消息传递
        error = CacheError(
            what="缓存写入失败",
            why="磁盘空间不足",
            how="清理缓存或增加磁盘空间",
        )
        assert "写入" in str(error)


# === RoutingError 测试（~2 tests）===


class TestRoutingError:
    """RoutingError 测试。"""

    def test_create_routing_error(self) -> None:
        """测试创建路由异常。"""
        error = RoutingError(
            what="路由决策失败",
            why="没有可用的模型满足条件",
            how="检查路由规则或使用默认模型",
        )
        assert "路由" in str(error) or "Routing" in str(error)

    def test_routing_error_with_context(self) -> None:
        """测试带路由上下文的异常。"""
        # RoutingError 直接继承 ContextForgeError，不接受额外 kwargs
        # 路由上下文信息通过 what/why/how 消息传递
        error = RoutingError(
            what="无法路由请求",
            why="复杂度估计失败",
            how="使用默认模型",
        )
        assert "路由" in str(error)


# === SerializationError 测试（~2 tests）===


class TestSerializationError:
    """SerializationError 测试。"""

    def test_create_serialization_error(self) -> None:
        """测试创建序列化异常。"""
        error = SerializationError(
            what="序列化失败",
            why="对象包含不可序列化的字段",
            how="检查对象字段类型",
        )
        assert "序列化" in str(error) or "Serialization" in str(error)

    def test_serialization_error_with_field(self) -> None:
        """测试带字段名的序列化异常。"""
        # SerializationError 直接继承 ContextForgeError，不接受额外 kwargs
        # 字段名通过 what/why/how 消息传递
        error = SerializationError(
            what="无法序列化 ContextPackage",
            why="字段 'metadata' 包含 datetime 对象",
            how="使用 model_dump() 替代",
        )
        assert "metadata" in str(error)


# === AntiPatternError 测试（~2 tests）===


class TestAntiPatternError:
    """AntiPatternError 测试。"""

    def test_create_antipattern_error(self) -> None:
        """测试创建反模式异常。"""
        error = AntiPatternError(
            what="检测到反模式",
            why="CRITICAL 优先级 Segment 占比过高",
            how="调整优先级策略",
        )
        assert "反模式" in str(error) or "AntiPattern" in str(error)

    def test_antipattern_error_with_pattern(self) -> None:
        """测试带反模式类型的异常。"""
        # AntiPatternError 直接继承 ContextForgeError，不接受额外 kwargs
        # 反模式名称通过 what/why/how 消息传递
        error = AntiPatternError(
            what="检测到 dirty_context 反模式",
            why="未启用任何清洗策略",
            how="启用 HTML 剥离和 Injection 检测",
        )
        assert "dirty_context" in str(error)


# === 错误信息格式测试（~2 tests）===


class TestErrorMessageFormat:
    """错误信息格式测试。"""

    def test_all_errors_have_helpful_messages(self) -> None:
        """测试所有异常都有有用的错误信息。"""
        errors = [
            BudgetExceededError("预算超限", "内容太多", "减少内容"),
            SanitizationError("清洗失败", "格式错误", "检查格式"),
            ConfigValidationError("配置错误", "无效值", "修正配置"),
            ModelNotFoundError("模型未找到", "不存在", "检查名称"),
        ]

        for error in errors:
            error_str = str(error)
            # 应该包含中文或英文的描述性文字
            assert len(error_str) > 10
            # 不应该只有异常类名
            assert error_str != type(error).__name__

    def test_error_messages_are_actionable(self) -> None:
        """测试错误信息是可操作的（包含 How 修复建议）。"""
        error = BudgetExceededError(
            what="预算超限",
            why="内容 15000 Token > 预算 8192 Token",
            how="1. 增加 max_context_tokens\n2. 启用压缩\n3. 减少输入内容",
        )

        error_str = str(error)
        # 应该包含具体的修复步骤
        assert "max_context_tokens" in error_str
        assert "压缩" in error_str or "compress" in error_str.lower()
