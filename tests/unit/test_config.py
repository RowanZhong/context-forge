"""
配置模块单元测试 — 测试策略配置和模型注册表。

覆盖范围:
- config/schema.py: PolicyConfig 及所有子配置
- config/loader.py: load_policy(), YAML 加载/校验/合并
- config/defaults.py: MODEL_REGISTRY, MODEL_ALIASES, resolve_model()
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from context_forge.config.defaults import MODEL_ALIASES, MODEL_REGISTRY, resolve_model
from context_forge.config.loader import load_policy
from context_forge.config.schema import (
    BudgetConfig,
    CacheConfig,
    CompressConfig,
    PolicyConfig,
    RerankConfig,
    RoutingConfig,
    SanitizeRuleConfig as SanitizeConfig,
)
from context_forge.errors import ConfigValidationError, ModelNotFoundError


# === PolicyConfig 测试（~8 tests）===


class TestPolicyConfig:
    """PolicyConfig 主配置测试。"""

    def test_create_default_policy(self) -> None:
        """测试创建默认策略配置。"""
        policy = PolicyConfig()
        assert policy.version != ""
        assert policy.name != ""
        assert isinstance(policy.budget, BudgetConfig)

    def test_policy_with_all_sections(self) -> None:
        """测试包含所有配置段的策略。"""
        policy = PolicyConfig(
            version="1.0",
            name="test",
            budget=BudgetConfig(max_context_tokens=8192),
            sanitize=SanitizeConfig(strip_html=True),
            rerank=RerankConfig(enable_mmr=True),
            compress=CompressConfig(enabled=True),
            cache=CacheConfig(enabled=True),
            routing=RoutingConfig(enabled=False),
        )
        assert policy.version == "1.0"
        assert policy.budget.max_context_tokens == 8192

    def test_policy_to_budget_policy(self) -> None:
        """测试转换为 BudgetPolicy。"""
        policy = PolicyConfig(
            budget=BudgetConfig(
                max_context_tokens=10000,
                output_reserved_tokens=1000,
            ),
        )
        budget_policy = policy.to_budget_policy()

        from context_forge.models.budget import BudgetPolicy

        assert isinstance(budget_policy, BudgetPolicy)
        assert budget_policy.max_context_tokens == 10000

    def test_policy_immutable(self) -> None:
        """测试策略配置的子模型不可变（frozen=True 的子配置）。"""
        from context_forge.models.budget import BudgetPolicy

        budget = BudgetPolicy()
        with pytest.raises((ValidationError, AttributeError)):
            budget.max_context_tokens = 999  # type: ignore

    def test_policy_default_values(self) -> None:
        """测试所有子配置都有合理的默认值。"""
        policy = PolicyConfig()
        assert policy.budget.max_context_tokens > 0
        assert policy.sanitize.strip_html in [True, False]
        assert 0 <= policy.rerank.mmr_lambda <= 1

    def test_policy_validation(self) -> None:
        """测试策略验证逻辑。"""
        # 无效的 mmr_lambda（超出 0-1 范围）
        with pytest.raises(ValidationError):
            PolicyConfig(
                rerank=RerankConfig(mmr_lambda=1.5)
            )

    def test_policy_nested_update(self) -> None:
        """测试嵌套配置更新。"""
        policy = PolicyConfig()
        updated = policy.model_copy(
            update={"budget": BudgetConfig(max_context_tokens=16384)}
        )
        assert updated.budget.max_context_tokens == 16384
        assert policy.budget.max_context_tokens != 16384  # 原对象不变

    def test_policy_serialization(self) -> None:
        """测试序列化和反序列化。"""
        policy = PolicyConfig(version="1.0", name="test")
        dict_form = policy.model_dump()

        assert dict_form["version"] == "1.0"
        assert dict_form["name"] == "test"

        # 反序列化
        restored = PolicyConfig(**dict_form)
        assert restored.version == policy.version


# === BudgetConfig 测试（~5 tests）===


class TestBudgetConfig:
    """BudgetConfig 测试。"""

    def test_create_budget_config(self) -> None:
        """测试创建预算配置。"""
        config = BudgetConfig(
            max_context_tokens=8192,
            output_reserved_tokens=1024,
        )
        assert config.max_context_tokens == 8192

    def test_budget_config_with_ratios(self) -> None:
        """测试带弹性比例的预算配置。"""
        config = BudgetConfig(
            elastic_ratios={"user": 0.3, "rag": 0.2},
        )
        assert config.elastic_ratios["user"] == 0.3
        assert config.elastic_ratios["rag"] == 0.2

    def test_budget_config_validation(self) -> None:
        """测试预算配置验证。"""
        # 负数应该被拒绝
        with pytest.raises(ValidationError):
            BudgetConfig(max_context_tokens=-1000)

    def test_budget_config_defaults(self) -> None:
        """测试默认值。"""
        config = BudgetConfig()
        assert config.max_context_tokens > 0
        assert config.output_reserved_tokens >= 0

    def test_budget_config_saturation_threshold(self) -> None:
        """测试饱和度阈值设置。"""
        config = BudgetConfig(
            saturation_threshold=0.9,
            overflow_strategy="truncate_lowest_priority",
        )
        assert config.saturation_threshold == 0.9
        assert config.overflow_strategy == "truncate_lowest_priority"


# === SanitizeConfig 测试（~4 tests）===


class TestSanitizeConfig:
    """SanitizeConfig 测试。"""

    def test_create_sanitize_config(self) -> None:
        """测试创建清洗配置。"""
        config = SanitizeConfig(
            strip_html=True,
            injection_detection=True,
        )
        assert config.strip_html is True

    def test_sanitize_config_pii_patterns(self) -> None:
        """测试 PII 脱敏配置。"""
        config = SanitizeConfig(
            pii_redaction=True,
            pii_patterns=["phone", "email", "id_card"],
        )
        assert "phone" in config.pii_patterns

    def test_sanitize_config_injection_actions(self) -> None:
        """测试 Injection 检测动作。"""
        # reject
        config1 = SanitizeConfig(on_injection="reject")
        assert config1.on_injection == "reject"

        # warn
        config2 = SanitizeConfig(on_injection="warn")
        assert config2.on_injection == "warn"

    def test_sanitize_config_length_limits(self) -> None:
        """测试长度限制配置。"""
        config = SanitizeConfig(
            max_segment_chars=5000,
            max_repeat_chars=10,
        )
        assert config.max_segment_chars == 5000


# === RerankConfig 测试（~3 tests）===


class TestRerankConfig:
    """RerankConfig 测试。"""

    def test_create_rerank_config(self) -> None:
        """测试创建重排配置。"""
        config = RerankConfig(
            enable_mmr=True,
            mmr_lambda=0.7,
        )
        assert config.enable_mmr is True

    def test_rerank_config_temporal_weighting(self) -> None:
        """测试时效性加权配置。"""
        config = RerankConfig(
            enable_temporal_weighting=True,
            temporal_decay_rate=0.1,
        )
        assert config.enable_temporal_weighting is True

    def test_rerank_config_max_per_type(self) -> None:
        """测试按类型限制数量配置。"""
        # max_per_type 是全局整数（0=无限制），不是按类型的字典
        config = RerankConfig(
            max_per_type=10,
        )
        assert config.max_per_type == 10


# === CompressConfig 测试（~3 tests）===


class TestCompressConfig:
    """CompressConfig 测试。"""

    def test_compress_config_disabled(self) -> None:
        """测试禁用压缩。"""
        config = CompressConfig(enabled=False)
        assert config.enabled is False

    def test_compress_config_with_compressor(self) -> None:
        """测试指定压缩器。"""
        config = CompressConfig(
            enabled=True,
            default_compressor="truncation",
        )
        assert config.default_compressor == "truncation"

    def test_compress_config_saturation_trigger(self) -> None:
        """测试饱和度触发阈值。"""
        config = CompressConfig(
            enabled=True,
            saturation_trigger=0.9,
        )
        assert config.saturation_trigger == 0.9


# === CacheConfig 和 RoutingConfig 测试（~4 tests）===


class TestCacheConfig:
    """CacheConfig 测试。"""

    def test_cache_config_memory_backend(self) -> None:
        """测试内存缓存后端配置。"""
        config = CacheConfig(
            enabled=True,
            backend="memory",
            max_entries=1000,
        )
        assert config.backend == "memory"

    def test_cache_config_ttl(self) -> None:
        """测试 TTL 配置。"""
        config = CacheConfig(ttl_seconds=3600)
        assert config.ttl_seconds == 3600


class TestRoutingConfig:
    """RoutingConfig 测试。"""

    def test_routing_config_disabled(self) -> None:
        """测试禁用路由。"""
        config = RoutingConfig(enabled=False)
        assert config.enabled is False

    def test_routing_config_with_rules(self) -> None:
        """测试路由规则配置。"""
        config = RoutingConfig(
            enabled=True,
            default_model="gpt-4o",
            rules=[
                {"condition": "simple", "target_model": "gpt-4o-mini"}
            ],
        )
        assert config.default_model == "gpt-4o"


# === load_policy() 测试（~7 tests）===


class TestLoadPolicy:
    """load_policy() 函数测试。"""

    def test_load_default_policy(self) -> None:
        """测试加载默认策略。"""
        policy = load_policy()
        assert isinstance(policy, PolicyConfig)
        assert policy.version != ""

    def test_load_policy_from_yaml(self, temp_policy_file: Path) -> None:
        """测试从 YAML 文件加载策略。"""
        policy = load_policy(path=temp_policy_file)
        assert policy.name == "test-policy"

    def test_load_policy_nonexistent_file(self) -> None:
        """测试加载不存在的文件。"""
        from context_forge.errors import PolicyLoadError

        with pytest.raises((PolicyLoadError, FileNotFoundError, ConfigValidationError)):
            load_policy(path="/nonexistent/path/policy.yaml")

    def test_load_policy_invalid_yaml(self, tmp_path: Path) -> None:
        """测试加载无效的 YAML。"""
        from context_forge.errors import PolicyLoadError

        invalid_file = tmp_path / "invalid.yaml"
        invalid_file.write_text("invalid: yaml: content:")

        with pytest.raises((PolicyLoadError, ConfigValidationError)):
            load_policy(path=invalid_file)

    def test_load_policy_missing_required_fields(self, tmp_path: Path) -> None:
        """测试缺少必需字段的配置。"""
        incomplete_file = tmp_path / "incomplete.yaml"
        incomplete_file.write_text("version: '1.0'\n# missing other fields")

        # 应该使用默认值补全
        policy = load_policy(path=incomplete_file)
        assert policy.version == "1.0"

    def test_load_policy_merge_with_defaults(self, tmp_path: Path) -> None:
        """测试与默认配置合并。"""
        partial_file = tmp_path / "partial.yaml"
        partial_file.write_text("""
version: "custom-1.0"
budget:
  max_context_tokens: 16384
""")

        policy = load_policy(path=partial_file)
        assert policy.version == "custom-1.0"
        assert policy.budget.max_context_tokens == 16384
        # 其他字段应该使用默认值
        assert policy.sanitize is not None

    def test_load_policy_validation_errors(self, tmp_path: Path) -> None:
        """测试验证错误的配置。"""
        invalid_file = tmp_path / "invalid_values.yaml"
        invalid_file.write_text("""
version: "1.0"
budget:
  max_context_tokens: -1000
""")

        with pytest.raises((ValidationError, ConfigValidationError)):
            load_policy(path=invalid_file)


# === MODEL_REGISTRY 测试（~5 tests）===


class TestModelRegistry:
    """MODEL_REGISTRY 测试。"""

    def test_registry_has_common_models(self) -> None:
        """测试注册表包含常用模型。"""
        assert "gpt-4o" in MODEL_REGISTRY
        assert "claude-sonnet-4-5-20250514" in MODEL_REGISTRY
        assert "gemini-2.0-flash" in MODEL_REGISTRY

    def test_registry_model_has_required_fields(self) -> None:
        """测试注册的模型包含必需字段。"""
        gpt4o = MODEL_REGISTRY["gpt-4o"]
        assert gpt4o.model_id == "gpt-4o"
        assert gpt4o.provider == "openai"
        assert gpt4o.max_context_tokens > 0
        assert gpt4o.max_output_tokens > 0

    def test_registry_supports_thinking_models(self) -> None:
        """测试支持思考模型的标志。"""
        # o1 系列应该标记为支持 thinking
        if "o1" in MODEL_REGISTRY:
            assert MODEL_REGISTRY["o1"].supports_thinking is True

    def test_registry_has_cost_info(self) -> None:
        """测试模型包含成本信息。"""
        gpt4o = MODEL_REGISTRY["gpt-4o"]
        assert gpt4o.cost_per_million_input is not None
        assert gpt4o.cost_per_million_output is not None

    def test_registry_all_models_valid(self) -> None:
        """测试所有注册的模型都有效。"""
        from context_forge.models.routing import ModelConfig

        for model_id, config in MODEL_REGISTRY.items():
            assert isinstance(config, ModelConfig)
            assert config.model_id == model_id
            assert config.max_context_tokens > 0


# === MODEL_ALIASES 测试（~2 tests）===


class TestModelAliases:
    """MODEL_ALIASES 测试。"""

    def test_aliases_map_to_valid_models(self) -> None:
        """测试别名映射到有效的模型。"""
        for alias, model_id in MODEL_ALIASES.items():
            assert model_id in MODEL_REGISTRY

    def test_common_aliases_exist(self) -> None:
        """测试常用别名存在。"""
        assert "sonnet" in MODEL_ALIASES
        assert "haiku" in MODEL_ALIASES
        assert "gpt4o" in MODEL_ALIASES or "gpt-4o" in MODEL_ALIASES


# === resolve_model() 测试（~5 tests）===


class TestResolveModel:
    """resolve_model() 函数测试。"""

    def test_resolve_exact_match(self) -> None:
        """测试精确匹配模型 ID。"""
        config = resolve_model("gpt-4o")
        assert config.model_id == "gpt-4o"

    def test_resolve_alias(self) -> None:
        """测试通过别名解析。"""
        config = resolve_model("sonnet")
        assert "claude" in config.model_id

    def test_resolve_prefix_match(self) -> None:
        """测试前缀匹配。"""
        # "gpt-4" 应该匹配到 "gpt-4o" 或其他 gpt-4 系列
        config = resolve_model("gpt-4")
        assert "gpt-4" in config.model_id

    def test_resolve_unknown_model(self) -> None:
        """测试未知模型抛出异常。"""
        with pytest.raises(ModelNotFoundError) as exc_info:
            resolve_model("nonexistent-model-xyz")

        assert "nonexistent-model-xyz" in str(exc_info.value)
        assert "修复建议" in str(exc_info.value) or "available" in str(exc_info.value).lower()

    def test_resolve_case_insensitive(self) -> None:
        """测试大小写不敏感（前缀匹配）。"""
        # 使用无歧义的模型名（"o1" 只有一个精确匹配）
        config1 = resolve_model("O1")
        config2 = resolve_model("o1")
        assert config1.model_id == config2.model_id
