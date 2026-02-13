"""
策略配置的 Schema 定义与校验。

→ 6.1.2.2 Policy-as-Code：基于 YAML/JSON 的策略编排与版本管理

所有构建策略（Budget Policy、Sanitization Rules、Routing Rules）
通过 YAML 文件定义，本模块定义了 YAML 文件的 Schema 并负责校验。

# [Design Decision] 使用 Pydantic 模型作为 Schema 定义，
# 既能做校验，又能自动生成 JSON Schema 用于编辑器提示——
# 开发者在 VS Code 中编辑 YAML 时就能获得自动补全和错误提示。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from context_forge.models.budget import BudgetPolicy

from pydantic import BaseModel, Field, model_validator


class SanitizeRuleConfig(BaseModel):
    """清洗规则配置。"""

    unicode_normalize: bool = Field(default=True, description="是否启用 Unicode 归一化")
    strip_html: bool = Field(default=True, description="是否剥离 HTML 标签")
    pii_redaction: bool = Field(default=False, description="是否启用 PII 脱敏")
    injection_detection: bool = Field(default=True, description="是否启用 Injection 检测")
    max_segment_chars: int = Field(
        default=50_000,
        description="单个 Segment 最大字符数",
        gt=0,
    )
    max_repeat_chars: int = Field(
        default=100,
        description="允许的最大重复字符数",
        gt=0,
    )
    on_injection: str = Field(
        default="warn_and_remove",
        description="Injection 检测后的处理策略：warn_and_remove / error / log_only",
    )
    injection_level: str = Field(
        default="heuristic",
        description="Injection 检测级别：heuristic / classifier",
    )
    injection_confidence_threshold: float = Field(
        default=0.7,
        description="Injection 检测的置信度阈值（仅 classifier 模式）",
        ge=0.0,
        le=1.0,
    )
    pii_patterns: list[str] = Field(
        default_factory=lambda: ["phone", "email", "id_card"],
        description="需要脱敏的 PII 类型",
    )


class BudgetConfig(BaseModel):
    """预算策略配置。"""

    max_context_tokens: int = Field(default=128_000, description="最大上下文窗口", gt=0)
    output_reserved_tokens: int = Field(default=4_096, description="Output 预留", ge=0)
    thinking_reserved_tokens: int = Field(default=0, description="Thinking 预留", ge=0)
    saturation_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    overflow_strategy: str = Field(default="truncate_lowest_priority")

    elastic_ratios: dict[str, float] = Field(
        default_factory=lambda: {
            "rag": 0.35,
            "user": 0.15,
            "assistant": 0.25,
            "few_shot": 0.10,
            "tool_result": 0.10,
            "tool_definition": 0.05,
        },
        description="弹性区间配额比例",
    )

    @model_validator(mode="after")
    def _validate_elastic_ratios(self) -> BudgetConfig:
        """校验弹性比例总和不超过 1.0。"""
        total = sum(self.elastic_ratios.values())
        if total > 1.01:  # 允许微小的浮点误差
            raise ValueError(
                f"elastic_ratios 的比例总和为 {total:.2f}，超过了 1.0。"
                f"请调整各类型的比例使总和不超过 1.0。"
            )
        return self


class CompressConfig(BaseModel):
    """压缩策略配置。"""

    enabled: bool = Field(default=True, description="是否启用自适应压缩")
    default_compressor: str = Field(
        default="truncation",
        description="默认压缩器：truncation / summary / dedup",
    )
    saturation_trigger: float = Field(
        default=0.85,
        description="触发压缩的饱和度阈值",
        ge=0.0,
        le=1.0,
    )
    preserve_must_keep: bool = Field(
        default=True,
        description="压缩时是否保护 must_keep 标记的 Segment",
    )
    min_segment_tokens: int = Field(
        default=50,
        description="压缩后 Segment 的最小 Token 数",
        gt=0,
    )


class CacheConfig(BaseModel):
    """缓存策略配置。"""

    enabled: bool = Field(default=True, description="是否启用缓存")
    backend: str = Field(default="memory", description="缓存后端：memory / redis")
    prefix_cache: bool = Field(default=True, description="是否启用前缀缓存")
    semantic_cache: bool = Field(default=False, description="是否启用语义缓存")
    ttl_seconds: int = Field(default=3600, description="缓存过期时间（秒）", gt=0)
    max_entries: int = Field(default=1000, description="最大缓存条目数", gt=0)
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis URL")


class RoutingConfig(BaseModel):
    """路由策略配置。"""

    enabled: bool = Field(default=False, description="是否启用意图路由")
    default_model: str = Field(default="gpt-4o", description="默认模型")
    router_type: str = Field(
        default="rule_based",
        description="路由器类型：rule_based / llm",
    )
    rules: list[dict[str, Any]] = Field(
        default_factory=list,
        description="路由规则列表",
    )


class RerankConfig(BaseModel):
    """重排策略配置。"""

    enable_mmr: bool = Field(default=False, description="是否启用 MMR 多样性过滤")
    mmr_lambda: float = Field(
        default=0.7,
        description="MMR 权衡参数（0=仅多样性，1=仅相关性）",
        ge=0.0,
        le=1.0,
    )
    similarity_threshold: float = Field(
        default=0.85,
        description="相似度阈值（超过此值视为重复）",
        ge=0.0,
        le=1.0,
    )
    max_per_type: int = Field(
        default=0,
        description="每种类型的最大 Segment 数（0=无限制）",
        ge=0,
    )
    enable_temporal_weighting: bool = Field(
        default=False,
        description="是否启用时效性加权",
    )
    temporal_decay_rate: float = Field(
        default=0.1,
        description="时效性衰减率（越大衰减越快）",
        ge=0.0,
        le=1.0,
    )
    temporal_min_weight: float = Field(
        default=0.3,
        description="时效性最小权重（防止过度衰减）",
        ge=0.0,
        le=1.0,
    )


class ObservabilityConfig(BaseModel):
    """可观测性配置。"""

    snapshot_enabled: bool = Field(default=True, description="是否生成 Snapshot")
    tracing_enabled: bool = Field(default=False, description="是否启用 OpenTelemetry Tracing")
    metrics_enabled: bool = Field(default=True, description="是否收集指标")
    export_format: str = Field(default="json", description="导出格式：json / otlp")
    snapshot_dir: str = Field(default=".context_forge/snapshots", description="Snapshot 存储目录")


class AntiPatternConfig(BaseModel):
    """
    反模式检测配置。

    → 6.7 反模式检测与诊断

    控制反模式检测器的行为，包括检测阈值和严重性级别。
    """

    enabled: bool = Field(default=True, description="是否启用反模式检测")

    # 各规则的阈值配置
    critical_ratio_threshold: float = Field(
        default=0.5,
        description="CRITICAL 优先级占比阈值（超过此值触发 OveruseCriticalRule）",
        ge=0.0,
        le=1.0,
    )

    rigid_budget_threshold: float = Field(
        default=0.7,
        description="刚性预算占比阈值（超过此值触发 RigidBudgetTooLargeRule）",
        ge=0.0,
        le=1.0,
    )

    compression_ratio_threshold: float = Field(
        default=0.1,
        description="压缩率阈值（低于此值触发 OverCompressionRule）",
        ge=0.0,
        le=1.0,
    )

    ttl_days_threshold: int = Field(
        default=30,
        description="TTL 过期天数阈值（超过此天数触发 ExpiredDataRule）",
        gt=0,
    )

    routing_effectiveness_threshold: float = Field(
        default=0.1,
        description="路由有效性阈值（差异小于此值触发 IneffectiveRoutingRule）",
        ge=0.0,
        le=1.0,
    )

    # 检测模式
    check_on_build: bool = Field(
        default=False,
        description="是否在每次 build() 时自动检测反模式",
    )

    fail_on_critical: bool = Field(
        default=False,
        description="是否在检测到 CRITICAL 级别反模式时抛出异常",
    )


class PolicyConfig(BaseModel):
    """
    完整的策略配置 — 对应 YAML 策略文件的根结构。

    → 6.1.2.2 Policy-as-Code

    这是 YAML 策略文件反序列化后的 Pydantic 模型。
    每个字段都有合理的默认值，遵循"约定优于配置"原则。

    YAML 文件示例::

        version: "1.0"
        budget:
          max_context_tokens: 128000
          output_reserved_tokens: 4096
        sanitize:
          pii_redaction: true
        compress:
          enabled: true
        cache:
          enabled: true
          backend: memory
    """

    version: str = Field(default="1.0", description="策略版本")
    name: str = Field(default="default", description="策略名称")
    description: str = Field(default="", description="策略描述")

    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    sanitize: SanitizeRuleConfig = Field(default_factory=SanitizeRuleConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    compress: CompressConfig = Field(default_factory=CompressConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    antipattern: AntiPatternConfig = Field(default_factory=AntiPatternConfig)

    def to_budget_policy(self) -> BudgetPolicy:
        """将配置转换为 BudgetPolicy 模型。"""
        from context_forge.models.budget import BudgetPolicy
        from context_forge.models.segment import SegmentType

        # 将字符串类型名转换为 SegmentType 枚举
        elastic_ratios = {}
        for type_name, ratio in self.budget.elastic_ratios.items():
            try:
                seg_type = SegmentType(type_name)
                elastic_ratios[seg_type] = ratio
            except ValueError:
                pass  # 忽略无法识别的类型名

        return BudgetPolicy(
            max_context_tokens=self.budget.max_context_tokens,
            output_reserved_tokens=self.budget.output_reserved_tokens,
            thinking_reserved_tokens=self.budget.thinking_reserved_tokens,
            saturation_threshold=self.budget.saturation_threshold,
            overflow_strategy=self.budget.overflow_strategy,
            elastic_ratios=elastic_ratios,
        )
