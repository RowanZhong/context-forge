"""
基于 LLM 的路由器——可选的高级路由实现。

→ 6.6.1 意图驱动路由

LLM 路由器通过调用 LLM 对用户查询进行意图分类和复杂度判断，
适用于需要语义理解的路由场景（如情感分析、细粒度分类、多意图识别）。

# [Design Decision] LLM 路由器是可选的高级功能，不是默认路径：
# 1. 需要外部依赖 — 必须有可用的 LLM API
# 2. 引入延迟 — 每次路由需要额外的 LLM 调用（50-200ms）
# 3. 产生成本 — 每次路由消耗 Token（虽然少，但累积起来可观）
# 4. Fallback 机制 — LLM 调用失败时自动降级到 RuleBasedRouter
#
# 适用场景：规则引擎难以描述的语义路由（如"检测用户是否在抱怨"）。
# 不适用场景：明确的复杂度分流、关键词路由（规则引擎更快更便宜）。

🏭 生产提示：
实际生产中，LLM 路由器的调用逻辑需要接入真实的 LLM Provider（OpenAI/Anthropic）。
本实现提供了接口和降级机制，具体的 API 调用需要用户根据自己的 Provider 实现。
"""

from __future__ import annotations

import warnings
from typing import Any

from context_forge.errors.exceptions import RoutingError
from context_forge.models.routing import ComplexityLevel, RoutingDecision
from context_forge.routing.base import RoutingContext
from context_forge.routing.complexity import ComplexityEstimator
from context_forge.routing.rule_based import RuleBasedRouter


class LLMRouter:
    """
    基于 LLM 的路由器。

    → 6.6.1 意图驱动路由

    通过调用小型 LLM（如 GPT-4o-mini）对查询进行意图分类，
    然后根据分类结果路由到合适的模型。

    # [Design Decision] Fallback 到 RuleBasedRouter：
    # LLM 调用可能失败（网络问题、配额不足、Provider 降级），
    # 这时不应该让整个路由失败，而是降级到纯规则路由。

    基本用法::

        # 方式一：提供自定义 LLM 调用函数
        def my_llm_call(prompt: str) -> str:
            # 调用你的 LLM Provider
            return openai_client.complete(prompt)

        router = LLMRouter(
            llm_call_fn=my_llm_call,
            fallback_router=RuleBasedRouter(),
        )

        # 方式二：仅使用 fallback（开发/测试阶段）
        router = LLMRouter(
            llm_call_fn=None,  # 总是降级
            fallback_router=RuleBasedRouter(),
        )
    """

    # → 6.6.1.5 语义路由：LLM 分类 Prompt 模板
    # 这个 Prompt 要求 LLM 输出结构化的 JSON，包含复杂度和推理过程

    CLASSIFICATION_PROMPT_TEMPLATE = """你是一个查询复杂度分类器。请分析用户查询的复杂度，并返回 JSON 格式的结果。

复杂度等级定义：
- simple: 简单查询（FAQ、直接查找、格式转换）
- moderate: 中等查询（需要检索和综合分析）
- complex: 复杂查询（需要多步推理、比较、创造性思考）
- expert: 专家级查询（需要深度推理、数学证明、代码生成）

用户查询：
{query}

请返回 JSON（不要有其他文本）：
{{
  "complexity": "simple|moderate|complex|expert",
  "confidence": 0.0-1.0,
  "reasoning": "判断理由"
}}
"""

    def __init__(
        self,
        llm_call_fn: Any | None = None,
        fallback_router: RuleBasedRouter | None = None,
        classifier_model: str = "gpt-4o-mini",
        enable_cache: bool = True,
    ) -> None:
        """
        初始化 LLM 路由器。

        参数:
            llm_call_fn: LLM 调用函数（接收 prompt: str，返回 response: str）
            fallback_router: 降级路由器（LLM 调用失败时使用）
            classifier_model: 分类器使用的模型 ID
            enable_cache: 是否启用分类结果缓存（避免重复调用）
        """
        self.llm_call_fn = llm_call_fn
        self.fallback_router = fallback_router or RuleBasedRouter()
        self.classifier_model = classifier_model
        self.enable_cache = enable_cache

        # 简单的内存缓存（生产环境应使用 Redis 等持久化缓存）
        self._cache: dict[str, dict[str, Any]] = {}

        # 复杂度估计器（用于 fallback）
        self.complexity_estimator = ComplexityEstimator()

    def route(self, context: RoutingContext) -> RoutingDecision:
        """
        执行路由决策。

        参数:
            context: 路由上下文

        返回:
            路由决策结果
        """
        # 1. 检查缓存
        if self.enable_cache and context.query in self._cache:
            cached = self._cache[context.query]
            return self._route_by_classification(
                context,
                complexity=cached["complexity"],
                confidence=cached["confidence"],
                reasoning=cached["reasoning"],
            )

        # 2. 尝试 LLM 分类
        if self.llm_call_fn is not None:
            try:
                classification = self._classify_with_llm(context.query)
                if classification:
                    # 缓存结果
                    if self.enable_cache:
                        self._cache[context.query] = classification

                    return self._route_by_classification(
                        context,
                        complexity=classification["complexity"],
                        confidence=classification["confidence"],
                        reasoning=classification["reasoning"],
                    )
            except Exception as e:
                # LLM 调用失败，记录警告并降级
                warnings.warn(
                    f"LLM 路由失败，降级到规则路由器: {e}",
                    category=RuntimeWarning,
                    stacklevel=2,
                )

        # 3. Fallback 到规则路由器
        return self.fallback_router.route(context)

    def _classify_with_llm(self, query: str) -> dict[str, Any] | None:
        """
        使用 LLM 对查询进行分类。

        参数:
            query: 用户查询文本

        返回:
            分类结果字典（complexity, confidence, reasoning），失败时返回 None
        """
        if self.llm_call_fn is None:
            return None

        # 构建 Prompt
        prompt = self.CLASSIFICATION_PROMPT_TEMPLATE.format(query=query)

        try:
            # 调用 LLM
            # 🏭 生产提示：这里需要处理超时、重试、速率限制等
            response = self.llm_call_fn(prompt)

            # 解析 JSON 响应
            import json

            # 去除可能的 Markdown 代码块包裹
            response = response.strip()
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1])
            if response.startswith("json"):
                response = response[4:].strip()

            data = json.loads(response)

            # 校验字段
            complexity_str = data.get("complexity", "moderate").lower()
            confidence = float(data.get("confidence", 0.5))
            reasoning = data.get("reasoning", "")

            # 映射到 ComplexityLevel
            complexity_map = {
                "simple": ComplexityLevel.SIMPLE,
                "moderate": ComplexityLevel.MODERATE,
                "complex": ComplexityLevel.COMPLEX,
                "expert": ComplexityLevel.EXPERT,
            }
            complexity = complexity_map.get(complexity_str, ComplexityLevel.MODERATE)

            return {
                "complexity": complexity,
                "confidence": confidence,
                "reasoning": reasoning,
            }

        except Exception as e:
            # 解析失败或 LLM 调用失败
            raise RoutingError(
                what="LLM 分类失败",
                why=f"无法解析 LLM 响应或调用失败: {e}",
                how="请检查 LLM Provider 是否正常，或使用 fallback_router",
            ) from e

    def _route_by_classification(
        self,
        context: RoutingContext,
        complexity: ComplexityLevel,
        confidence: float,
        reasoning: str,
    ) -> RoutingDecision:
        """
        根据分类结果进行路由。

        参数:
            context: 路由上下文
            complexity: 分类得出的复杂度
            confidence: 分类置信度
            reasoning: 分类推理

        返回:
            路由决策
        """
        # [Design Decision] 使用 fallback_router 的规则引擎来实际执行路由，
        # LLM 只负责复杂度分类。这样可以复用规则配置，避免重复定义映射逻辑。

        # 临时修改 context 的复杂度（通过在 metadata 中传递）
        modified_context = RoutingContext(
            segments=context.segments,
            query=context.query,
            max_budget_tokens=context.max_budget_tokens,
            current_turn=context.current_turn,
            metadata={
                **(context.metadata or {}),
                "_llm_complexity": complexity,
                "_llm_confidence": confidence,
            },
        )

        # 使用 fallback_router 的规则引擎进行路由
        decision = self.fallback_router.route(modified_context)

        # 更新决策信息，标注为 LLM 路由
        return RoutingDecision(
            selected_model=decision.selected_model,
            complexity=complexity,
            matched_rule=f"llm_classified:{decision.matched_rule}",
            is_fallback=decision.is_fallback,
            confidence=confidence,
            reasoning=f"LLM 分类: {reasoning} | {decision.reasoning}",
            estimated_cost=decision.estimated_cost,
        )


def create_mock_llm_call_fn() -> Any:
    """
    创建 Mock LLM 调用函数（用于测试和示例）。

    → 仅供开发测试使用

    返回:
        Mock 函数，根据查询长度返回固定的分类结果
    """

    def mock_llm_call(prompt: str) -> str:
        """Mock LLM 调用 — 根据查询长度简单分类。"""
        # 从 prompt 中提取查询（简单字符串匹配）
        lines = prompt.split("\n")
        query = ""
        for i, line in enumerate(lines):
            if "用户查询：" in line and i + 1 < len(lines):
                query = lines[i + 1]
                break

        # 简单的长度分类
        query_len = len(query)
        if query_len < 50:
            complexity = "simple"
            confidence = 0.85
            reasoning = "查询简短，可能是简单问题"
        elif query_len < 150:
            complexity = "moderate"
            confidence = 0.75
            reasoning = "查询中等长度，需要一定分析"
        elif query_len < 400:
            complexity = "complex"
            confidence = 0.80
            reasoning = "查询较长，可能需要深度推理"
        else:
            complexity = "expert"
            confidence = 0.90
            reasoning = "查询非常长，可能是专家级问题"

        import json

        return json.dumps(
            {
                "complexity": complexity,
                "confidence": confidence,
                "reasoning": reasoning,
            },
            ensure_ascii=False,
        )

    return mock_llm_call
