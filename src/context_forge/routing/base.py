"""
路由器基础协议与上下文定义。

→ 6.6 上下文路由与动态调度

路由器解决"不同查询应该用不同模型"的问题。这个模块定义了路由器的标准接口，
以及路由决策所需的上下文信息。

# [Design Decision] 使用 Protocol 而非抽象基类，允许用户实现鸭子类型的路由器，
# 不强制继承。这让集成更灵活——用户可以把现有的路由逻辑包装成符合协议的对象。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from context_forge.models.routing import RoutingDecision
    from context_forge.models.segment import Segment


@dataclass(frozen=True)
class RoutingContext:
    """
    路由决策的上下文信息。

    → 6.6.1 意图驱动路由

    路由器需要的所有输入信息都封装在这个结构中。它包含：
    - 当前请求的所有 Segment（用于分析内容复杂度和类型分布）
    - 用户原始查询文本（用于关键词匹配和复杂度估计）
    - 预算约束（某些路由规则可能考虑成本）
    - 扩展元数据（用于自定义路由逻辑）

    # [DX Decision] 使用 frozen dataclass 而非 Pydantic，
    # 因为它是纯内部数据结构，不需要 JSON 序列化和校验的开销。

    属性:
        segments: 当前请求的所有 Segment
        query: 用户原始查询文本（用于关键词匹配和复杂度分析）
        max_budget_tokens: 预算约束（Token 上限）
        current_turn: 当前对话轮次
        metadata: 扩展元数据（自定义路由字段）
    """

    segments: list[Segment]
    query: str
    max_budget_tokens: int
    current_turn: int = 0
    metadata: dict[str, object] | None = None

    @property
    def total_tokens(self) -> int:
        """计算所有 Segment 的总 Token 数。"""
        return sum(seg.token_count or 0 for seg in self.segments)

    @property
    def segment_types(self) -> set[str]:
        """获取所有出现的 Segment 类型。"""
        return {seg.type.value for seg in self.segments}

    def has_segment_type(self, segment_type: str) -> bool:
        """判断是否包含指定类型的 Segment。"""
        return segment_type in self.segment_types


class Router(Protocol):
    """
    路由器协议——定义路由器的标准接口。

    → 6.6.1 意图驱动路由

    路由器接收 RoutingContext，返回 RoutingDecision。
    具体的路由策略由实现类决定：可以是规则引擎、LLM 分类器、ML 模型等。

    # [Design Decision] route() 方法是同步的，因为大多数路由逻辑是纯计算（启发式规则）。
    # LLMRouter 可以内部使用 asyncio.run() 或提供异步版本，但协议保持简单。
    """

    def route(self, context: RoutingContext) -> RoutingDecision:
        """
        执行路由决策。

        参数:
            context: 路由上下文（包含请求的所有信息）

        返回:
            路由决策结果（包含选中的模型和决策理由）

        异常:
            RoutingError: 路由失败时抛出
        """
        ...


@dataclass(frozen=True)
class AgentContext:
    """
    多 Agent 协调的上下文信息。

    → 6.3.4.1 Namespace Design

    在多 Agent 场景中，不同 Agent 之间需要隔离上下文（避免信息泄露），
    同时支持显式的上下文传递（handoff）。这个结构记录了当前 Agent 的身份和命名空间。

    # [Design Decision] 命名空间是字符串而非枚举，支持动态创建 Agent。
    # 这让用户可以在运行时根据任务类型创建 Agent，而不需要预先注册。

    属性:
        agent_id: Agent 唯一标识符
        namespace: Agent 的命名空间（用于 Segment 可见性过滤）
        role: Agent 角色描述（如 "planner"、"executor"、"reviewer"）
        parent_agent_id: 父 Agent ID（用于 Agent 层级关系）
        metadata: 扩展元数据
    """

    agent_id: str
    namespace: str
    role: str = "default"
    parent_agent_id: str | None = None
    metadata: dict[str, object] | None = None

    def is_child_of(self, parent_id: str) -> bool:
        """判断是否为指定 Agent 的子 Agent。"""
        return self.parent_agent_id == parent_id

    def can_access_namespace(self, target_namespace: str) -> bool:
        """
        判断是否可以访问目标命名空间。

        规则:
            - 总是可以访问自己的命名空间
            - 可以访问 "default" 公共命名空间
            - 其他命名空间需要显式授权（通过 Segment 的 visibility）
        """
        if target_namespace == self.namespace:
            return True
        if target_namespace == "default":
            return True
        return False
