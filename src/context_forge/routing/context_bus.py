"""
上下文总线——多 Agent 协调与上下文传递。

→ 6.3.4.1 Namespace Design：System / User / RAG / Tool 的物理隔离

在多 Agent 场景中，不同 Agent 需要维护各自的上下文状态，
同时支持显式的上下文传递（handoff）和事件广播。

ContextBus 提供三个核心能力：
1. Namespace 隔离 — 不同 Agent 的上下文互不干扰
2. Publish/Subscribe — Agent 间的异步消息传递
3. Handoff — 显式的上下文和控制权移交

# [Design Decision] 为什么需要 ContextBus：
# 在多 Agent 系统中，直接共享全局上下文会导致三个问题：
# 1. 信息泄露 — Agent A 的内部推理被 Agent B 看到
# 2. Token 浪费 — Agent B 不需要 Agent A 的全部历史
# 3. Prompt Injection — 恶意 Agent 可以通过上下文影响其他 Agent
#
# ContextBus 通过 Namespace + Visibility 机制解决这些问题。

⚠️ 反模式（→ 6.7.6 Context Leakage）：
不要在多 Agent 场景中共享全局 ContextPackage，会导致敏感信息泄露和 Token 浪费。
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from context_forge.models.control import Visibility
from context_forge.models.segment import Segment
from context_forge.routing.base import AgentContext


@dataclass
class HandoffRequest:
    """
    上下文移交请求。

    → 6.3.4.2 Agent Handoff

    当 Agent A 需要将控制权和部分上下文移交给 Agent B 时，
    创建一个 HandoffRequest。它指定了：
    - 从哪个 Agent 到哪个 Agent
    - 移交哪些 Segment（通过 ID 列表或过滤条件）
    - 移交的原因和元数据

    属性:
        from_agent_id: 源 Agent ID
        to_agent_id: 目标 Agent ID
        segment_ids: 移交的 Segment ID 列表（None 表示移交所有可见 Segment）
        reason: 移交原因
        metadata: 扩展元数据
    """

    from_agent_id: str
    to_agent_id: str
    segment_ids: list[str] | None = None
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextEvent:
    """
    上下文事件——Agent 间的异步消息。

    → 6.3.4.3 Event-Driven Context

    Agent 可以发布事件到 ContextBus，其他 Agent 订阅感兴趣的事件类型。
    这是松耦合的通信方式，不需要 Agent 之间相互知道对方的存在。

    属性:
        event_type: 事件类型（如 "task_completed", "error_occurred"）
        publisher_id: 发布者 Agent ID
        data: 事件数据（任意 JSON 可序列化对象）
        target_namespace: 目标命名空间（None 表示广播到所有）
        timestamp: 事件时间戳
    """

    event_type: str
    publisher_id: str
    data: Any
    target_namespace: str | None = None
    timestamp: float = field(default_factory=lambda: __import__("time").time())


class ContextBus:
    """
    上下文总线——多 Agent 协调中心。

    → 6.3.4 Isolate 策略：多 Agent 上下文隔离

    基本用法::

        from context_forge.routing.base import AgentContext

        # 创建 Agent 上下文
        planner = AgentContext(
            agent_id="planner",
            namespace="planning",
            role="planner",
        )
        executor = AgentContext(
            agent_id="executor",
            namespace="execution",
            role="executor",
        )

        # 创建总线
        bus = ContextBus()
        bus.register_agent(planner)
        bus.register_agent(executor)

        # Planner 发布 Segment
        segment = Segment(...)
        bus.publish_segment(planner, segment)

        # Planner 移交上下文给 Executor
        handoff = HandoffRequest(
            from_agent_id="planner",
            to_agent_id="executor",
            reason="规划完成，开始执行",
        )
        bus.handoff(handoff)

        # Executor 获取可见的 Segment
        visible = bus.get_visible_segments(executor)
    """

    def __init__(self) -> None:
        """初始化上下文总线。"""
        # Agent 注册表
        self._agents: dict[str, AgentContext] = {}

        # Namespace -> Segments 映射
        self._segments_by_namespace: dict[str, list[Segment]] = defaultdict(list)

        # 事件订阅者 (event_type -> [agent_ids])
        self._subscribers: dict[str, list[str]] = defaultdict(list)

        # 事件历史（仅保留最近 N 条，避免内存溢出）
        self._event_history: list[ContextEvent] = []
        self._max_history_size = 100

    def register_agent(self, agent: AgentContext) -> None:
        """
        注册 Agent 到总线。

        参数:
            agent: Agent 上下文
        """
        if agent.agent_id in self._agents:
            warnings.warn(
                f"Agent '{agent.agent_id}' 已存在，将被覆盖",
                category=RuntimeWarning,
                stacklevel=2,
            )
        self._agents[agent.agent_id] = agent

    def unregister_agent(self, agent_id: str) -> None:
        """
        注销 Agent。

        参数:
            agent_id: Agent ID
        """
        if agent_id in self._agents:
            del self._agents[agent_id]

    def publish_segment(
        self,
        agent: AgentContext,
        segment: Segment,
    ) -> None:
        """
        Agent 发布 Segment 到总线。

        → 6.3.4.1 Namespace Design

        Segment 会被添加到 Agent 的命名空间中。
        其他 Agent 是否能看到该 Segment 取决于 Segment 的 visibility 和 namespace。

        参数:
            agent: 发布者 Agent
            segment: 要发布的 Segment
        """
        # 确保 Segment 的 namespace 与 Agent 一致
        # [Design Decision] 如果 Segment 没有显式设置 namespace，使用 Agent 的 namespace
        if segment.control.namespace == "default":
            segment = segment.model_copy(
                update={
                    "control": segment.control.with_namespace(agent.namespace),
                }
            )

        # 添加到对应的 namespace
        self._segments_by_namespace[agent.namespace].append(segment)

    def get_visible_segments(
        self,
        agent: AgentContext,
        include_default: bool = True,
    ) -> list[Segment]:
        """
        获取 Agent 可见的所有 Segment。

        → 6.3.4.1 Namespace Design

        可见性规则：
        1. Agent 总是可以看到自己 namespace 下的所有 Segment
        2. Agent 可以看到 visibility=ALL 的 Segment（如果 include_default=True）
        3. Agent 可以看到显式授权给它的 Segment（visibility=AGENT_ONLY + namespace 匹配）
        4. Agent 看不到 visibility=INTERNAL 的 Segment

        参数:
            agent: 查询者 Agent
            include_default: 是否包含 default namespace 的公共 Segment

        返回:
            可见的 Segment 列表
        """
        visible: list[Segment] = []

        # 1. 自己 namespace 的 Segment
        for seg in self._segments_by_namespace.get(agent.namespace, []):
            if seg.control.visibility != Visibility.INTERNAL:
                visible.append(seg)

        # 2. default namespace 的公共 Segment
        if include_default and agent.namespace != "default":
            for seg in self._segments_by_namespace.get("default", []):
                if seg.control.is_visible_to(agent.namespace):
                    visible.append(seg)

        # 3. 其他 namespace 中显式授权的 Segment
        for namespace, segments in self._segments_by_namespace.items():
            if namespace in (agent.namespace, "default"):
                continue
            for seg in segments:
                if seg.control.visibility == Visibility.AGENT_ONLY:
                    if seg.control.namespace == agent.namespace:
                        visible.append(seg)

        return visible

    def handoff(self, request: HandoffRequest) -> None:
        """
        执行上下文移交。

        → 6.3.4.2 Agent Handoff

        移交的实现是将源 Agent 的部分 Segment 复制到目标 Agent 的 namespace，
        并更新 Segment 的 visibility 确保只有目标 Agent 可见。

        参数:
            request: 移交请求
        """
        # 验证 Agent 存在
        if request.from_agent_id not in self._agents:
            raise ValueError(f"源 Agent '{request.from_agent_id}' 不存在")
        if request.to_agent_id not in self._agents:
            raise ValueError(f"目标 Agent '{request.to_agent_id}' 不存在")

        from_agent = self._agents[request.from_agent_id]
        to_agent = self._agents[request.to_agent_id]

        # 获取要移交的 Segment
        source_segments = self._segments_by_namespace.get(from_agent.namespace, [])

        if request.segment_ids is not None:
            # 仅移交指定 ID 的 Segment
            segment_id_set = set(request.segment_ids)
            segments_to_handoff = [s for s in source_segments if s.id in segment_id_set]
        else:
            # 移交所有非 INTERNAL 的 Segment
            segments_to_handoff = [
                s for s in source_segments if s.control.visibility != Visibility.INTERNAL
            ]

        # 将 Segment 复制到目标 namespace
        # [Design Decision] 复制而非移动，源 Agent 仍然保留原始上下文
        for seg in segments_to_handoff:
            # 更新 namespace 和 visibility
            handoff_seg = seg.model_copy(
                update={
                    "control": seg.control.with_namespace(to_agent.namespace),
                }
            )
            self._segments_by_namespace[to_agent.namespace].append(handoff_seg)

        # 发布 handoff 事件
        event = ContextEvent(
            event_type="context_handoff",
            publisher_id=request.from_agent_id,
            data={
                "to_agent_id": request.to_agent_id,
                "segment_count": len(segments_to_handoff),
                "reason": request.reason,
            },
        )
        self.publish_event(event)

    def publish_event(self, event: ContextEvent) -> None:
        """
        发布事件到总线。

        → 6.3.4.3 Event-Driven Context

        参数:
            event: 上下文事件
        """
        # 记录事件历史
        self._event_history.append(event)
        if len(self._event_history) > self._max_history_size:
            self._event_history.pop(0)

        # 通知订阅者
        # 🏭 生产提示：这里应该异步通知订阅者，避免阻塞发布者
        # 可以使用消息队列（Redis Pub/Sub, RabbitMQ 等）
        subscribers = self._subscribers.get(event.event_type, [])
        for agent_id in subscribers:
            # 这里仅记录，实际通知需要用户实现回调机制
            pass

    def subscribe(self, agent_id: str, event_type: str) -> None:
        """
        订阅事件类型。

        参数:
            agent_id: 订阅者 Agent ID
            event_type: 事件类型
        """
        if agent_id not in self._subscribers[event_type]:
            self._subscribers[event_type].append(agent_id)

    def unsubscribe(self, agent_id: str, event_type: str) -> None:
        """
        取消订阅事件类型。

        参数:
            agent_id: 订阅者 Agent ID
            event_type: 事件类型
        """
        if agent_id in self._subscribers[event_type]:
            self._subscribers[event_type].remove(agent_id)

    def get_recent_events(
        self,
        event_type: str | None = None,
        limit: int = 10,
    ) -> list[ContextEvent]:
        """
        获取最近的事件。

        参数:
            event_type: 事件类型过滤（None 表示所有类型）
            limit: 返回数量限制

        返回:
            事件列表（按时间倒序）
        """
        events = self._event_history
        if event_type is not None:
            events = [e for e in events if e.event_type == event_type]

        # 按时间倒序
        events = sorted(events, key=lambda e: e.timestamp, reverse=True)
        return events[:limit]

    def clear_namespace(self, namespace: str) -> None:
        """
        清空指定 namespace 的所有 Segment。

        参数:
            namespace: 命名空间
        """
        if namespace in self._segments_by_namespace:
            self._segments_by_namespace[namespace] = []

    def get_namespace_stats(self, namespace: str) -> dict[str, Any]:
        """
        获取 namespace 的统计信息。

        参数:
            namespace: 命名空间

        返回:
            统计信息字典
        """
        segments = self._segments_by_namespace.get(namespace, [])
        total_tokens = sum(seg.token_count or 0 for seg in segments)
        type_counts = defaultdict(int)
        for seg in segments:
            type_counts[seg.type.value] += 1

        return {
            "namespace": namespace,
            "segment_count": len(segments),
            "total_tokens": total_tokens,
            "type_distribution": dict(type_counts),
        }
