"""
反模式检测规则实现。

→ 6.7 反模式检测与诊断

本模块实现 10 个核心反模式检测规则：

**CRITICAL 级别**（可能导致功能异常）：
1. MissingTokenCountRule — 检测 token_count 缺失
2. NamespaceLeakageRule — 检测命名空间隔离失败
3. CircularDependencyRule — 检测循环依赖

**WARNING 级别**（效率或成本问题）：
4. OveruseCriticalRule — 检测 CRITICAL 优先级滥用
5. RigidBudgetTooLargeRule — 检测刚性预算过大
6. ExpiredDataRule — 检测过期数据未清理
7. OverCompressionRule — 检测过度压缩

**INFO 级别**（优化建议）：
8. IneffectiveRoutingRule — 检测无效的路由决策
9. CacheKeyCollisionRule — 检测缓存键冲突风险
10. UnusedSanitizerRule — 检测未使用的清洗规则

每个规则都遵循三段式错误信息：What（问题是什么）+ Why（为什么）+ How（如何修复）。
"""

from __future__ import annotations

from datetime import datetime, timezone

from context_forge.antipattern.base import (
    AntiPatternSeverity,
    DetectionContext,
    DetectionResult,
)
from context_forge.models.audit import DecisionType
from context_forge.models.segment import Priority, SegmentType

# ============================================================
# CRITICAL 级别规则（功能异常风险）
# ============================================================


class MissingTokenCountRule:
    """
    检测 token_count 缺失的 Segment。

    → 6.7.2 Dirty Context 反模式

    # [Design Decision] token_count 是预算分配的核心依据。
    # 如果 Segment 缺失 token_count，Budget Manager 无法正确分配预算，
    # 可能导致窗口溢出或预算分配不公。

    严重性: CRITICAL
    """

    @property
    def name(self) -> str:
        return "MissingTokenCountRule"

    @property
    def severity(self) -> AntiPatternSeverity:
        return AntiPatternSeverity.CRITICAL

    def detect(self, context: DetectionContext) -> list[DetectionResult]:
        """检测缺失 token_count 的 Segment。"""
        missing_segments = [
            seg for seg in context.segments
            if seg.token_count is None or seg.token_count == 0
        ]

        if not missing_segments:
            return []

        ids = [seg.id for seg in missing_segments]
        missing_count = len(missing_segments)
        total_count = len(context.segments)

        return [DetectionResult(
            rule_name=self.name,
            severity=self.severity,
            title="检测到缺失 Token 计数的 Segment",
            message=f"在 {total_count} 个 Segment 中，有 {missing_count} 个缺失 token_count 字段。",
            why=(
                "Token 计数是预算分配的核心依据。缺失 token_count 会导致 Budget Manager "
                "无法正确估算窗口占用，可能引发窗口溢出或预算分配不公。"
            ),
            how=(
                "确保所有 Segment 在进入 Pipeline 前经过 NormalizeStage 阶段，"
                "该阶段会自动填充 token_count。如果手动构建 Segment，"
                "请调用 tokenizer.count_tokens() 填充此字段。"
            ),
            segment_ids=ids,
            metadata={
                "missing_count": missing_count,
                "total_count": total_count,
                "missing_ratio": f"{missing_count / total_count:.1%}",
            },
        )]


class NamespaceLeakageRule:
    """
    检测命名空间隔离失败。

    → 6.7.3 Context Clash 反模式

    多 Agent 场景中，不同 Agent 的上下文应通过命名空间隔离。
    如果发现本应属于其他命名空间的 Segment 出现在当前上下文中，
    说明命名空间隔离机制失败。

    严重性: CRITICAL
    """

    @property
    def name(self) -> str:
        return "NamespaceLeakageRule"

    @property
    def severity(self) -> AntiPatternSeverity:
        return AntiPatternSeverity.CRITICAL

    def detect(self, context: DetectionContext) -> list[DetectionResult]:
        """检测命名空间泄漏。"""
        # 获取目标命名空间（从 context.config 或默认为 "default"）
        target_namespace = context.config.get("target_namespace", "default")

        # 查找不属于目标命名空间的 Segment
        leaked_segments = [
            seg for seg in context.segments
            if seg.control and seg.control.namespace
            and seg.control.namespace != target_namespace
            and seg.control.namespace != "global"  # global 命名空间允许跨越
        ]

        if not leaked_segments:
            return []

        # 按来源命名空间分组
        from collections import defaultdict
        by_namespace: dict[str, list[str]] = defaultdict(list)
        for seg in leaked_segments:
            ns = seg.control.namespace if seg.control else "unknown"
            by_namespace[ns].append(seg.id)

        ids = [seg.id for seg in leaked_segments]
        leaked_count = len(leaked_segments)

        namespace_summary = ", ".join(
            f"{ns}({len(ids)} 个)" for ns, ids in by_namespace.items()
        )

        return [DetectionResult(
            rule_name=self.name,
            severity=self.severity,
            title="检测到命名空间泄漏",
            message=(
                f"目标命名空间为 '{target_namespace}'，但发现 {leaked_count} 个 "
                f"Segment 属于其他命名空间：{namespace_summary}。"
            ),
            why=(
                "命名空间用于隔离多 Agent 的上下文。命名空间泄漏会导致 Agent A 的私有信息 "
                "泄漏到 Agent B 的上下文中，引发安全风险或逻辑混乱。"
            ),
            how=(
                "检查 Pipeline 的 IsolateStage 是否正确配置。确保在多 Agent 场景中 "
                "为每个 Agent 指定独立的命名空间，并在 build() 时传入正确的 "
                "target_namespace 参数。"
            ),
            segment_ids=ids,
            metadata={
                "target_namespace": target_namespace,
                "leaked_namespaces": dict(by_namespace),
                "leaked_count": leaked_count,
            },
        )]


class CircularDependencyRule:
    """
    检测 Segment 之间的循环依赖。

    → 6.7.3 Context Clash 反模式

    如果 Segment A 的 depends_on 包含 B，而 B 的 depends_on 包含 A，
    形成循环依赖，流水线无法确定正确的处理顺序。

    严重性: CRITICAL
    """

    @property
    def name(self) -> str:
        return "CircularDependencyRule"

    @property
    def severity(self) -> AntiPatternSeverity:
        return AntiPatternSeverity.CRITICAL

    def detect(self, context: DetectionContext) -> list[DetectionResult]:
        """检测循环依赖。"""
        # 构建依赖图
        # 注意：ControlFlags 当前没有 depends_on 字段，此规则暂时不生效
        # 🏭 生产提示：如果需要依赖管理，需要在 ControlFlags 中添加 depends_on 字段
        graph: dict[str, set[str]] = {}
        for seg in context.segments:
            if seg.control and hasattr(seg.control, "depends_on") and seg.control.depends_on:
                graph[seg.id] = set(seg.control.depends_on)
            else:
                graph[seg.id] = set()

        # DFS 检测环
        def has_cycle(node: str, visited: set[str], rec_stack: set[str]) -> list[str] | None:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.get(node, set()):
                if neighbor not in visited:
                    cycle = has_cycle(neighbor, visited, rec_stack)
                    if cycle:
                        return [node] + cycle
                elif neighbor in rec_stack:
                    # 找到环
                    return [node, neighbor]

            rec_stack.remove(node)
            return None

        # 检测所有节点
        visited: set[str] = set()
        cycles: list[list[str]] = []

        for node in graph:
            if node not in visited:
                cycle = has_cycle(node, visited, set())
                if cycle:
                    cycles.append(cycle)

        if not cycles:
            return []

        # 只报告第一个环（避免重复）
        first_cycle = cycles[0]
        cycle_str = " -> ".join(first_cycle[:4])
        if len(first_cycle) > 4:
            cycle_str += f" -> ... ({len(first_cycle)} 个 Segment)"

        return [DetectionResult(
            rule_name=self.name,
            severity=self.severity,
            title="检测到 Segment 循环依赖",
            message=f"发现 {len(cycles)} 个循环依赖链。示例: {cycle_str}",
            why=(
                "循环依赖会导致流水线无法确定 Segment 的处理顺序，"
                "可能引发死锁或无限循环。"
            ),
            how=(
                "检查 Segment 的 control.depends_on 字段，移除循环引用。"
                "如果 A 和 B 需要相互协作，考虑引入第三个 Coordinator Segment "
                "来打破循环。"
            ),
            segment_ids=first_cycle,
            metadata={
                "cycle_count": len(cycles),
                "first_cycle": first_cycle,
            },
        )]


# ============================================================
# WARNING 级别规则（效率或成本问题）
# ============================================================


class OveruseCriticalRule:
    """
    检测 CRITICAL 优先级滥用。

    → 6.7.2 All-in-Context 反模式

    如果超过 50% 的 Segment 被标记为 CRITICAL，说明优先级设置失当。
    CRITICAL 应仅用于系统提示、Schema 等不可丢弃的内容。

    严重性: WARNING
    """

    @property
    def name(self) -> str:
        return "OveruseCriticalRule"

    @property
    def severity(self) -> AntiPatternSeverity:
        return AntiPatternSeverity.WARNING

    def detect(self, context: DetectionContext) -> list[DetectionResult]:
        """检测 CRITICAL 优先级滥用。"""
        if not context.segments:
            return []

        critical_segments = [
            seg for seg in context.segments
            if seg.priority == Priority.CRITICAL
        ]

        critical_ratio = len(critical_segments) / len(context.segments)
        threshold = context.config.get("critical_ratio_threshold", 0.5)

        if critical_ratio <= threshold:
            return []

        ids = [seg.id for seg in critical_segments]

        return [DetectionResult(
            rule_name=self.name,
            severity=self.severity,
            title="检测到 CRITICAL 优先级滥用",
            message=(
                f"{len(critical_segments)}/{len(context.segments)} "
                f"({critical_ratio:.1%}) 的 Segment 被标记为 CRITICAL，"
                f"超过阈值 {threshold:.1%}。"
            ),
            why=(
                "CRITICAL 优先级表示'不可丢弃'，应仅用于系统提示、Schema 等核心内容。"
                "过度使用 CRITICAL 会导致 Budget Manager 失去弹性调节能力，"
                "在预算不足时无法动态裁剪内容。"
            ),
            how=(
                "审查 Segment 的优先级设置。将非核心内容（如 RAG 片段、对话历史）"
                "调整为 MEDIUM 或 LOW 优先级，仅保留系统提示和 Schema 为 CRITICAL。"
            ),
            segment_ids=ids[:10],  # 仅显示前 10 个
            metadata={
                "critical_count": len(critical_segments),
                "total_count": len(context.segments),
                "critical_ratio": f"{critical_ratio:.1%}",
                "threshold": f"{threshold:.1%}",
            },
        )]


class RigidBudgetTooLargeRule:
    """
    检测刚性预算占比过大。

    → 6.7.2 All-in-Context 反模式

    如果刚性预算（CRITICAL 优先级 Segment）占总预算的 70% 以上，
    弹性预算空间过小，无法应对动态场景。

    严重性: WARNING
    """

    @property
    def name(self) -> str:
        return "RigidBudgetTooLargeRule"

    @property
    def severity(self) -> AntiPatternSeverity:
        return AntiPatternSeverity.WARNING

    def detect(self, context: DetectionContext) -> list[DetectionResult]:
        """检测刚性预算占比。"""
        if not context.budget_allocation or not context.budget_policy:
            return []

        allocation = context.budget_allocation
        policy = context.budget_policy

        # 计算刚性预算占比
        if allocation.content_budget == 0:
            return []

        rigid_ratio = allocation.rigid_used / allocation.content_budget
        threshold = context.config.get("rigid_budget_threshold", 0.7)

        if rigid_ratio <= threshold:
            return []

        return [DetectionResult(
            rule_name=self.name,
            severity=self.severity,
            title="检测到刚性预算占比过大",
            message=(
                f"刚性预算占用 {allocation.rigid_used:,} tokens，"
                f"占总预算的 {rigid_ratio:.1%}，超过阈值 {threshold:.1%}。"
            ),
            why=(
                "刚性预算用于保障 CRITICAL 优先级的 Segment。占比过大会压缩弹性空间，"
                "导致 RAG 片段、对话历史等动态内容无法获得足够配额，"
                "影响上下文的丰富度和多样性。"
            ),
            how=(
                "减少 CRITICAL 优先级 Segment 的数量或长度。将可压缩的系统提示 "
                "改为 HIGH 优先级，或启用压缩功能减少刚性区间占用。"
            ),
            segment_ids=[],
            metadata={
                "rigid_used": allocation.rigid_used,
                "content_budget": allocation.content_budget,
                "rigid_ratio": f"{rigid_ratio:.1%}",
                "threshold": f"{threshold:.1%}",
                "elastic_available": allocation.content_budget - allocation.rigid_used,
            },
        )]


class ExpiredDataRule:
    """
    检测过期数据未清理。

    → 6.7.4 Context Confusion 反模式

    如果 Segment 的 created_at 距今超过 30 天（可配置），
    说明可能存在过期数据未被 TTL 机制清理。

    严重性: WARNING
    """

    @property
    def name(self) -> str:
        return "ExpiredDataRule"

    @property
    def severity(self) -> AntiPatternSeverity:
        return AntiPatternSeverity.WARNING

    def detect(self, context: DetectionContext) -> list[DetectionResult]:
        """检测过期数据。"""
        now = datetime.now(timezone.utc)
        ttl_days = context.config.get("ttl_days_threshold", 30)

        expired_segments = []
        for seg in context.segments:
            if seg.metadata and seg.metadata.injected_at:
                age_days = (now - seg.metadata.injected_at).days
                if age_days > ttl_days:
                    expired_segments.append((seg, age_days))

        if not expired_segments:
            return []

        ids = [seg.id for seg, _ in expired_segments]
        max_age = max(age for _, age in expired_segments)

        return [DetectionResult(
            rule_name=self.name,
            severity=self.severity,
            title="检测到过期数据未清理",
            message=(
                f"发现 {len(expired_segments)} 个 Segment 的年龄超过 {ttl_days} 天，"
                f"最旧的达到 {max_age} 天。"
            ),
            why=(
                "过期数据占用上下文窗口，降低新鲜信息的占比。"
                "长时间未更新的数据可能已过时，影响模型的回答准确性。"
            ),
            how=(
                "启用 TTL（Time-To-Live）机制，在 Rerank 阶段自动过滤过期 Segment。"
                "或在应用层定期清理缓存中的过期数据。"
            ),
            segment_ids=ids[:10],
            metadata={
                "expired_count": len(expired_segments),
                "ttl_days_threshold": ttl_days,
                "max_age_days": max_age,
            },
        )]


class OverCompressionRule:
    """
    检测过度压缩。

    → 6.7.2 All-in-Context 反模式（反向）

    如果压缩后的 Segment 长度 < 原长度的 10%，说明压缩率过高，
    可能丢失关键信息。

    严重性: WARNING
    """

    @property
    def name(self) -> str:
        return "OverCompressionRule"

    @property
    def severity(self) -> AntiPatternSeverity:
        return AntiPatternSeverity.WARNING

    def detect(self, context: DetectionContext) -> list[DetectionResult]:
        """检测过度压缩。"""
        # 从审计日志中查找压缩操作
        over_compressed = []
        threshold = context.config.get("compression_ratio_threshold", 0.1)

        for entry in context.audit_log:
            if entry.decision == DecisionType.COMPRESS:
                # 从 metadata 中获取压缩前后的 token 数
                original_tokens = entry.metadata.get("original_tokens", 0)
                compressed_tokens = entry.metadata.get("compressed_tokens", 0)

                if original_tokens > 0:
                    ratio = compressed_tokens / original_tokens
                    if ratio < threshold:
                        over_compressed.append((entry, ratio))

        if not over_compressed:
            return []

        ids = [entry.segment_id for entry, _ in over_compressed]
        min_ratio = min(ratio for _, ratio in over_compressed)

        return [DetectionResult(
            rule_name=self.name,
            severity=self.severity,
            title="检测到过度压缩",
            message=(
                f"发现 {len(over_compressed)} 个 Segment 的压缩率低于 {threshold:.1%}，"
                f"最低达到 {min_ratio:.1%}。"
            ),
            why=(
                "过度压缩会丢失关键细节，导致模型理解不完整。"
                "虽然节省了 Token，但可能影响回答质量。"
            ),
            how=(
                "调整压缩器的配置，增加 min_segment_tokens 或降低 saturation_trigger。"
                "对于重要的 Segment，设置 control.compressible=False 禁止压缩。"
            ),
            segment_ids=ids[:10],
            metadata={
                "over_compressed_count": len(over_compressed),
                "compression_ratio_threshold": f"{threshold:.1%}",
                "min_compression_ratio": f"{min_ratio:.1%}",
            },
        )]


# ============================================================
# INFO 级别规则（优化建议）
# ============================================================


class IneffectiveRoutingRule:
    """
    检测无效的路由决策。

    → 6.7.4 Context Confusion 反模式

    如果路由器选择了一个模型，但实际使用的窗口大小与目标模型的窗口差异很小，
    说明路由决策可能无效（没有带来实质性的优化）。

    严重性: INFO
    """

    @property
    def name(self) -> str:
        return "IneffectiveRoutingRule"

    @property
    def severity(self) -> AntiPatternSeverity:
        return AntiPatternSeverity.INFO

    def detect(self, context: DetectionContext) -> list[DetectionResult]:
        """检测无效的路由决策。"""
        # 检查是否启用了路由
        if not context.config.get("routing_enabled", False):
            return []

        # 检查是否有路由决策（需从 metadata 传入）
        routing_decision = context.config.get("routing_decision")
        if not routing_decision:
            return []

        # 检查路由前后的窗口大小差异
        original_window = context.config.get("original_window_size", 0)
        selected_window = context.config.get("selected_window_size", 0)

        if original_window == 0 or selected_window == 0:
            return []

        diff_ratio = abs(selected_window - original_window) / original_window
        threshold = context.config.get("routing_effectiveness_threshold", 0.1)

        if diff_ratio >= threshold:
            return []

        return [DetectionResult(
            rule_name=self.name,
            severity=self.severity,
            title="检测到无效的路由决策",
            message=(
                f"路由器选择了窗口大小为 {selected_window:,} 的模型，"
                f"与原模型窗口 {original_window:,} 的差异仅 {diff_ratio:.1%}。"
            ),
            why=(
                "路由的目标是根据任务复杂度选择更合适的模型（更大或更小的窗口）。"
                "如果窗口差异很小，路由决策没有带来实质性的优化，反而增加了复杂度。"
            ),
            how=(
                "检查路由规则配置，确保复杂度阈值设置合理。"
                "如果大多数任务不需要路由，考虑禁用路由功能以简化系统。"
            ),
            segment_ids=[],
            metadata={
                "original_window_size": original_window,
                "selected_window_size": selected_window,
                "diff_ratio": f"{diff_ratio:.1%}",
                "threshold": f"{threshold:.1%}",
            },
        )]


class CacheKeyCollisionRule:
    """
    检测缓存键冲突风险。

    → 6.7.4 Context Confusion 反模式

    如果多个 Segment 具有相同的 provenance.source_id，
    可能导致缓存键冲突（特别是在语义缓存中）。

    严重性: INFO
    """

    @property
    def name(self) -> str:
        return "CacheKeyCollisionRule"

    @property
    def severity(self) -> AntiPatternSeverity:
        return AntiPatternSeverity.INFO

    def detect(self, context: DetectionContext) -> list[DetectionResult]:
        """检测缓存键冲突。"""
        # 按 source_id 分组
        from collections import defaultdict
        by_source_id: dict[str, list[str]] = defaultdict(list)

        for seg in context.segments:
            if seg.provenance and seg.provenance.source_id:
                by_source_id[seg.provenance.source_id].append(seg.id)

        # 查找重复的 source_id
        collisions = {
            source_id: seg_ids
            for source_id, seg_ids in by_source_id.items()
            if len(seg_ids) > 1
        }

        if not collisions:
            return []

        total_collisions = sum(len(ids) for ids in collisions.values())
        collision_examples = list(collisions.items())[:3]
        examples_str = ", ".join(
            f"{source_id}({len(ids)} 个)" for source_id, ids in collision_examples
        )

        return [DetectionResult(
            rule_name=self.name,
            severity=self.severity,
            title="检测到缓存键冲突风险",
            message=(
                f"发现 {len(collisions)} 个 source_id 被多个 Segment 共享，"
                f"涉及 {total_collisions} 个 Segment。示例: {examples_str}"
            ),
            why=(
                "source_id 用于生成缓存键。如果多个 Segment 共享同一个 source_id，"
                "可能导致缓存命中错误的内容，或缓存失效逻辑异常。"
            ),
            how=(
                "确保每个 Segment 的 provenance.source_id 唯一。"
                "可以在 source_id 中加入时间戳或随机后缀（如 'doc_123_v2'）。"
            ),
            segment_ids=[],
            metadata={
                "collision_count": len(collisions),
                "total_segments_affected": total_collisions,
                "examples": {k: len(v) for k, v in collision_examples},
            },
        )]


class UnusedSanitizerRule:
    """
    检测未使用的清洗规则。

    → 6.7.2 Dirty Context 反模式

    如果审计日志中没有任何 SANITIZE 类型的决策，
    说明清洗规则可能未生效，或所有内容都是干净的。

    严重性: INFO
    """

    @property
    def name(self) -> str:
        return "UnusedSanitizerRule"

    @property
    def severity(self) -> AntiPatternSeverity:
        return AntiPatternSeverity.INFO

    def detect(self, context: DetectionContext) -> list[DetectionResult]:
        """检测未使用的清洗规则。"""
        # 检查审计日志中是否有 SANITIZE 决策
        sanitize_entries = [
            entry for entry in context.audit_log
            if entry.decision == DecisionType.SANITIZE
        ]

        if sanitize_entries:
            return []

        # 检查是否有用户输入或 RAG 片段（这些类型通常需要清洗）
        needs_sanitization = [
            seg for seg in context.segments
            if seg.type in (SegmentType.USER, SegmentType.RAG, SegmentType.TOOL_RESULT)
        ]

        if not needs_sanitization:
            # 没有需要清洗的内容，不算问题
            return []

        return [DetectionResult(
            rule_name=self.name,
            severity=self.severity,
            title="检测到未使用的清洗规则",
            message=(
                f"上下文包含 {len(needs_sanitization)} 个需要清洗的 Segment "
                f"(USER/RAG/TOOL_RESULT)，但审计日志中没有 SANITIZE 决策记录。"
            ),
            why=(
                "清洗规则用于移除不可见字符、HTML 标签、检测 Injection 等。"
                "如果清洗规则未生效，可能存在安全风险或格式问题。"
            ),
            how=(
                "检查策略配置，确保 sanitize.unicode_normalize、sanitize.strip_html "
                "等选项已启用。验证 SanitizeStage 是否正确添加到 Pipeline 中。"
            ),
            segment_ids=[seg.id for seg in needs_sanitization[:5]],
            metadata={
                "segments_needing_sanitization": len(needs_sanitization),
                "sanitize_entries_in_audit": len(sanitize_entries),
            },
        )]
