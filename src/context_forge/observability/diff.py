"""
DiffEngine — 结构化 Diff，比对两次上下文组装。

→ 6.5.2 Prompt Diff：变更可视化

在迭代优化上下文组装策略时,最常见的需求是："修改策略后,上下文变化了什么?"
传统的文本 diff 无法很好地处理结构化数据,DiffEngine 提供了针对 ContextPackage
的专用对比能力,可以精确识别:

- 哪些 Segment 被添加/删除/修改
- Token 分配的变化（哪些类型增加了配额,哪些减少了）
- 审计日志的差异（哪些 Segment 在新策略下被丢弃/保留）
- 元数据的变化（模型、策略版本等）

这些信息以结构化的 DiffEntry 列表形式呈现,可以输出为文本、JSON 或 Rich 格式。

⚠️ 反模式对照:不提供结构化 diff 的系统在对比上下文时只能手动对照两个 JSON,
无法快速定位关键变化,极大降低了策略调优的效率。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from context_forge.models.context_package import ContextPackage
from context_forge.models.segment import Segment


class DiffType(str, Enum):
    """Diff 变更类型。"""

    ADDED = "added"
    """Segment 被添加"""

    REMOVED = "removed"
    """Segment 被删除"""

    MODIFIED = "modified"
    """Segment 内容被修改"""

    REORDERED = "reordered"
    """Segment 位置被调整"""

    METADATA_CHANGED = "metadata_changed"
    """元数据发生变化"""

    BUDGET_CHANGED = "budget_changed"
    """预算分配发生变化"""


@dataclass(frozen=True)
class DiffEntry:
    """
    单个 Diff 条目。

    # [Design Decision] 使用 frozen dataclass 保证不可变性。

    属性:
        diff_type: 变更类型
        path: 变更路径（如 "segments[0]", "budget.rigid_budget"）
        old_value: 旧值
        new_value: 新值
        description: 人类可读的描述
    """

    diff_type: DiffType
    path: str
    old_value: Any
    new_value: Any
    description: str


@dataclass(frozen=True)
class ContextDiff:
    """
    两个 ContextPackage 之间的完整 Diff。

    → 6.5.2 Prompt Diff

    属性:
        old_package: 旧的 ContextPackage
        new_package: 新的 ContextPackage
        entries: Diff 条目列表
        summary: 汇总信息（添加/删除/修改的数量）
    """

    old_package: ContextPackage
    new_package: ContextPackage
    entries: list[DiffEntry]
    summary: dict[str, int]


class DiffEngine:
    """
    Diff 引擎 — 比对两个 ContextPackage 并生成结构化 Diff。

    → 6.5.2 Prompt Diff

    基本用法::

        engine = DiffEngine()

        # 比对两个 ContextPackage
        diff = await engine.diff(old_package, new_package)

        # 查看摘要
        print(diff.summary)
        # {"added": 2, "removed": 1, "modified": 3, "reordered": 0}

        # 输出为文本
        text = engine.format_text(diff)
        print(text)

        # 输出为 JSON
        json_data = engine.format_json(diff)

    属性:
        ignore_fields: 比对时忽略的字段（如 request_id, created_at）
    """

    def __init__(
        self,
        ignore_fields: list[str] | None = None,
    ) -> None:
        """
        初始化 DiffEngine。

        参数:
            ignore_fields: 比对时忽略的字段列表
        """
        # [Design Decision] 默认忽略 request_id 和 created_at,
        # 这些字段每次组装都不同,但不代表实质性变化。
        self.ignore_fields = ignore_fields or ["request_id", "created_at"]

    async def diff(
        self,
        old_package: ContextPackage,
        new_package: ContextPackage,
    ) -> ContextDiff:
        """
        比对两个 ContextPackage。

        → 6.5.2.1 结构化差异识别

        参数:
            old_package: 旧的 ContextPackage
            new_package: 新的 ContextPackage

        返回:
            ContextDiff 实例
        """
        entries: list[DiffEntry] = []

        # 1. 比对元数据
        entries.extend(self._diff_metadata(old_package, new_package))

        # 2. 比对 Segment 列表
        entries.extend(self._diff_segments(old_package.segments, new_package.segments))

        # 3. 比对预算分配
        entries.extend(self._diff_budget(old_package, new_package))

        # 4. 比对审计日志
        entries.extend(self._diff_audit_log(old_package, new_package))

        # 汇总统计
        summary = self._compute_summary(entries)

        return ContextDiff(
            old_package=old_package,
            new_package=new_package,
            entries=entries,
            summary=summary,
        )

    def format_text(self, diff: ContextDiff, max_entries: int = 50) -> str:
        """
        格式化为人类可读的文本。

        参数:
            diff: ContextDiff 实例
            max_entries: 最大显示条目数

        返回:
            格式化后的文本
        """
        lines = [
            "═══ Context Diff ═══",
            f"Old: {diff.old_package.request_id} ({diff.old_package.model})",
            f"New: {diff.new_package.request_id} ({diff.new_package.model})",
            "",
            "── Summary ──",
        ]

        for key, count in sorted(diff.summary.items()):
            if count > 0:
                lines.append(f"  {key}: {count}")

        if diff.entries:
            lines.append("")
            lines.append("── Changes ──")
            for i, entry in enumerate(diff.entries[:max_entries]):
                lines.append(f"  [{entry.diff_type.value}] {entry.description}")

            if len(diff.entries) > max_entries:
                lines.append(f"  ... 还有 {len(diff.entries) - max_entries} 条变更")

        return "\n".join(lines)

    def format_json(self, diff: ContextDiff) -> dict[str, Any]:
        """
        格式化为 JSON 兼容字典。

        参数:
            diff: ContextDiff 实例

        返回:
            JSON 兼容字典
        """
        return {
            "old_package": {
                "request_id": diff.old_package.request_id,
                "model": diff.old_package.model,
                "policy_version": diff.old_package.policy_version,
            },
            "new_package": {
                "request_id": diff.new_package.request_id,
                "model": diff.new_package.model,
                "policy_version": diff.new_package.policy_version,
            },
            "summary": diff.summary,
            "entries": [
                {
                    "type": entry.diff_type.value,
                    "path": entry.path,
                    "old_value": entry.old_value,
                    "new_value": entry.new_value,
                    "description": entry.description,
                }
                for entry in diff.entries
            ],
        }

    def format_rich(self, diff: ContextDiff) -> str:
        """
        格式化为 Rich 兼容的标记文本。

        参数:
            diff: ContextDiff 实例

        返回:
            Rich 标记文本
        """
        # 🏭 生产提示：这里应集成 Rich 库生成彩色输出
        # 由于 Rich 是可选依赖，这里提供简化版本
        lines = [
            "[bold cyan]═══ Context Diff ═══[/bold cyan]",
            f"[dim]Old:[/dim] {diff.old_package.request_id} ({diff.old_package.model})",
            f"[dim]New:[/dim] {diff.new_package.request_id} ({diff.new_package.model})",
            "",
            "[bold]── Summary ──[/bold]",
        ]

        for key, count in sorted(diff.summary.items()):
            if count > 0:
                color = self._get_diff_color(key)
                lines.append(f"  [{color}]{key}: {count}[/{color}]")

        if diff.entries:
            lines.append("")
            lines.append("[bold]── Changes ──[/bold]")
            for entry in diff.entries:
                color = self._get_diff_color(entry.diff_type.value)
                lines.append(f"  [{color}][{entry.diff_type.value}][/{color}] {entry.description}")

        return "\n".join(lines)

    # --- 内部比对方法 ---

    def _diff_metadata(
        self,
        old: ContextPackage,
        new: ContextPackage,
    ) -> list[DiffEntry]:
        """比对元数据。"""
        entries: list[DiffEntry] = []

        # 模型变化
        if old.model != new.model:
            entries.append(
                DiffEntry(
                    diff_type=DiffType.METADATA_CHANGED,
                    path="model",
                    old_value=old.model,
                    new_value=new.model,
                    description=f"模型从 {old.model} 变更为 {new.model}",
                )
            )

        # 策略版本变化
        if old.policy_version != new.policy_version:
            entries.append(
                DiffEntry(
                    diff_type=DiffType.METADATA_CHANGED,
                    path="policy_version",
                    old_value=old.policy_version,
                    new_value=new.policy_version,
                    description=f"策略版本从 {old.policy_version} 变更为 {new.policy_version}",
                )
            )

        return entries

    def _diff_segments(
        self,
        old_segments: list[Segment],
        new_segments: list[Segment],
    ) -> list[DiffEntry]:
        """比对 Segment 列表。"""
        entries: list[DiffEntry] = []

        # 构建 ID 映射
        old_ids = {seg.id: seg for seg in old_segments}
        new_ids = {seg.id: seg for seg in new_segments}

        # 检测添加的 Segment
        for seg_id, seg in new_ids.items():
            if seg_id not in old_ids:
                entries.append(
                    DiffEntry(
                        diff_type=DiffType.ADDED,
                        path=f"segments[{seg_id}]",
                        old_value=None,
                        new_value=seg.content[:100],
                        description=f"添加 Segment {seg_id} (type={seg.type.value}, {seg.token_count} tokens)",
                    )
                )

        # 检测删除的 Segment
        for seg_id, seg in old_ids.items():
            if seg_id not in new_ids:
                entries.append(
                    DiffEntry(
                        diff_type=DiffType.REMOVED,
                        path=f"segments[{seg_id}]",
                        old_value=seg.content[:100],
                        new_value=None,
                        description=f"删除 Segment {seg_id} (type={seg.type.value}, {seg.token_count} tokens)",
                    )
                )

        # 检测修改的 Segment
        for seg_id in old_ids.keys() & new_ids.keys():
            old_seg = old_ids[seg_id]
            new_seg = new_ids[seg_id]

            if old_seg.content != new_seg.content:
                entries.append(
                    DiffEntry(
                        diff_type=DiffType.MODIFIED,
                        path=f"segments[{seg_id}].content",
                        old_value=old_seg.content[:100],
                        new_value=new_seg.content[:100],
                        description=f"修改 Segment {seg_id} 的内容",
                    )
                )

        # 检测位置变化
        old_positions = {seg.id: i for i, seg in enumerate(old_segments)}
        new_positions = {seg.id: i for i, seg in enumerate(new_segments)}

        for seg_id in old_positions.keys() & new_positions.keys():
            old_pos = old_positions[seg_id]
            new_pos = new_positions[seg_id]

            if old_pos != new_pos:
                entries.append(
                    DiffEntry(
                        diff_type=DiffType.REORDERED,
                        path=f"segments[{seg_id}].position",
                        old_value=old_pos,
                        new_value=new_pos,
                        description=f"Segment {seg_id} 位置从 {old_pos} 变为 {new_pos}",
                    )
                )

        return entries

    def _diff_budget(
        self,
        old: ContextPackage,
        new: ContextPackage,
    ) -> list[DiffEntry]:
        """比对预算分配。"""
        entries: list[DiffEntry] = []

        if not old.budget_allocation or not new.budget_allocation:
            return entries

        old_budget = old.budget_allocation
        new_budget = new.budget_allocation

        # 比对总预算
        if old_budget.total_budget != new_budget.total_budget:
            entries.append(
                DiffEntry(
                    diff_type=DiffType.BUDGET_CHANGED,
                    path="budget.total_budget",
                    old_value=old_budget.total_budget,
                    new_value=new_budget.total_budget,
                    description=f"总预算从 {old_budget.total_budget:,} 变为 {new_budget.total_budget:,}",
                )
            )

        # 比对刚性支出
        if old_budget.rigid_used != new_budget.rigid_used:
            entries.append(
                DiffEntry(
                    diff_type=DiffType.BUDGET_CHANGED,
                    path="budget.rigid_used",
                    old_value=old_budget.rigid_used,
                    new_value=new_budget.rigid_used,
                    description=f"刚性支出从 {old_budget.rigid_used:,} 变为 {new_budget.rigid_used:,}",
                )
            )

        return entries

    def _diff_audit_log(
        self,
        old: ContextPackage,
        new: ContextPackage,
    ) -> list[DiffEntry]:
        """比对审计日志（仅统计级别）。"""
        entries: list[DiffEntry] = []

        old_drops = len(old.dropped_segments)
        new_drops = len(new.dropped_segments)

        if old_drops != new_drops:
            entries.append(
                DiffEntry(
                    diff_type=DiffType.METADATA_CHANGED,
                    path="audit_log.dropped_count",
                    old_value=old_drops,
                    new_value=new_drops,
                    description=f"丢弃的 Segment 数量从 {old_drops} 变为 {new_drops}",
                )
            )

        return entries

    def _compute_summary(self, entries: list[DiffEntry]) -> dict[str, int]:
        """计算汇总统计。"""
        summary: dict[str, int] = {
            "added": 0,
            "removed": 0,
            "modified": 0,
            "reordered": 0,
            "metadata_changed": 0,
            "budget_changed": 0,
        }

        for entry in entries:
            summary[entry.diff_type.value] = summary.get(entry.diff_type.value, 0) + 1

        return summary

    def _get_diff_color(self, diff_type: str) -> str:
        """获取 Diff 类型对应的 Rich 颜色。"""
        colors = {
            "added": "green",
            "removed": "red",
            "modified": "yellow",
            "reordered": "blue",
            "metadata_changed": "cyan",
            "budget_changed": "magenta",
        }
        return colors.get(diff_type, "white")
