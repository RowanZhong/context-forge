"""
ContextForge 可观测性扩展方法。

→ 6.5 可观测性套件

将可观测性相关的便捷方法从主 Facade 中拆分出来，
保持 facade.py 聚焦于核心组装逻辑。

提供三个便捷方法：
- diff()：对比两个快照的差异
- snapshot()：保存 ContextPackage 快照
- golden_record()：与黄金快照对比，检测回归
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from context_forge.models.context_package import ContextPackage


class ObservabilityMixin:
    """
    可观测性便捷方法 Mixin。

    → 6.5 可观测性套件

    通过 Mixin 模式注入到 ContextForge 类中，
    提供快照保存、差异对比、黄金集回归测试等便捷方法。

    # [Design Decision] 使用 Mixin 拆分可观测性方法，
    # 保持 facade.py 聚焦于核心组装逻辑（< 800 行）。
    """

    # 以下属性由 ContextForge.__init__() 初始化
    _snapshot_manager: Any
    _debug: bool

    async def diff(
        self, snapshot_id1: str, snapshot_id2: str
    ) -> dict[str, Any]:
        """
        对比两个快照的差异。

        → 6.5.2 Prompt Diff

        参数:
            snapshot_id1: 第一个快照 ID
            snapshot_id2: 第二个快照 ID

        返回:
            差异字典，包含变更的 Segment、预算差异等

        异常:
            RuntimeError: 快照管理器未启用
        """
        if not self._snapshot_manager:
            raise RuntimeError(
                "快照管理器未启用。请在初始化 ContextForge 时设置 "
                "observability.snapshot_enabled=True"
            )

        from context_forge.observability import DiffEngine

        diff_engine = DiffEngine()
        snap1 = await self._snapshot_manager.load(snapshot_id1)
        snap2 = await self._snapshot_manager.load(snapshot_id2)
        context_diff = await diff_engine.diff(snap1.package, snap2.package)
        return diff_engine.format_json(context_diff)

    async def snapshot(self, package: ContextPackage) -> str:
        """
        保存 ContextPackage 快照。

        → 6.5.1 Context Snapshot

        参数:
            package: 要保存的 ContextPackage

        返回:
            快照 ID

        异常:
            RuntimeError: 快照管理器未启用
        """
        if not self._snapshot_manager:
            raise RuntimeError(
                "快照管理器未启用。请在初始化 ContextForge 时设置 "
                "observability.snapshot_enabled=True"
            )

        result: str = await self._snapshot_manager.save(package)
        return result

    async def golden_record(
        self,
        golden_snapshot_id: str,
        current_package: ContextPackage,
    ) -> dict[str, Any]:
        """
        与黄金快照对比，检测回归。

        → 6.5.3 Golden Set 回归测试

        加载黄金快照并与当前 ContextPackage 进行结构化差异对比，
        返回包含是否通过和详细差异的结果字典。

        参数:
            golden_snapshot_id: 黄金快照 ID
            current_package: 当前构建的 ContextPackage

        返回:
            回归检测结果字典，包含:
            - passed: 是否通过（无差异即为通过）
            - summary: 差异汇总
            - entries: 差异条目列表
            - old_package / new_package: 新旧包元信息

        异常:
            RuntimeError: 快照管理器未启用
        """
        if not self._snapshot_manager:
            raise RuntimeError(
                "快照管理器未启用。请在初始化 ContextForge 时设置 "
                "observability.snapshot_enabled=True"
            )

        from context_forge.observability import DiffEngine

        diff_engine = DiffEngine()
        golden_snapshot = await self._snapshot_manager.load(golden_snapshot_id)
        context_diff = await diff_engine.diff(
            golden_snapshot.package, current_package
        )
        result = diff_engine.format_json(context_diff)
        result["passed"] = len(context_diff.entries) == 0
        return result
