"""
SnapshotManager — Context Snapshot 保存与加载。

→ 6.5.1 Context Snapshot：完整重现包

Context Snapshot 是上下文组装结果的完整快照，包含重现这次组装所需的全部信息。
它不仅记录最终的 ContextPackage，还包括输入参数、环境配置、审计日志等，
使得任何一次组装都可以在未来完整重现。

这个设计解决了生产环境中最常见的排查场景："用户说 3 天前有个 bad case，
但那次上下文已经丢失了，只能靠日志碎片猜测。" 有了 Snapshot，每次组装都可以
持久化保存，后续可以精确回放、对比、回归测试。

⚠️ 反模式对照：不记录 Snapshot 的系统在排查历史问题时只能靠日志拼凑，
无法精确重现当时的上下文状态，排查效率极低。
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from context_forge.errors.exceptions import SerializationError

if TYPE_CHECKING:
    from context_forge.models.context_package import ContextPackage


def _generate_snapshot_id(request_id: str) -> str:
    """
    生成 Snapshot ID。

    格式：snap_{timestamp}_{request_id[:8]}
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"snap_{timestamp}_{request_id[:8]}"


@dataclass(frozen=True)
class SnapshotMetadata:
    """
    Snapshot 元数据。

    # [Design Decision] 使用 frozen dataclass 保证不可变性。

    属性:
        snapshot_id: Snapshot 唯一 ID
        request_id: 关联的请求 ID
        created_at: 创建时间戳
        model: 目标模型 ID
        policy_version: 策略版本
        tags: 用户自定义标签（用于分类和检索）
    """

    snapshot_id: str
    request_id: str
    created_at: datetime
    model: str
    policy_version: str
    tags: dict[str, str]


@dataclass(frozen=True)
class Snapshot:
    """
    完整的 Context Snapshot。

    → 6.5.1.1 完整重现包

    包含重现一次上下文组装所需的全部信息。

    属性:
        metadata: Snapshot 元数据
        package: ContextPackage 实例
        build_inputs: 原始输入参数（用于重现）
        environment: 环境信息（Python 版本、库版本等）
    """

    metadata: SnapshotMetadata
    package: ContextPackage
    build_inputs: dict[str, Any]
    environment: dict[str, str]


class SnapshotManager:
    """
    Snapshot 管理器 — 负责保存、加载、查询 Context Snapshot。

    → 6.5.1 Context Snapshot

    基本用法::

        manager = SnapshotManager(storage_dir="./snapshots")

        # 保存 Snapshot
        snapshot_id = await manager.save(
            package=package,
            build_inputs={"system_prompt": "...", "messages": [...]},
            tags={"env": "production", "user_id": "12345"},
        )

        # 加载 Snapshot
        snapshot = await manager.load(snapshot_id)

        # 查询 Snapshot
        snapshots = await manager.search(tags={"env": "production"})

    属性:
        storage_dir: 存储目录（默认 ./snapshots）
        auto_cleanup_days: 自动清理 N 天前的 Snapshot（0=不清理）
    """

    def __init__(
        self,
        storage_dir: str | Path = "./snapshots",
        auto_cleanup_days: int = 0,
    ) -> None:
        """
        初始化 SnapshotManager。

        参数:
            storage_dir: 存储目录
            auto_cleanup_days: 自动清理 N 天前的 Snapshot（0=不清理）
        """
        self.storage_dir = Path(storage_dir)
        self.auto_cleanup_days = auto_cleanup_days

        # 创建存储目录
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    async def save(
        self,
        package: ContextPackage,
        build_inputs: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
    ) -> str:
        """
        保存 ContextPackage 为 Snapshot。

        → 6.5.1.1 完整重现包

        参数:
            package: 要保存的 ContextPackage
            build_inputs: 原始输入参数（用于重现）
            tags: 自定义标签（用于分类和检索）

        返回:
            snapshot_id: 生成的 Snapshot ID

        异常:
            SerializationError: 序列化失败
        """
        snapshot_id = _generate_snapshot_id(package.request_id)

        metadata = SnapshotMetadata(
            snapshot_id=snapshot_id,
            request_id=package.request_id,
            created_at=datetime.now(timezone.utc),
            model=package.model,
            policy_version=package.policy_version,
            tags=tags or {},
        )

        # 收集环境信息
        import sys

        environment = {
            "python_version": sys.version,
            "platform": sys.platform,
        }

        # 🏭 生产提示：在生产环境中应记录更多环境信息：
        # - context-forge 版本
        # - 依赖库版本（pydantic, tiktoken 等）
        # - 部署环境标识（k8s pod name, instance id 等）

        snapshot = Snapshot(
            metadata=metadata,
            package=package,
            build_inputs=build_inputs or {},
            environment=environment,
        )

        # 序列化并保存
        try:
            snapshot_data = self._serialize_snapshot(snapshot)
            file_path = self._get_snapshot_path(snapshot_id)
            file_path.write_text(
                json.dumps(snapshot_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            raise SerializationError(
                what=f"保存 Snapshot {snapshot_id} 失败。",
                why=f"序列化或写入文件时出错：{e}",
                how="检查存储目录权限和磁盘空间。",
                details={"snapshot_id": snapshot_id, "error": str(e)},
            ) from e

        # 自动清理旧 Snapshot
        if self.auto_cleanup_days > 0:
            await self._cleanup_old_snapshots()

        return snapshot_id

    async def load(self, snapshot_id: str) -> Snapshot:
        """
        加载指定的 Snapshot。

        参数:
            snapshot_id: Snapshot ID

        返回:
            Snapshot 实例

        异常:
            SerializationError: Snapshot 不存在或加载失败
        """
        file_path = self._get_snapshot_path(snapshot_id)

        if not file_path.exists():
            raise SerializationError(
                what=f"Snapshot {snapshot_id} 不存在。",
                why=f"文件 {file_path} 未找到。",
                how="检查 snapshot_id 是否正确，或该 Snapshot 是否已被清理。",
                details={"snapshot_id": snapshot_id, "file_path": str(file_path)},
            )

        try:
            snapshot_data = json.loads(file_path.read_text(encoding="utf-8"))
            return self._deserialize_snapshot(snapshot_data)
        except Exception as e:
            raise SerializationError(
                what=f"加载 Snapshot {snapshot_id} 失败。",
                why=f"反序列化时出错：{e}",
                how="检查文件是否损坏。如果问题持续，可能需要重新生成该 Snapshot。",
                details={"snapshot_id": snapshot_id, "error": str(e)},
            ) from e

    async def search(
        self,
        tags: dict[str, str] | None = None,
        model: str | None = None,
        limit: int = 100,
    ) -> list[SnapshotMetadata]:
        """
        搜索 Snapshot（按元数据过滤）。

        参数:
            tags: 标签过滤条件（必须完全匹配）
            model: 模型 ID 过滤
            limit: 最大返回数量

        返回:
            符合条件的 SnapshotMetadata 列表（按创建时间倒序）
        """
        results: list[SnapshotMetadata] = []

        # 遍历所有 Snapshot 文件
        for file_path in sorted(self.storage_dir.glob("snap_*.json"), reverse=True):
            if len(results) >= limit:
                break

            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
                metadata = self._deserialize_metadata(data["metadata"])

                # 应用过滤条件
                if tags and not self._match_tags(metadata.tags, tags):
                    continue
                if model and metadata.model != model:
                    continue

                results.append(metadata)
            except Exception as e:
                warnings.warn(f"读取 Snapshot {file_path.name} 失败：{e}", stacklevel=2)
                continue

        return results

    async def delete(self, snapshot_id: str) -> bool:
        """
        删除指定的 Snapshot。

        参数:
            snapshot_id: Snapshot ID

        返回:
            是否成功删除
        """
        file_path = self._get_snapshot_path(snapshot_id)

        if not file_path.exists():
            return False

        try:
            file_path.unlink()
            return True
        except Exception as e:
            warnings.warn(f"删除 Snapshot {snapshot_id} 失败：{e}", stacklevel=2)
            return False

    async def list_all(self) -> list[SnapshotMetadata]:
        """
        列出所有 Snapshot 的元数据。

        返回:
            所有 SnapshotMetadata 列表（按创建时间倒序）
        """
        return await self.search(limit=10000)

    # --- 内部方法 ---

    def _get_snapshot_path(self, snapshot_id: str) -> Path:
        """获取 Snapshot 文件路径。"""
        return self.storage_dir / f"{snapshot_id}.json"

    def _serialize_snapshot(self, snapshot: Snapshot) -> dict[str, Any]:
        """序列化 Snapshot 为 JSON 兼容字典。"""
        # [Design Decision] 使用 model_dump() 保存完整的 ContextPackage,
        # 包括所有 Segment 的完整内容,以便后续能够完整重建。
        package_dict = snapshot.package.model_dump(mode="json")

        return {
            "metadata": {
                "snapshot_id": snapshot.metadata.snapshot_id,
                "request_id": snapshot.metadata.request_id,
                "created_at": snapshot.metadata.created_at.isoformat(),
                "model": snapshot.metadata.model,
                "policy_version": snapshot.metadata.policy_version,
                "tags": snapshot.metadata.tags,
            },
            "package": package_dict,
            "build_inputs": snapshot.build_inputs,
            "environment": snapshot.environment,
        }

    def _deserialize_snapshot(self, data: dict[str, Any]) -> Snapshot:
        """反序列化 Snapshot。"""
        metadata = self._deserialize_metadata(data["metadata"])

        # 从完整的字典重建 ContextPackage
        from context_forge.models.context_package import ContextPackage

        package = ContextPackage.model_validate(data["package"])

        return Snapshot(
            metadata=metadata,
            package=package,
            build_inputs=data["build_inputs"],
            environment=data["environment"],
        )

    def _deserialize_metadata(self, data: dict[str, Any]) -> SnapshotMetadata:
        """反序列化 SnapshotMetadata。"""
        return SnapshotMetadata(
            snapshot_id=data["snapshot_id"],
            request_id=data["request_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            model=data["model"],
            policy_version=data["policy_version"],
            tags=data["tags"],
        )

    def _match_tags(self, snapshot_tags: dict[str, str], filter_tags: dict[str, str]) -> bool:
        """检查 Snapshot 的标签是否匹配过滤条件。"""
        return all(snapshot_tags.get(k) == v for k, v in filter_tags.items())

    async def _cleanup_old_snapshots(self) -> None:
        """清理超过 auto_cleanup_days 天的 Snapshot。"""
        if self.auto_cleanup_days <= 0:
            return

        cutoff_time = datetime.now(timezone.utc).timestamp() - (self.auto_cleanup_days * 86400)

        for file_path in self.storage_dir.glob("snap_*.json"):
            try:
                # 使用文件修改时间判断
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
            except Exception as e:
                warnings.warn(f"清理旧 Snapshot {file_path.name} 失败：{e}", stacklevel=2)
