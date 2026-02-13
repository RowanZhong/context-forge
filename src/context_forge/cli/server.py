"""
FastAPI HTTP 服务器实现。

→ 6.5.4 HTTP API 层

提供 RESTful API，支持以下端点：
- POST /build — 执行 build
- GET /snapshots — 列出快照
- GET /snapshots/{id} — 查看快照
- POST /diff — 比对快照
- GET /metrics — 查看指标
- POST /antipatterns — 检测反模式
- POST /golden/record — 记录 golden case
- POST /golden/verify — 验证 golden case
- GET /health — 健康检查

所有响应遵循统一格式：
{
    "success": bool,
    "data": {...} | null,
    "error": str | null,
    "metadata": {...}
}
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from context_forge import ContextForge
from context_forge.antipattern.base import AntiPatternSeverity
from context_forge.errors.exceptions import ContextForgeError
from context_forge.observability.diff import DiffEngine
from context_forge.observability.snapshot import SnapshotManager

# ============================================================
# Request/Response 模型（Pydantic）
# ============================================================


class BuildRequest(BaseModel):
    """Build 请求模型。"""

    model: str = Field(description="目标模型（如 'gpt-4o', 'claude-sonnet-4-5'）")
    system_prompt: str | None = Field(default=None, description="系统提示")
    messages: list[dict[str, str]] = Field(default_factory=list, description="对话消息列表")
    rag_chunks: list[dict[str, Any]] = Field(default_factory=list, description="RAG 片段")
    few_shot: list[dict[str, str]] = Field(default_factory=list, description="Few-shot 示例")
    tools: list[dict[str, Any]] = Field(default_factory=list, description="工具定义")
    state: dict[str, Any] = Field(default_factory=dict, description="状态信息")
    policy_path: str | None = Field(default=None, description="策略文件路径（可选）")


class BuildResponse(BaseModel):
    """Build 响应模型。"""

    success: bool = Field(description="是否成功")
    data: dict[str, Any] | None = Field(default=None, description="ContextPackage 数据")
    error: str | None = Field(default=None, description="错误信息")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元数据")


class SnapshotListResponse(BaseModel):
    """快照列表响应。"""

    success: bool
    data: list[dict[str, Any]] | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SnapshotDetailResponse(BaseModel):
    """快照详情响应。"""

    success: bool
    data: dict[str, Any] | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DiffRequest(BaseModel):
    """Diff 请求模型。"""

    snapshot_id_1: str = Field(description="第一个快照 ID")
    snapshot_id_2: str = Field(description="第二个快照 ID")


class DiffResponse(BaseModel):
    """Diff 响应模型。"""

    success: bool
    data: dict[str, Any] | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MetricsResponse(BaseModel):
    """指标响应。"""

    success: bool
    data: dict[str, Any] | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AntiPatternRequest(BaseModel):
    """反模式检测请求。"""

    snapshot_id: str | None = Field(default=None, description="快照 ID（可选）")
    segments: list[dict[str, Any]] = Field(default_factory=list, description="Segment 数据（可选）")
    config: dict[str, Any] = Field(default_factory=dict, description="检测配置")


class AntiPatternResponse(BaseModel):
    """反模式检测响应。"""

    success: bool
    data: dict[str, Any] | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GoldenRecordRequest(BaseModel):
    """Golden Case 记录请求。"""

    name: str = Field(description="用例名称")
    snapshot_id: str = Field(description="快照 ID")
    tags: list[str] = Field(default_factory=list, description="标签")


class GoldenRecordResponse(BaseModel):
    """Golden Case 记录响应。"""

    success: bool
    data: dict[str, Any] | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GoldenVerifyRequest(BaseModel):
    """Golden Case 验证请求。"""

    name: str = Field(description="用例名称")
    current_snapshot_id: str = Field(description="当前快照 ID")


class GoldenVerifyResponse(BaseModel):
    """Golden Case 验证响应。"""

    success: bool
    data: dict[str, Any] | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """健康检查响应。"""

    success: bool
    data: dict[str, Any]
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ============================================================
# FastAPI 应用
# ============================================================


def create_app(
    model: str = "gpt-4o",
    policy_path: str | None = None,
    enable_cors: bool = False,
) -> FastAPI:
    """
    创建 FastAPI 应用实例。

    Args:
        model: 默认模型
        policy_path: 策略文件路径
        enable_cors: 是否启用 CORS

    Returns:
        FastAPI 应用实例
    """
    app = FastAPI(
        title="Context Forge API",
        description="高性能动态上下文组装引擎 HTTP API",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS 配置
    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # 🏭 生产提示：应限制为具体的域名
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # 全局异常处理器
    @app.exception_handler(ContextForgeError)
    async def context_forge_error_handler(request: Request, exc: ContextForgeError) -> JSONResponse:
        """处理 ContextForge 异常，返回三段式错误信息。"""
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "data": None,
                "error": str(exc),
                "metadata": {
                    "error_type": type(exc).__name__,
                    "timestamp": datetime.now().isoformat(),
                },
            },
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        """处理 HTTP 异常。"""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "data": None,
                "error": exc.detail,
                "metadata": {
                    "status_code": exc.status_code,
                    "timestamp": datetime.now().isoformat(),
                },
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """处理通用异常。"""
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "data": None,
                "error": f"服务器内部错误: {exc!s}",
                "metadata": {
                    "error_type": type(exc).__name__,
                    "timestamp": datetime.now().isoformat(),
                },
            },
        )

    # 依赖注入：ContextForge 单例
    # [DX Decision] 使用依赖注入避免每次请求都创建新实例
    forge_instance: ContextForge | None = None

    def get_forge() -> ContextForge:
        nonlocal forge_instance
        if forge_instance is None:
            forge_instance = ContextForge(model=model, policy_path=policy_path)
        return forge_instance

    # 依赖注入：SnapshotManager 单例
    snapshot_manager_instance: SnapshotManager | None = None

    def get_snapshot_manager() -> SnapshotManager:
        nonlocal snapshot_manager_instance
        if snapshot_manager_instance is None:
            snapshot_manager_instance = SnapshotManager()
        return snapshot_manager_instance

    # ============================================================
    # API 端点
    # ============================================================

    @app.post("/build", response_model=BuildResponse, summary="执行 Context Build")
    async def build(request: BuildRequest) -> BuildResponse:
        """
        执行上下文组装，返回 ContextPackage。

        # [Design Decision] 允许用户通过请求参数覆盖默认模型和策略，
        # 提供灵活性同时保持合理默认值。
        """
        try:
            # 如果请求中指定了不同的模型或策略，创建新实例
            if request.model != model or request.policy_path != policy_path:
                forge = ContextForge(
                    model=request.model,
                    policy_path=request.policy_path,
                )
            else:
                forge = get_forge()

            # 执行 build
            package = await forge.build(
                system_prompt=request.system_prompt or "",
                messages=request.messages,
                rag_chunks=request.rag_chunks,
                few_shot_examples=request.few_shot,
                tools=request.tools,
                state=request.state,
            )

            # 序列化 ContextPackage
            # 统计审计日志中的决策类型
            from collections import Counter
            decision_counts: Counter[str] = Counter(entry.decision.value for entry in package.audit_log)

            data: dict[str, Any] = {
                "segments": [
                    {
                        "id": seg.id,
                        "type": seg.type.value,
                        "content": seg.content,
                        "priority": seg.priority.value if seg.priority else None,
                        "token_count": seg.token_count,
                    }
                    for seg in package.segments
                ],
                "token_usage": {
                    "total_tokens": package.token_usage.total_tokens,
                    "by_role": package.token_usage.by_role,
                    "by_type": package.token_usage.by_type,
                    "segment_count": package.token_usage.segment_count,
                },
                "model": package.model,
                "policy_version": package.policy_version,
                "audit_summary": {
                    "total_entries": len(package.audit_log),
                    "decisions": dict(decision_counts),
                },
            }

            return BuildResponse(
                success=True,
                data=data,
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "model": request.model,
                },
            )

        except ContextForgeError as e:
            return BuildResponse(
                success=False,
                error=str(e),
                metadata={"error_type": type(e).__name__},
            )

    @app.get("/snapshots", response_model=SnapshotListResponse, summary="列出所有快照")
    async def list_snapshots() -> SnapshotListResponse:
        """列出所有已保存的快照。"""
        try:
            manager = get_snapshot_manager()
            snapshots_metadata = await manager.list_all()

            data = [
                {
                    "snapshot_id": snap.snapshot_id,
                    "model": snap.model,
                    "created_at": snap.created_at.isoformat(),
                    "policy_version": snap.policy_version,
                    "tags": snap.tags,
                }
                for snap in snapshots_metadata
            ]

            return SnapshotListResponse(
                success=True,
                data=data,
                metadata={"total": len(snapshots_metadata)},
            )

        except Exception as e:
            return SnapshotListResponse(
                success=False,
                error=f"列出快照失败: {e!s}",
            )

    @app.get("/snapshots/{snapshot_id}", response_model=SnapshotDetailResponse, summary="查看快照详情")
    async def get_snapshot(snapshot_id: str) -> SnapshotDetailResponse:
        """获取指定快照的详细信息。"""
        try:
            manager = get_snapshot_manager()
            snapshot = await manager.load(snapshot_id)

            data = {
                "snapshot_id": snapshot.metadata.snapshot_id,
                "model": snapshot.metadata.model,
                "created_at": snapshot.metadata.created_at.isoformat(),
                "policy_version": snapshot.metadata.policy_version,
                "segments": [
                    {
                        "id": seg.id,
                        "type": seg.type.value,
                        "content": seg.content[:100] + "..." if len(seg.content) > 100 else seg.content,
                        "token_count": seg.token_count,
                    }
                    for seg in snapshot.package.segments
                ],
                "audit_entries": len(snapshot.package.audit_log),
            }

            return SnapshotDetailResponse(
                success=True,
                data=data,
            )

        except Exception as e:
            # 检查是否是 SerializationError（Snapshot 不存在）
            if "不存在" in str(e):
                raise HTTPException(status_code=404, detail=str(e)) from e
            return SnapshotDetailResponse(
                success=False,
                error=f"加载快照失败: {e!s}",
            )

    @app.post("/diff", response_model=DiffResponse, summary="比对两个快照")
    async def diff_snapshots(request: DiffRequest) -> DiffResponse:
        """比对两个快照，返回差异报告。"""
        try:
            manager = get_snapshot_manager()

            # 加载两个快照
            try:
                snapshot_1 = await manager.load(request.snapshot_id_1)
            except Exception as e:
                raise HTTPException(status_code=404, detail=f"快照 '{request.snapshot_id_1}' 不存在") from e

            try:
                snapshot_2 = await manager.load(request.snapshot_id_2)
            except Exception as e:
                raise HTTPException(status_code=404, detail=f"快照 '{request.snapshot_id_2}' 不存在") from e

            # 使用 DiffEngine 比对
            diff_engine = DiffEngine()
            diff_result = await diff_engine.diff(snapshot_1.package, snapshot_2.package)

            data = {
                "snapshot_1": request.snapshot_id_1,
                "snapshot_2": request.snapshot_id_2,
                "summary": diff_result.summary,
                "entries": [
                    {
                        "type": entry.diff_type.value,
                        "path": entry.path,
                        "old_value": str(entry.old_value)[:50] if entry.old_value else None,
                        "new_value": str(entry.new_value)[:50] if entry.new_value else None,
                        "description": entry.description,
                    }
                    for entry in diff_result.entries[:20]  # 只返回前 20 个变更
                ],
            }

            return DiffResponse(
                success=True,
                data=data,
            )

        except HTTPException:
            raise
        except Exception as e:
            return DiffResponse(
                success=False,
                error=f"比对快照失败: {e!s}",
            )

    @app.get("/metrics", response_model=MetricsResponse, summary="查看系统指标")
    async def get_metrics() -> MetricsResponse:
        """获取系统运行指标（快照统计、缓存命中率等）。"""
        try:
            manager = get_snapshot_manager()
            snapshots_metadata = await manager.list_all()

            # 计算统计信息
            total_snapshots = len(snapshots_metadata)

            # 加载快照以获得 Token 信息
            total_tokens = 0
            for snap_meta in snapshots_metadata:
                try:
                    snap = await manager.load(snap_meta.snapshot_id)
                    total_tokens += snap.package.token_usage.total_tokens
                except Exception:
                    # 跳过加载失败的快照
                    continue

            avg_tokens = total_tokens / total_snapshots if total_snapshots > 0 else 0

            # 按模型分组统计
            from collections import Counter
            model_counts = Counter(snap.model for snap in snapshots_metadata)

            data = {
                "snapshots": {
                    "total": total_snapshots,
                    "total_tokens": total_tokens,
                    "avg_tokens_per_snapshot": round(avg_tokens, 2),
                    "by_model": dict(model_counts),
                },
                # 🏭 生产提示：这里应该接入真实的缓存和可观测性指标
                "cache": {
                    "enabled": False,
                    "hit_rate": 0.0,
                },
                "system": {
                    "uptime_seconds": 0,  # 需要在 app 启动时记录时间
                },
            }

            return MetricsResponse(
                success=True,
                data=data,
            )

        except Exception as e:
            return MetricsResponse(
                success=False,
                error=f"获取指标失败: {e!s}",
            )

    @app.post("/antipatterns", response_model=AntiPatternResponse, summary="检测反模式")
    async def detect_antipatterns(request: AntiPatternRequest) -> AntiPatternResponse:
        """
        检测反模式。

        可以通过快照 ID 或直接传入 Segment 数据进行检测。
        """
        try:
            from context_forge.antipattern.detector import create_default_detector

            # 如果提供了 snapshot_id，从快照加载数据
            if request.snapshot_id:
                manager = get_snapshot_manager()
                try:
                    snapshot = await manager.load(request.snapshot_id)
                except Exception as e:
                    raise HTTPException(status_code=404, detail=f"快照 '{request.snapshot_id}' 不存在") from e

                # 使用 detect_from_package 方法
                detector = create_default_detector(config=request.config)
                results = detector.detect_from_package(snapshot.package, config=request.config)
            else:
                # 从请求中解析 Segment
                # 🏭 生产提示：这里需要完整的反序列化逻辑
                # 对于简单情况，返回错误提示用户提供快照 ID
                raise HTTPException(
                    status_code=400,
                    detail="请提供 snapshot_id。直接的 Segment 反序列化尚未实现。"
                )

            # 格式化结果
            data = {
                "total_issues": len(results),
                "by_severity": {
                    "CRITICAL": len([r for r in results if r.severity == AntiPatternSeverity.CRITICAL]),
                    "WARNING": len([r for r in results if r.severity == AntiPatternSeverity.WARNING]),
                    "INFO": len([r for r in results if r.severity == AntiPatternSeverity.INFO]),
                },
                "issues": [
                    {
                        "rule_name": result.rule_name,
                        "severity": result.severity.value,
                        "title": result.title,
                        "message": result.message,
                        "why": result.why,
                        "how": result.how,
                        "segment_ids": result.segment_ids[:5],  # 只返回前 5 个
                    }
                    for result in results
                ],
            }

            return AntiPatternResponse(
                success=True,
                data=data,
            )

        except HTTPException:
            raise
        except Exception as e:
            return AntiPatternResponse(
                success=False,
                error=f"检测反模式失败: {e!s}",
            )

    @app.post("/golden/record", response_model=GoldenRecordResponse, summary="记录 Golden Case")
    async def record_golden_case(request: GoldenRecordRequest) -> GoldenRecordResponse:
        """将快照记录为 Golden Case，用于回归测试。"""
        try:
            manager = get_snapshot_manager()
            try:
                snapshot = await manager.load(request.snapshot_id)
            except Exception as e:
                raise HTTPException(status_code=404, detail=f"快照 '{request.snapshot_id}' 不存在") from e

            # 🏭 生产提示：这里应实现真实的 Golden Case 存储逻辑
            # 目前我们将快照保存为一个特殊的 Golden Case
            # 使用 SnapshotManager 的 save 方法，但标记为 golden case
            # 将 list[str] 转换为 dict[str, str]（每个 tag 作为 key，value 为空字符串）
            golden_tags: dict[str, str] = {tag: "" for tag in (request.tags or [])}
            golden_tags["golden_case"] = request.name

            golden_snapshot_id = await manager.save(
                package=snapshot.package,
                build_inputs=snapshot.build_inputs,
                tags=golden_tags,
            )

            return GoldenRecordResponse(
                success=True,
                data={
                    "golden_id": golden_snapshot_id,
                    "name": request.name,
                    "snapshot_id": request.snapshot_id,
                    "tags": request.tags,
                },
            )

        except HTTPException:
            raise
        except Exception as e:
            return GoldenRecordResponse(
                success=False,
                error=f"记录 Golden Case 失败: {e!s}",
            )

    @app.post("/golden/verify", response_model=GoldenVerifyResponse, summary="验证 Golden Case")
    async def verify_golden_case(request: GoldenVerifyRequest) -> GoldenVerifyResponse:
        """将当前快照与 Golden Case 比对，检查是否有回归。"""
        try:
            manager = get_snapshot_manager()

            # 搜索 Golden Case（按标签 golden_case 匹配）
            golden_cases = await manager.search(tags={"golden_case": request.name})
            if not golden_cases:
                raise HTTPException(status_code=404, detail=f"Golden Case '{request.name}' 不存在")

            # 加载 Golden Case（使用最新的）
            golden_snapshot = await manager.load(golden_cases[0].snapshot_id)

            # 加载当前快照
            try:
                current_snapshot = await manager.load(request.current_snapshot_id)
            except Exception as e:
                raise HTTPException(status_code=404, detail=f"快照 '{request.current_snapshot_id}' 不存在") from e

            # 使用 DiffEngine 比对
            diff_engine = DiffEngine()
            diff_result = await diff_engine.diff(golden_snapshot.package, current_snapshot.package)

            # 判断是否完全相同（仅检查 summary 中是否全为 0）
            passed = all(count == 0 for count in diff_result.summary.values())

            data = {
                "passed": passed,
                "golden_case": request.name,
                "current_snapshot_id": request.current_snapshot_id,
                "diff": diff_result.summary,
            }

            return GoldenVerifyResponse(
                success=True,
                data=data,
            )

        except HTTPException:
            raise
        except Exception as e:
            return GoldenVerifyResponse(
                success=False,
                error=f"验证 Golden Case 失败: {e!s}",
            )

    @app.get("/health", response_model=HealthResponse, summary="健康检查")
    async def health_check() -> HealthResponse:
        """健康检查端点，返回服务状态。"""
        return HealthResponse(
            success=True,
            data={
                "status": "healthy",
                "version": "0.1.0",
                "timestamp": datetime.now().isoformat(),
            },
        )

    return app
