# Context Forge Dockerfile
# 多阶段构建：builder（依赖安装）+ runtime（最小运行时）
# 目标：镜像大小 < 500MB

# ===== Stage 1: Builder（依赖构建）=====
FROM python:3.12-slim AS builder

# [Design Decision] 使用 Python 3.12 官方镜像的 slim 变体，平衡大小和功能
# slim 比 alpine 更适合 Python（避免 musl libc 兼容性问题）

LABEL maintainer="Context Forge Contributors"
LABEL description="Context Forge 构建阶段"

# 设置工作目录
WORKDIR /build

# 安装构建依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv（快速 Python 包管理器）
RUN pip install --no-cache-dir uv

# 复制依赖定义文件
COPY pyproject.toml .

# [Design Decision] 使用 CPU-only PyTorch 减小镜像大小
# sentence-transformers 依赖 PyTorch，CPU 版本足够用于 Embedding 计算
# GPU 版本会增加 ~2GB 镜像大小

# 创建虚拟环境并安装依赖（CPU-only）
RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install --no-cache-dir \
        pydantic>=2.0,<3.0 \
        pyyaml>=6.0 \
        tiktoken>=0.7.0 \
        typer>=0.12.0 \
        rich>=13.0 \
        fastapi>=0.111.0 \
        uvicorn[standard]>=0.30.0 \
        opentelemetry-api>=1.25.0 \
        opentelemetry-sdk>=1.25.0 \
        opentelemetry-exporter-otlp>=1.25.0 \
        redis>=5.0 \
        sentence-transformers>=3.0 \
        protobuf>=5.0 \
        anyio>=4.0

# ===== Stage 2: Runtime（最小运行时）=====
FROM python:3.12-slim AS runtime

LABEL maintainer="Context Forge Contributors"
LABEL description="Context Forge - 高性能动态上下文组装引擎"
LABEL version="0.1.0"

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    CONTEXT_FORGE_LOG_LEVEL=INFO \
    CONTEXT_FORGE_ENABLE_CACHE=false \
    CONTEXT_FORGE_SERVER_HOST=0.0.0.0 \
    CONTEXT_FORGE_SERVER_PORT=8000

# 创建非 root 用户（安全最佳实践）
RUN groupadd -r contextforge && \
    useradd -r -g contextforge -u 1000 contextforge && \
    mkdir -p /app /data /cache && \
    chown -R contextforge:contextforge /app /data /cache

# 安装运行时依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 从 builder 阶段复制虚拟环境
COPY --from=builder /opt/venv /opt/venv

# 设置工作目录
WORKDIR /app

# 复制源代码
COPY --chown=contextforge:contextforge src ./src
COPY --chown=contextforge:contextforge configs ./configs

# 复制配置文件
COPY --chown=contextforge:contextforge pyproject.toml .
COPY --chown=contextforge:contextforge README.md .
COPY --chown=contextforge:contextforge LICENSE .

# 安装 Context Forge（editable mode）
RUN . /opt/venv/bin/activate && \
    pip install --no-cache-dir -e .

# 切换到非 root 用户
USER contextforge

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from context_forge import ContextForge; print('OK')" || exit 1

# 暴露端口（HTTP API 服务）
EXPOSE 8000

# 挂载点（数据持久化）
VOLUME ["/data", "/cache"]

# 默认命令：启动 HTTP 服务器
CMD ["context-forge", "serve", "--host", "0.0.0.0", "--port", "8000"]

# ===== 镜像元数据 =====
# 构建命令：
#   docker build -t context-forge:latest .
#   docker build -t context-forge:0.1.0 --build-arg VERSION=0.1.0 .
#
# 运行命令：
#   docker run -p 8000:8000 context-forge:latest
#   docker run -p 8000:8000 -v $(pwd)/data:/data context-forge:latest
#
# 环境变量覆盖：
#   docker run -e CONTEXT_FORGE_LOG_LEVEL=DEBUG -p 8000:8000 context-forge:latest
#
# 预期镜像大小：~450MB（Python 3.12 slim + CPU PyTorch + 依赖）
