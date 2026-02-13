# Context Forge Makefile
# 常用命令快捷方式

.PHONY: help install install-dev test lint typecheck format clean build docker docs serve

# 默认目标：显示帮助
help:
	@echo "Context Forge - 常用命令"
	@echo ""
	@echo "环境设置:"
	@echo "  make install        安装生产依赖"
	@echo "  make install-dev    安装开发依赖"
	@echo "  make clean          清理构建产物和缓存"
	@echo ""
	@echo "代码质量:"
	@echo "  make lint           运行 Ruff Linter"
	@echo "  make format         自动格式化代码"
	@echo "  make typecheck      运行 MyPy 类型检查"
	@echo "  make test           运行测试套件"
	@echo "  make test-cov       运行测试并生成覆盖率报告"
	@echo ""
	@echo "构建和发布:"
	@echo "  make build          构建 Python 分发包"
	@echo "  make docker         构建 Docker 镜像"
	@echo "  make docker-run     运行 Docker 容器"
	@echo "  make docker-compose 启动 Docker Compose 服务"
	@echo ""
	@echo "文档:"
	@echo "  make docs           构建文档"
	@echo "  make docs-serve     启动文档预览服务器"
	@echo ""
	@echo "开发服务:"
	@echo "  make serve          启动 HTTP API 服务器"
	@echo "  make dev            开发模式（热重载）"

# ===== 环境设置 =====

install:
	@echo "安装生产依赖..."
	pip install -e .

install-dev:
	@echo "安装开发依赖..."
	pip install -e ".[dev,docs]"

clean:
	@echo "清理构建产物和缓存..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .tox/
	rm -rf site/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

# ===== 代码质量 =====

lint:
	@echo "运行 Ruff Linter..."
	ruff check .

format:
	@echo "自动格式化代码..."
	ruff format .
	ruff check --fix .

typecheck:
	@echo "运行 MyPy 类型检查..."
	mypy src/context_forge

test:
	@echo "运行测试套件..."
	pytest tests/ -v

test-cov:
	@echo "运行测试并生成覆盖率报告..."
	pytest tests/ -v --cov=context_forge --cov-report=html --cov-report=term

test-watch:
	@echo "监视模式运行测试..."
	pytest-watch tests/ -v

# ===== 构建和发布 =====

build: clean
	@echo "构建 Python 分发包..."
	python -m build
	@echo "检查分发包..."
	twine check dist/*

docker:
	@echo "构建 Docker 镜像..."
	docker build -t context-forge:latest .

docker-run:
	@echo "运行 Docker 容器..."
	docker run -p 8000:8000 --env-file .env context-forge:latest

docker-compose:
	@echo "启动 Docker Compose 服务..."
	docker-compose up -d
	@echo "查看日志："
	docker-compose logs -f

docker-compose-down:
	@echo "停止 Docker Compose 服务..."
	docker-compose down

docker-compose-rebuild:
	@echo "重新构建并启动 Docker Compose 服务..."
	docker-compose down
	docker-compose build --no-cache
	docker-compose up -d

# ===== 文档 =====

docs:
	@echo "构建文档..."
	mkdocs build

docs-serve:
	@echo "启动文档预览服务器..."
	mkdocs serve

# ===== 开发服务 =====

serve:
	@echo "启动 HTTP API 服务器..."
	context-forge serve --host 0.0.0.0 --port 8000

dev:
	@echo "开发模式（热重载）..."
	uvicorn context_forge.cli.app:app --reload --host 0.0.0.0 --port 8000

# ===== 快捷组合命令 =====

check: lint typecheck test
	@echo "所有检查通过！"

ci: clean install-dev check build
	@echo "CI 流水线完成！"

release: check build
	@echo "准备发布..."
	@echo "请创建 Git 标签并推送："
	@echo "  git tag -a v$(shell grep '^version = ' pyproject.toml | sed 's/version = \"\(.*\)\"/\1/') -m 'Release version $(shell grep '^version = ' pyproject.toml | sed 's/version = \"\(.*\)\"/\1/')'"
	@echo "  git push origin --tags"

# ===== 实用工具 =====

shell:
	@echo "启动交互式 Python Shell..."
	python -c "from context_forge import *; import asyncio"

validate-policy:
	@echo "验证策略文件..."
	context-forge validate configs/default_policy.yaml

init-project:
	@echo "初始化项目配置..."
	context-forge init

# ===== 性能测试 =====

benchmark:
	@echo "运行性能基准测试..."
	python benchmarks/run_benchmarks.py

profile:
	@echo "性能分析..."
	python -m cProfile -o profile.stats examples/quickstart.py
	python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative'); p.print_stats(20)"

# ===== 依赖管理 =====

deps-update:
	@echo "更新依赖..."
	pip install --upgrade pip uv
	uv pip compile pyproject.toml -o requirements.txt

deps-check:
	@echo "检查依赖安全漏洞..."
	pip-audit

# ===== Git 钩子 =====

pre-commit-install:
	@echo "安装 pre-commit 钩子..."
	pre-commit install

pre-commit-run:
	@echo "运行 pre-commit 检查..."
	pre-commit run --all-files
