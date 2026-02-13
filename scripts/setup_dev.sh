#!/usr/bin/env bash
# Context Forge 开发环境配置脚本（Linux / macOS）
# 使用方式：bash scripts/setup_dev.sh

set -euo pipefail

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

echo_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

echo_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检测操作系统
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    else
        echo_error "不支持的操作系统: $OSTYPE"
        exit 1
    fi
    echo_info "检测到操作系统: $OS"
}

# 检查 Python 版本
check_python() {
    echo_info "检查 Python 版本..."

    if ! command -v python3 &> /dev/null; then
        echo_error "未找到 Python 3，请先安装 Python 3.10+"
        exit 1
    fi

    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
    PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

    if [[ "$PYTHON_MAJOR" -lt 3 ]] || [[ "$PYTHON_MAJOR" -eq 3 && "$PYTHON_MINOR" -lt 10 ]]; then
        echo_error "Python 版本过低: $PYTHON_VERSION（需要 3.10+）"
        exit 1
    fi

    echo_success "Python 版本: $PYTHON_VERSION"
}

# 创建虚拟环境
create_venv() {
    echo_info "创建虚拟环境..."

    if [[ -d ".venv" ]]; then
        echo_warning "虚拟环境已存在，跳过创建"
    else
        python3 -m venv .venv
        echo_success "虚拟环境创建成功"
    fi
}

# 激活虚拟环境
activate_venv() {
    echo_info "激活虚拟环境..."
    source .venv/bin/activate
    echo_success "虚拟环境已激活"
}

# 安装 uv
install_uv() {
    echo_info "安装 uv 包管理器..."

    if command -v uv &> /dev/null; then
        echo_warning "uv 已安装，跳过安装"
    else
        pip install --upgrade pip
        pip install uv
        echo_success "uv 安装成功"
    fi
}

# 安装依赖
install_dependencies() {
    echo_info "安装项目依赖..."

    uv pip install -e ".[dev,docs]"
    echo_success "项目依赖安装成功"
}

# 安装 Redis
install_redis() {
    echo_info "检查 Redis 安装..."

    if command -v redis-server &> /dev/null; then
        echo_success "Redis 已安装"
        return
    fi

    echo_warning "未检测到 Redis，正在安装..."

    if [[ "$OS" == "linux" ]]; then
        # Ubuntu/Debian
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y redis-server
        # Fedora/RHEL/CentOS
        elif command -v dnf &> /dev/null; then
            sudo dnf install -y redis
        elif command -v yum &> /dev/null; then
            sudo yum install -y redis
        else
            echo_error "无法自动安装 Redis，请手动安装"
            return
        fi
    elif [[ "$OS" == "macos" ]]; then
        if command -v brew &> /dev/null; then
            brew install redis
        else
            echo_error "未检测到 Homebrew，请先安装 Homebrew: https://brew.sh/"
            return
        fi
    fi

    echo_success "Redis 安装成功"
}

# 启动 Redis
start_redis() {
    echo_info "启动 Redis 服务..."

    if pgrep -x "redis-server" > /dev/null; then
        echo_warning "Redis 已在运行"
        return
    fi

    if [[ "$OS" == "linux" ]]; then
        if command -v systemctl &> /dev/null; then
            sudo systemctl start redis-server || sudo systemctl start redis
        else
            redis-server --daemonize yes
        fi
    elif [[ "$OS" == "macos" ]]; then
        brew services start redis
    fi

    echo_success "Redis 服务已启动"
}

# 配置环境变量
setup_env() {
    echo_info "配置环境变量..."

    if [[ -f ".env" ]]; then
        echo_warning ".env 文件已存在，跳过创建"
    else
        cp .env.example .env
        echo_success ".env 文件已创建，请根据需要修改"
    fi
}

# 安装 pre-commit 钩子
setup_pre_commit() {
    echo_info "安装 pre-commit 钩子..."

    if [[ -f ".git/hooks/pre-commit" ]]; then
        echo_warning "pre-commit 钩子已存在"
    else
        pre-commit install
        echo_success "pre-commit 钩子安装成功"
    fi
}

# 运行测试
run_tests() {
    echo_info "运行测试套件..."

    pytest tests/ -v --cov=context_forge --cov-report=term

    if [[ $? -eq 0 ]]; then
        echo_success "所有测试通过！"
    else
        echo_error "部分测试失败，请检查"
    fi
}

# 显示完成信息
show_completion() {
    echo ""
    echo_success "=========================================="
    echo_success "开发环境配置完成！"
    echo_success "=========================================="
    echo ""
    echo_info "下一步："
    echo "  1. 激活虚拟环境："
    echo "     source .venv/bin/activate"
    echo ""
    echo "  2. 运行快速上手示例："
    echo "     python examples/quickstart.py"
    echo ""
    echo "  3. 启动 HTTP 服务器："
    echo "     make serve"
    echo ""
    echo "  4. 运行测试："
    echo "     make test"
    echo ""
    echo "  5. 查看所有可用命令："
    echo "     make help"
    echo ""
}

# 主流程
main() {
    echo_info "=========================================="
    echo_info "Context Forge 开发环境配置"
    echo_info "=========================================="
    echo ""

    detect_os
    check_python
    create_venv
    activate_venv
    install_uv
    install_dependencies
    install_redis
    start_redis
    setup_env
    setup_pre_commit

    echo ""
    echo_info "是否运行测试套件？(y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        run_tests
    fi

    show_completion
}

# 执行主流程
main
