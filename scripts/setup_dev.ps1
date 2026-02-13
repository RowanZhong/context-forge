# Context Forge 开发环境配置脚本（Windows PowerShell）
# 使用方式：powershell -ExecutionPolicy Bypass -File scripts\setup_dev.ps1

# 错误时停止
$ErrorActionPreference = "Stop"

# 颜色输出函数
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# 检查 Python 版本
function Test-Python {
    Write-Info "检查 Python 版本..."

    try {
        $pythonVersion = & python --version 2>&1
        if ($pythonVersion -match "Python (\d+)\.(\d+)\.(\d+)") {
            $major = [int]$matches[1]
            $minor = [int]$matches[2]

            if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 10)) {
                Write-Error "Python 版本过低: $pythonVersion（需要 3.10+）"
                exit 1
            }

            Write-Success "Python 版本: $pythonVersion"
        } else {
            Write-Error "无法解析 Python 版本"
            exit 1
        }
    } catch {
        Write-Error "未找到 Python，请先安装 Python 3.10+"
        Write-Info "下载地址: https://www.python.org/downloads/"
        exit 1
    }
}

# 创建虚拟环境
function New-VirtualEnv {
    Write-Info "创建虚拟环境..."

    if (Test-Path ".venv") {
        Write-Warning "虚拟环境已存在，跳过创建"
    } else {
        python -m venv .venv
        Write-Success "虚拟环境创建成功"
    }
}

# 激活虚拟环境
function Enable-VirtualEnv {
    Write-Info "激活虚拟环境..."

    $activateScript = ".\.venv\Scripts\Activate.ps1"
    if (Test-Path $activateScript) {
        & $activateScript
        Write-Success "虚拟环境已激活"
    } else {
        Write-Error "虚拟环境激活脚本不存在"
        exit 1
    }
}

# 安装 uv
function Install-Uv {
    Write-Info "安装 uv 包管理器..."

    try {
        $uvVersion = & uv --version 2>&1
        Write-Warning "uv 已安装: $uvVersion，跳过安装"
    } catch {
        python -m pip install --upgrade pip
        pip install uv
        Write-Success "uv 安装成功"
    }
}

# 安装依赖
function Install-Dependencies {
    Write-Info "安装项目依赖..."

    uv pip install -e ".[dev,docs]"
    Write-Success "项目依赖安装成功"
}

# 安装 Redis
function Install-Redis {
    Write-Info "检查 Redis 安装..."

    try {
        $redisVersion = & redis-server --version 2>&1
        Write-Success "Redis 已安装: $redisVersion"
        return
    } catch {
        Write-Warning "未检测到 Redis，正在安装..."
    }

    # 检查 Chocolatey
    try {
        $chocoVersion = & choco --version 2>&1
        Write-Info "使用 Chocolatey 安装 Redis..."
        choco install redis-64 -y
        Write-Success "Redis 安装成功"
    } catch {
        Write-Warning "未检测到 Chocolatey，请手动安装 Redis"
        Write-Info "Chocolatey 安装: https://chocolatey.org/install"
        Write-Info "Redis 下载: https://github.com/microsoftarchive/redis/releases"
    }
}

# 启动 Redis
function Start-Redis {
    Write-Info "启动 Redis 服务..."

    try {
        # 检查 Redis 服务是否已安装
        $service = Get-Service -Name Redis -ErrorAction SilentlyContinue

        if ($service) {
            if ($service.Status -eq "Running") {
                Write-Warning "Redis 服务已在运行"
            } else {
                Start-Service -Name Redis
                Write-Success "Redis 服务已启动"
            }
        } else {
            # 尝试作为独立进程启动
            $redisProcess = Get-Process -Name redis-server -ErrorAction SilentlyContinue
            if ($redisProcess) {
                Write-Warning "Redis 进程已在运行"
            } else {
                Start-Process redis-server -WindowStyle Hidden
                Write-Success "Redis 进程已启动"
            }
        }
    } catch {
        Write-Warning "无法启动 Redis，请手动启动"
    }
}

# 配置环境变量
function Set-EnvironmentFile {
    Write-Info "配置环境变量..."

    if (Test-Path ".env") {
        Write-Warning ".env 文件已存在，跳过创建"
    } else {
        Copy-Item ".env.example" ".env"
        Write-Success ".env 文件已创建，请根据需要修改"
    }
}

# 安装 pre-commit 钩子
function Install-PreCommit {
    Write-Info "安装 pre-commit 钩子..."

    if (Test-Path ".git\hooks\pre-commit") {
        Write-Warning "pre-commit 钩子已存在"
    } else {
        try {
            pre-commit install
            Write-Success "pre-commit 钩子安装成功"
        } catch {
            Write-Warning "pre-commit 安装失败，请检查 pre-commit 是否已安装"
        }
    }
}

# 运行测试
function Invoke-Tests {
    Write-Info "运行测试套件..."

    pytest tests/ -v --cov=context_forge --cov-report=term

    if ($LASTEXITCODE -eq 0) {
        Write-Success "所有测试通过！"
    } else {
        Write-Error "部分测试失败，请检查"
    }
}

# 显示完成信息
function Show-Completion {
    Write-Host ""
    Write-Success "=========================================="
    Write-Success "开发环境配置完成！"
    Write-Success "=========================================="
    Write-Host ""
    Write-Info "下一步："
    Write-Host "  1. 激活虚拟环境："
    Write-Host "     .\.venv\Scripts\Activate.ps1"
    Write-Host ""
    Write-Host "  2. 运行快速上手示例："
    Write-Host "     python examples\quickstart.py"
    Write-Host ""
    Write-Host "  3. 启动 HTTP 服务器："
    Write-Host "     context-forge serve"
    Write-Host ""
    Write-Host "  4. 运行测试："
    Write-Host "     pytest tests\"
    Write-Host ""
    Write-Host "  5. 查看所有可用命令："
    Write-Host "     context-forge --help"
    Write-Host ""
}

# 主流程
function Main {
    Write-Info "=========================================="
    Write-Info "Context Forge 开发环境配置"
    Write-Info "=========================================="
    Write-Host ""

    Test-Python
    New-VirtualEnv
    Enable-VirtualEnv
    Install-Uv
    Install-Dependencies
    Install-Redis
    Start-Redis
    Set-EnvironmentFile
    Install-PreCommit

    Write-Host ""
    $response = Read-Host "是否运行测试套件？(y/N)"
    if ($response -match "^[Yy]$") {
        Invoke-Tests
    }

    Show-Completion
}

# 执行主流程
try {
    Main
} catch {
    Write-Error "配置过程中出现错误: $_"
    exit 1
}
