# ============================================================
#  Citrus QA Agent 一键启动脚本（v8.5.0）
#  ------------------------------------------------------------
#  零门槛：下载解压后，右键 → 使用 PowerShell 运行（或: powershell -File run.ps1）
#  自动完成: Python 检测/安装 → 虚拟环境 → 依赖安装 → 模型下载 → 启动服务
#  然后自动打开浏览器 http://localhost:8000，在页面填写 DeepSeek API Key 即可使用
# ============================================================
$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$AgentDir = Join-Path $Root 'agent'
$VenvDir = Join-Path $AgentDir '.venv'
$VenvPy = Join-Path $VenvDir 'Scripts\python.exe'
$VenvPip = Join-Path $VenvDir 'Scripts\pip.exe'

function Find-Python {
    # 依次尝试: python / py launcher
    $p = Get-Command python -ErrorAction SilentlyContinue
    if ($p) { return $p.Source }
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        try {
            $v = & $py.Source -3.11 -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $v) { return $v.Trim() }
        } catch { }
    }
    return $null
}

Write-Host ""
Write-Host "  🍊 Citrus QA Agent 一键启动" -ForegroundColor Yellow
Write-Host "  ============================" -ForegroundColor DarkGray

# ── [1/5] Python ──
$py = Find-Python
if (-not $py) {
    Write-Host "[1/5] 未检测到 Python，正在通过 winget 自动安装 Python 3.11 ..." -ForegroundColor Cyan
    try {
        winget install --id Python.Python.3.11 -e --accept-source-agreements --accept-package-agreements
        # winget 安装后需刷新 PATH
        $env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
        $py = Find-Python
    } catch {
        Write-Host "    ⚠ winget 安装失败，请手动安装 Python 3.11（https://www.python.org/downloads/，勾选 Add to PATH）后重新运行本脚本" -ForegroundColor Red
        exit 1
    }
    if (-not $py) {
        Write-Host "    ⚠ 安装完成但未找到 python，请关闭本窗口重开后再运行" -ForegroundColor Red
        exit 1
    }
}
Write-Host "[1/5] Python: $py" -ForegroundColor Green
& $py -c "import sys; assert sys.version_info >= (3, 10), '需要 Python 3.10+'" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "    ⚠ Python 版本过低，请安装 Python 3.11+（https://www.python.org/downloads/）" -ForegroundColor Red
    exit 1
}

# ── [2/5] 虚拟环境 ──
if (-not (Test-Path $VenvPy)) {
    Write-Host "[2/5] 创建虚拟环境（首次一次性）..." -ForegroundColor Cyan
    & $py -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { Write-Host "    虚拟环境创建失败" -ForegroundColor Red; exit 1 }
} else {
    Write-Host "[2/5] 虚拟环境已就绪" -ForegroundColor Green
}

# ── [3/5] 依赖 ──
$depsOk = & $VenvPy -c "import fastapi, uvicorn, langchain_core, fastembed, qdrant_client, pydantic" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[3/5] 安装依赖（首次约 5-10 分钟，取决于网络；进度条较长请耐心等待）..." -ForegroundColor Cyan
    & $VenvPip install -r (Join-Path $Root 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { Write-Host "    依赖安装失败，请检查网络后重试" -ForegroundColor Red; exit 1 }
} else {
    Write-Host "[3/5] 依赖已就绪" -ForegroundColor Green
}

# ── [4/5] 模型 ──
Write-Host "[4/5] 检查模型（首次自动下载向量编码/重排模型，约 5-15 分钟；之后秒级启动）..." -ForegroundColor Cyan
Push-Location $AgentDir
try {
    & $VenvPy prepare_models.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "    模型准备未完全成功，可继续启动（部分功能降级）" -ForegroundColor Yellow
    }
} finally {
    Pop-Location
}

# ── [5/5] 启动 ──
Write-Host "[5/5] 启动服务，浏览器将自动打开 http://localhost:8000 ..." -ForegroundColor Cyan
Write-Host "      （关闭本窗口即停止服务；Key 首次在页面内填写，保存于本机）" -ForegroundColor DarkGray
try {
    Start-Process 'http://localhost:8000'
} catch { }
Push-Location $AgentDir
try {
    & $VenvPy -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
} finally {
    Pop-Location
}
