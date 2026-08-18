# ============================================================
#  Citrus QA Agent 一键启动脚本（v8.9.0）
#  ------------------------------------------------------------
#  零门槛：下载解压后，右键 → 使用 PowerShell 运行（或: powershell -File run.ps1）
#  自动完成: 语料下载 → Python 检测/安装 → 虚拟环境 → 依赖安装 → 模型下载 → 启动服务
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

function Test-Gpu {
    # v8.13-b5b: 独立显卡探测（排除无驱动的虚拟/基础显示适配器）
    $adapters = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue
    if (-not $adapters) { return $false }
    foreach ($a in $adapters) {
        $n = "$($a.Name)"
        if ($n -match 'Microsoft Basic|Virtual|Remote Display|VMware|Hyper-V|QEMU') { continue }
        return $true
    }
    return $false
}

Write-Host ""
Write-Host "  🍊 Citrus QA Agent 一键启动" -ForegroundColor Yellow
Write-Host "  ============================" -ForegroundColor DarkGray

# ── [1/6] 语料数据（首次从 GitHub Releases 自动下载，之后秒级跳过）──
#  v8.9.0: 向量库已迁移至 LanceDB（data/lancedb），不再需要本地 Qdrant；
#  语料作为 Releases 附件分发，首次运行自动下载约 1.2GB。
#  国内加速: 设置环境变量 GH_MIRROR（例如 https://ghproxy.net/）即可自动加前缀。
$Repo = 'w1-23/citrus-qa-agent'
$ReleaseVersion = '8.9.0'
$CorpusZip = Join-Path $Root 'corpus.zip'
$DataDir = Join-Path $AgentDir 'data'
$LanceDir = Join-Path $DataDir 'lancedb'
if (-not (Test-Path $LanceDir)) {
    Write-Host "[1/6] 未检测到本地语料库，正在从 GitHub Releases 自动下载（约 1.2GB，首次约 10-30 分钟，取决于网络）..." -ForegroundColor Cyan
    $ghBase = if ($env:GH_MIRROR) { $env:GH_MIRROR } else { 'https://github.com' }
    $url = "$ghBase/$Repo/releases/download/v$ReleaseVersion/corpus-$ReleaseVersion.zip"
    Write-Host "      下载地址: $url" -ForegroundColor DarkGray
    try {
        Invoke-WebRequest -Uri $url -OutFile $CorpusZip -UseBasicParsing
        Write-Host "      下载完成，正在解压..." -ForegroundColor Cyan
        Expand-Archive -Path $CorpusZip -DestinationPath $Root -Force
        Remove-Item $CorpusZip -Force
        if (-not (Test-Path $LanceDir)) {
            Write-Host "    ⚠ 压缩包解压后未找到 agent/data/lancedb，请检查压缩包内容" -ForegroundColor Red
            exit 1
        }
    } catch {
        Write-Host "    ⚠ 语料下载失败: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "      可稍后重试；或手动下载 corpus-$ReleaseVersion.zip 解压到本目录后重新运行" -ForegroundColor Yellow
        exit 1
    }
} else {
    Write-Host "[1/6] 语料数据已就绪" -ForegroundColor Green
}

# ── [2/6] Python ──
$py = Find-Python
if (-not $py) {
    Write-Host "[2/6] 未检测到 Python，正在通过 winget 自动安装 Python 3.11 ..." -ForegroundColor Cyan
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
Write-Host "[2/6] Python: $py" -ForegroundColor Green
& $py -c "import sys; assert sys.version_info >= (3, 10), '需要 Python 3.10+'" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "    ⚠ Python 版本过低，请安装 Python 3.11+（https://www.python.org/downloads/）" -ForegroundColor Red
    exit 1
}

# ── [3/6] 虚拟环境 ──
if (-not (Test-Path $VenvPy)) {
    Write-Host "[3/6] 创建虚拟环境（首次一次性）..." -ForegroundColor Cyan
    & $py -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { Write-Host "    虚拟环境创建失败" -ForegroundColor Red; exit 1 }
} else {
    Write-Host "[3/6] 虚拟环境已就绪" -ForegroundColor Green
}

# ── [4/6] 依赖 ──
$depsOk = & $VenvPy -c "import fastapi, uvicorn, langchain_core, fastembed, qdrant_client, lancedb, pydantic" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[4/6] 安装依赖（首次约 5-10 分钟，取决于网络；进度条较长请耐心等待）..." -ForegroundColor Cyan
    & $VenvPip install -r (Join-Path $Root 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { Write-Host "    依赖安装失败，请检查网络后重试" -ForegroundColor Red; exit 1 }
} else {
    Write-Host "[4/6] 依赖已就绪" -ForegroundColor Green
}

# ── [4b/6] GPU 加速（v8.13-b5b：有独立显卡自动换 DirectML 版 onnxruntime）──
#  DirectML 把嵌入/重排模型放进显存（不占内存）且更快；无独显则维持 CPU。
if (Test-Gpu) {
    $dml = & $VenvPy -c "import onnxruntime as ort; print('DmlExecutionProvider' in ort.get_available_providers())" 2>$null
    if ("$dml".Trim() -eq 'True') {
        Write-Host "[4b/6] 检测到独立显卡，GPU 加速已可用（DirectML）✅" -ForegroundColor Green
    } else {
        Write-Host "[4b/6] 检测到独立显卡，安装 DirectML 版 onnxruntime（模型将使用显存、不占内存；约 120MB）..." -ForegroundColor Cyan
        & $VenvPip uninstall -y onnxruntime 2>$null | Out-Null
        & $VenvPip install onnxruntime-directml
        if ($LASTEXITCODE -ne 0) {
            Write-Host "    ⚠ DirectML 安装失败，将使用 CPU 运行（不影响功能，仅嵌入/重排稍慢）" -ForegroundColor Yellow
        } else {
            Write-Host "[4b/6] DirectML 安装完成 ✅" -ForegroundColor Green
        }
    }
} else {
    Write-Host "[4b/6] 未检测到独立显卡，使用 CPU 运行（模型走内存，属正常）" -ForegroundColor DarkGray
}

# ── [5/6] 模型 ──
# v8.9.0: 模型走 HuggingFace 国内镜像（hf-mirror.com）自动下载，无需手动配置；
# 如需官方源，注释下一行即可
$env:HF_ENDPOINT = 'https://hf-mirror.com'
Write-Host "[5/6] 检查模型（首次自动从镜像下载向量编码/重排模型，约 5-15 分钟；之后秒级启动）..." -ForegroundColor Cyan
Push-Location $AgentDir
try {
    & $VenvPy prepare_models.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "    模型准备未完全成功，可继续启动（部分功能降级）" -ForegroundColor Yellow
    }
} finally {
    Pop-Location
}

# ── [6/6] 启动 ──
Write-Host "[6/6] 启动服务，浏览器将自动打开 http://localhost:8000 ..." -ForegroundColor Cyan
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
