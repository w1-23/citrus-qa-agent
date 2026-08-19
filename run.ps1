# ============================================================
#  Citrus QA Agent 一键启动脚本（v8.14.1）
#  ------------------------------------------------------------
#  零门槛：下载解压后，右键 → 使用 PowerShell 运行（或: powershell -File run.ps1）
#  自动完成: 语料下载 → Python 检测/安装 → 虚拟环境 → 依赖安装 → 模型下载 → 启动服务
#  然后自动打开浏览器 http://localhost:8000，在页面填写 DeepSeek API Key 即可使用
# ============================================================
$ErrorActionPreference = 'Stop'
# v8.13-b5c: 全局兜底——任何未捕获异常都停下让用户看得见错误，不再闪退
trap {
    Write-Host ""
    Write-Host "  ⚠ 脚本异常中断: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "    请把上方完整输出（特别是 Python Traceback）复制反馈给维护者" -ForegroundColor Yellow
    Read-Host "    按回车键关闭窗口"
    exit 2
}
$Root = $PSScriptRoot
$AgentDir = Join-Path $Root 'agent'
$VenvDir = Join-Path $AgentDir '.venv'
$VenvPy = Join-Path $VenvDir 'Scripts\python.exe'
$VenvPip = Join-Path $VenvDir 'Scripts\pip.exe'

# v8.13-b5g: 中文/非 ASCII 路径守卫——Windows 下 onnxruntime / HF 缓存 / GBK 控制台在中文路径会报莫名 Traceback
$badChar = $Root.ToCharArray() | Where-Object { [int]$_ -gt 127 } | Select-Object -First 1
if ($badChar) {
    Write-Host ""
    Write-Host "  ⚠ 项目目录路径包含中文字符（当前: $Root）" -ForegroundColor Red
    Write-Host "    Windows 下 Python 组件（模型加载/检索库）在中文路径下会报莫名 Traceback" -ForegroundColor Yellow
    Write-Host "    解决: 关闭本窗口，把整个文件夹改名到纯英文路径（如 E:\citrus），数据再放回 agent\ 下后重新运行" -ForegroundColor Yellow
    Read-Host "    按回车键关闭窗口"
    exit 1
}

# v8.13-b5h: 统一 Python 运行时环境（作用于后续所有 python/uvicorn 子进程）——
#   PYTHONUTF8=1: 根治 GBK 代码页下打印 ✓/中文的 UnicodeEncodeError 崩溃
#   FASTEMBED_CACHE_PATH/HF_HOME: 模型缓存锚定项目内 agent\.hf_cache（整目录拷贝即自带，免重新下载）
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:FASTEMBED_CACHE_PATH = Join-Path (Join-Path $AgentDir '.hf_cache') 'fastembed'
$env:HF_HOME = Join-Path (Join-Path $AgentDir '.hf_cache') 'hf'

function Find-Python {
    # v8.13-b5d: 优先 py 启动器的 3.11/3.12/3.10（避免拿到 3.13+ 装不上依赖），
    #  再退回 PATH 上的 python.exe（版本仍会在 [2/6] 严格校验）
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        foreach ($minor in 11, 12, 10) {
            try {
                $v = & $py.Source -3.$minor -c "import sys; print(sys.executable)" 2>$null
                if ($LASTEXITCODE -eq 0 -and $v) { return $v.Trim() }
            } catch { }
        }
    }
    $p = Get-Command python -ErrorAction SilentlyContinue
    if ($p) { return $p.Source }
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
#  v8.13.0: 向量库已迁移至 LanceDB（data/lancedb），不再需要本地 Qdrant；
#  语料作为 Releases 附件分发，首次运行自动下载约 1.2GB。
#  国内加速: 设置环境变量 GH_MIRROR（例如 https://ghproxy.net/）即可自动加前缀。
$Repo = 'w1-23/citrus-qa-agent'
$ReleaseVersion = '8.14.1'
$CorpusZip = Join-Path $Root 'corpus.zip'
$DataDir = Join-Path $AgentDir 'data'
$LanceDir = Join-Path $DataDir 'lancedb'

# ── [0/6] 包完整性自检（防止解压错包/旧包/漏拷贝；v8.13-b5i）──
$cfgVer = $null
$cfgFile = Join-Path $AgentDir 'src\config.py'
if (Test-Path $cfgFile) {
    $m = Select-String -Path $cfgFile -Pattern 'VERSION\s*=\s*"([^"]+)"' | Select-Object -First 1
    if ($m) { $cfgVer = $m.Matches[0].Groups[1].Value }
}
if ($cfgVer -and $cfgVer -ne $ReleaseVersion) {
    Write-Host ""
    Write-Host "  ⚠ 当前包版本: $cfgVer，预期: v$ReleaseVersion —— 你在用旧包！" -ForegroundColor Red
    Write-Host "    旧发布包（v8.5.0 / v8.9.0）已废止删除；请从 https://github.com/w1-23/citrus-qa-agent/releases 下载最新主包" -ForegroundColor Yellow
    Write-Host ""
} else {
    Write-Host "  ✔ 包版本校验通过: v$ReleaseVersion" -ForegroundColor DarkGray
}
Write-Host "  ── 部署完整性自检 ──" -ForegroundColor DarkGray
$hc = Join-Path $AgentDir '.hf_cache'
$checks = @(
    @{ n = "代码包 (agent\src\config.py)";            p = $cfgFile },
    @{ n = "语料库 (agent\data\lancedb)";            p = $LanceDir },
    @{ n = "e5 编码模型缓存 (.hf_cache\fastembed)";   p = (Join-Path (Join-Path $hc 'fastembed') 'models--qdrant--multilingual-e5-large-onnx') },
    @{ n = "重排模型缓存 (.hf_cache\onnx_reranker\model.onnx)"; p = (Join-Path (Join-Path $hc 'onnx_reranker') 'model.onnx') }
)
foreach ($c in $checks) {
    if (Test-Path $c.p) { Write-Host "    ✔ $($c.n)" -ForegroundColor Green }
    else { Write-Host "    ⚠ $($c.n) —— 缺失：将自动在线下载（首次较慢）；若已拷贝，请检查是否放对位置" -ForegroundColor Yellow }
}
Write-Host ""

if (-not (Test-Path $LanceDir)) {
    Write-Host "[1/6] 未检测到本地语料库，正在从 GitHub Releases 自动下载语料分卷（约 2.2GB，首次约 20-40 分钟，取决于网络）..." -ForegroundColor Cyan
    $ghBase = if ($env:GH_MIRROR) { $env:GH_MIRROR } else { 'https://github.com' }
    $partNo = 1
    while ($true) {
        $url = "$ghBase/$Repo/releases/download/v$ReleaseVersion/corpus-$ReleaseVersion-$partNo.zip"
        Write-Host "      分卷 ${partNo}: $url" -ForegroundColor DarkGray
        try {
            Invoke-WebRequest -Uri $url -OutFile $CorpusZip -UseBasicParsing
            Write-Host "      分卷 $partNo 下载完成，解压合并中..." -ForegroundColor Cyan
            Expand-Archive -Path $CorpusZip -DestinationPath $Root -Force
            Remove-Item $CorpusZip -Force
        } catch {
            if ($partNo -eq 1) {
                Write-Host "    ⚠ 语料下载失败: $($_.Exception.Message)" -ForegroundColor Red
                Write-Host "      可稍后重试；或手动下载 corpus-v$ReleaseVersion-1.zip 解压到本目录后重新运行" -ForegroundColor Yellow
                Read-Host "按回车键关闭窗口"; exit 1
            }
            break
        }
        $partNo++
    }
    if (-not (Test-Path $LanceDir)) {
        Write-Host "    ⚠ 分卷解压后未找到 agent/data/lancedb，请检查压缩包内容" -ForegroundColor Red
        Read-Host "按回车键关闭窗口"; exit 1
    }
} else {
    Write-Host "[1/6] 语料数据已就绪" -ForegroundColor Green
}

# ── [2/6] Python（v8.13-b5e: 不迁就目标机——没有就装 3.11，版本不对也自动装 3.11，与现有并存）──
function Install-Py311 {
    Write-Host "    正在通过 winget 自动安装 Python 3.11（与现有 Python 并存互不影响，约 1-2 分钟）..." -ForegroundColor Cyan
    try {
        winget install --id Python.Python.3.11 -e --accept-source-agreements --accept-package-agreements
        # winget 安装后刷新 PATH（installer 同时注册 py 启动器）
        $env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
    } catch {
        Write-Host "    ⚠ winget 不可用或安装失败，请手动安装 Python 3.11：https://www.python.org/downloads/ （勾选 Add to PATH）后重跑本脚本" -ForegroundColor Red
        Read-Host "    按回车键关闭窗口"; exit 1
    }
}
$py = Find-Python
if (-not $py) {
    Write-Host "[2/6] 未检测到 Python，自动安装 Python 3.11..." -ForegroundColor Cyan
    Install-Py311
    $py = Find-Python
    if (-not $py) {
        Write-Host "    ⚠ 安装完成仍未找到 python，请关闭窗口重开后再运行" -ForegroundColor Red
        Read-Host "    按回车键关闭窗口"; exit 1
    }
}
$verInfo = & $py -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
if ($verInfo -notmatch '^3\.(10|11|12)$') {
    Write-Host "[2/6] 当前 Python $verInfo 不在支持范围（3.10~3.12），自动安装 Python 3.11 ..." -ForegroundColor Yellow
    Install-Py311
    $py = Find-Python
    $verInfo = if ($py) { & $py -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null } else { '' }
    if ($verInfo -notmatch '^3\.(10|11|12)$') {
        Write-Host "    ⚠ 自动安装后版本仍为 $verInfo，请手动安装 Python 3.11：https://www.python.org/downloads/ （勾选 Add to PATH）后重跑" -ForegroundColor Red
        Read-Host "    按回车键关闭窗口"; exit 1
    }
    Write-Host "    ✅ 已自动改用 Python 3.11" -ForegroundColor Green
}
Write-Host "[2/6] Python: $py (v$verInfo)" -ForegroundColor Green

# ── [3/6] 虚拟环境 ──
if (-not (Test-Path $VenvPy)) {
    Write-Host "[3/6] 创建虚拟环境（首次一次性）..." -ForegroundColor Cyan
    & $py -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { Write-Host "    虚拟环境创建失败" -ForegroundColor Red; Read-Host "按回车键关闭窗口"; exit 1 }
} else {
    Write-Host "[3/6] 虚拟环境已就绪" -ForegroundColor Green
}

# ── [4/6] 依赖 ──
$depsOk = & $VenvPy -c "import fastapi, uvicorn, langchain_core, fastembed, qdrant_client, lancedb, pydantic" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[4/6] 安装依赖（首次约 5-10 分钟，取决于网络；进度条较长请耐心等待）..." -ForegroundColor Cyan
    & $VenvPip install -r (Join-Path $Root 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { Write-Host "    依赖安装失败，请检查网络后重试" -ForegroundColor Red; Read-Host "按回车键关闭窗口"; exit 1 }
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
            Write-Host "    ⚠ DirectML 安装失败，自动恢复 CPU 版 onnxruntime（避免启动时找不到运行库崩掉）..." -ForegroundColor Yellow
            & $VenvPip install onnxruntime
            if ($LASTEXITCODE -ne 0) {
                Write-Host "    ⚠ onnxruntime 恢复失败，请检查网络后重新运行（否则服务将无法启动）" -ForegroundColor Red
                Read-Host "    按回车键关闭窗口"; exit 1
            }
            Write-Host "    已恢复 CPU 版（不影响功能，仅嵌入/重排稍慢）" -ForegroundColor Green
        } else {
            Write-Host "[4b/6] DirectML 安装完成 ✅" -ForegroundColor Green
        }
    }
} else {
    Write-Host "[4b/6] 未检测到独立显卡，使用 CPU 运行（模型走内存，属正常）" -ForegroundColor DarkGray
}

# ── [5/6] 模型 ──
# v8.13.0: 模型走 HuggingFace 国内镜像（hf-mirror.com）自动下载，无需手动配置；
# 如需官方源，注释下一行即可
$env:HF_ENDPOINT = 'https://hf-mirror.com'
# v8.13-b5f: 输出实时写入 agent\logs\last_run.log（PS5.1 兼容：局部放开 EAP + stderr 规范化，杜绝 2>&1 触发终止）
$LogDir = Join-Path $AgentDir 'logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogFile = Join-Path $LogDir 'last_run.log'
Write-Host "[5/6] 检查模型（首次自动从镜像下载向量编码/重排模型，约 5-15 分钟；之后秒级启动）..." -ForegroundColor Cyan
Push-Location $AgentDir
$oldEap = $ErrorActionPreference
try {
    $ErrorActionPreference = 'Continue'
    & $VenvPy prepare_models.py 2>&1 | ForEach-Object { if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.ToString() } else { $_ } } | Tee-Object -FilePath $LogFile
    $rc = $LASTEXITCODE
} finally {
    Pop-Location
    $ErrorActionPreference = $oldEap
}
if ($rc -ne 0) {
    Write-Host "    模型准备未完全成功（完整输出见 agent\logs\last_run.log），可继续启动（部分功能降级）" -ForegroundColor Yellow
}

# ── [6/6] 启动 ──
Write-Host "[6/6] 启动服务，浏览器将自动打开 http://localhost:8000 ..." -ForegroundColor Cyan
Write-Host "      （关闭本窗口即停止服务；Key 首次在页面内填写，保存于本机；运行日志实时写入 agent\logs\last_run.log）" -ForegroundColor DarkGray
try { Start-Process 'http://localhost:8000' } catch { }
Push-Location $AgentDir
$oldEap2 = $ErrorActionPreference
try {
    $ErrorActionPreference = 'Continue'
    & $VenvPy -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 2>&1 | ForEach-Object { if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.ToString() } else { $_ } } | Tee-Object -FilePath $LogFile
    $rc = $LASTEXITCODE
} finally {
    Pop-Location
    $ErrorActionPreference = $oldEap2
}
if ($rc -ne 0) {
    Write-Host ""
    Write-Host "  ⚠ 服务未能正常启动（原因见上方红字，完整日志在 agent\logs\last_run.log）" -ForegroundColor Red
    Write-Host "    常见原因：依赖没装全 / 数据或模型没放对位置 / 端口 8000 被占用" -ForegroundColor Yellow
    Read-Host "    按回车键关闭窗口（把 agent\logs\last_run.log 整个文件发我即可）"
}
