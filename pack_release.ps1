# ============================================================
#  Citrus QA Agent 发布打包（v8.9.0）
#  ------------------------------------------------------------
#  用法:
#    powershell -File pack_release.ps1                # 主包（代码，~2MB）
#    powershell -File pack_release.ps1 -CorpusOnly    # 语料附件（~1.2GB，上传 GitHub Releases）
#    powershell -File pack_release.ps1 -IncludeData   # 主包+语料（~1.2GB，可上传）
#    powershell -File pack_release.ps1 -IncludeModels -IncludeData  # 本地完整包（~3.3GB，仅本地/其他渠道分发）
#  说明: GitHub Releases 单文件上限 2GB——大模型（reranker 2.2GB）不走发布包，
#        首次运行经 HF 镜像自动下载（run.ps1 / prepare_models.py 已内置）；
#        语料（LanceDB 向量库）用 -CorpusOnly 单独打包，run.ps1 首次运行自动下载。
#  输出: dist/citrus-qa-agent-v8.9.0[-full[-data]].zip / dist/corpus-v8.9.0.zip
#  用户拿到 zip → 解压 → 双击运行 run.ps1 → 浏览器打开即用
# ============================================================
param([switch]$IncludeModels, [switch]$IncludeData, [switch]$CorpusOnly)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$Version = '8.9.0'
$Dist = Join-Path $Root 'dist'
$Suffix = $(if ($IncludeModels) { '-full' } else { '' }) + $(if ($IncludeData) { '-data' } else { '' })
$ZipName = "citrus-qa-agent-v$Version$Suffix.zip"
$ZipPath = Join-Path $Dist $ZipName
$Stage = Join-Path $env:TEMP ("citrus-pack-" + [guid]::NewGuid().ToString('N'))

# ── 仅语料附件（zip 内路径 agent/data/...，解压到仓库根目录即得 agent/data）──
#  v8.9.0: 只打包 LanceDB 向量库 + 各批次 chunks.jsonl（证据定位必需）；
#  旧 Qdrant 批次目录（向量库本体，1.18GB 冗余）不进发布包——体积减半且远低于
#  GitHub 2GiB 单文件上限；本地仍完整保留，可随时回退 qdrant 后端。
if ($CorpusOnly) {
    $DataDir = Join-Path $Root 'agent\data'
    if (-not (Test-Path $DataDir)) {
        Write-Host "⚠ agent/data 不存在，无法打包语料" -ForegroundColor Red
        exit 1
    }
    Write-Host "打包语料附件 corpus-v$Version.zip (LanceDB + chunks.jsonl) ..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Force -Path (Join-Path $Stage 'agent') | Out-Null
    $StageData = Join-Path $Stage 'agent\data'
    robocopy (Join-Path $DataDir 'lancedb') (Join-Path $StageData 'lancedb') /E /XF *.lock /R:0 /W:0 /NFL /NDL /NJH /NJS | Out-Null
    Get-ChildItem $DataDir -Directory | Where-Object { Test-Path (Join-Path $_.FullName 'chunks\chunks.jsonl') } | ForEach-Object {
        $dst = Join-Path $StageData (Join-Path $_.Name 'chunks')
        New-Item -ItemType Directory -Force -Path $dst | Out-Null
        Copy-Item -Force (Join-Path $_.FullName 'chunks\chunks.jsonl') (Join-Path $dst 'chunks.jsonl')
    }
    New-Item -ItemType Directory -Force -Path $Dist | Out-Null
    $CorpusZipPath = Join-Path $Dist "corpus-v$Version.zip"
    if (Test-Path $CorpusZipPath) { Remove-Item $CorpusZipPath -Force }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::CreateFromDirectory(
        $Stage, $CorpusZipPath,
        [System.IO.Compression.CompressionLevel]::NoCompression, $false)
    $sizeMB = [Math]::Round((Get-Item $CorpusZipPath).Length / 1MB, 1)
    Remove-Item -Recurse -Force $Stage
    Write-Host ""
    Write-Host "✅ 语料附件打包完成: $CorpusZipPath  ($sizeMB MB)" -ForegroundColor Green
    Write-Host "   上传到 GitHub Releases v$Version，run.ps1 首次运行自动下载"
    exit 0
}

$ExcludeDirs = @('state', 'logs', 'workspace', 'data', '__pycache__',
                 '.pytest_cache', '.tmp_runner', '.venv', '.git', '.hf_cache',
                 'tests')
$ExcludeFiles = @('.env', '*.pyc', '*.tmp', '*.lock', 'MODEL_ROUTING.md')
$DataDir = Join-Path $Root 'agent\data'

Write-Host "打包 Citrus QA Agent v$Version $(if ($IncludeModels) { '(含模型) ' } else { '(主包) ' })$(if ($IncludeData) { '(含示例语料) ' } else { '' })" -ForegroundColor Yellow

# ── 暂存区 ──
New-Item -ItemType Directory -Force -Path $Stage | Out-Null
$StageAgent = Join-Path $Stage 'agent'
New-Item -ItemType Directory -Force -Path $StageAgent | Out-Null

# ── 复制 agent/（排除运行时目录与密钥文件；-notlike 对数组失效须逐项判断）──
Get-ChildItem -Path (Join-Path $Root 'agent') -Force | ForEach-Object {
    $skip = $_.Name -in $ExcludeDirs
    if (-not $skip) {
        foreach ($pat in $ExcludeFiles) {
            if ($_.Name -like $pat) { $skip = $true; break }
        }
    }
    if ($skip) { return }
    if ($_.PSIsContainer) {
        Copy-Item -Recurse -Force $_.FullName (Join-Path $StageAgent $_.Name)
    } else {
        Copy-Item -Force $_.FullName (Join-Path $StageAgent $_.Name)
    }
}
# 确保目录结构
New-Item -ItemType Directory -Force -Path (Join-Path $StageAgent 'workspace\output') | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $StageAgent 'state') | Out-Null

# ── 完整包: 附带模型缓存 ──
if ($IncludeModels) {
    $HfCache = Join-Path $Root 'agent\.hf_cache'
    if (Test-Path $HfCache) {
        Write-Host "附带模型缓存 (.hf_cache) ..." -ForegroundColor Cyan
        $HfStage = Join-Path $StageAgent '.hf_cache'
        if (Test-Path $HfStage) { Remove-Item -Recurse -Force $HfStage }
        # robocopy: 跳过被占用文件（模型被运行中服务加载）+ 排除锁文件
        robocopy $HfCache $HfStage /E /XF *.lock /R:0 /W:0 /NFL /NDL /NJH /NJS | Out-Null
    } else {
        Write-Host "⚠ agent/.hf_cache 不存在，跳过模型缓存" -ForegroundColor Yellow
    }
}

# ── 完整包: 附带示例语料（LanceDB 向量库 + chunks.jsonl，开箱可测检索/引用/写作全链路）──
if ($IncludeData) {
    if (Test-Path $DataDir) {
        Write-Host "附带示例语料 (data/lancedb + chunks.jsonl) ..." -ForegroundColor Cyan
        $DataStage = Join-Path $StageAgent 'data'
        if (Test-Path $DataStage) { Remove-Item -Recurse -Force $DataStage }
        robocopy (Join-Path $DataDir 'lancedb') (Join-Path $DataStage 'lancedb') /E /XF *.lock /R:0 /W:0 /NFL /NDL /NJH /NJS | Out-Null
        Get-ChildItem $DataDir -Directory | Where-Object { Test-Path (Join-Path $_.FullName 'chunks\chunks.jsonl') } | ForEach-Object {
            $dst = Join-Path $DataStage (Join-Path $_.Name 'chunks')
            New-Item -ItemType Directory -Force -Path $dst | Out-Null
            Copy-Item -Force (Join-Path $_.FullName 'chunks\chunks.jsonl') (Join-Path $dst 'chunks.jsonl')
        }
    } else {
        Write-Host "⚠ agent/data 不存在，跳过语料" -ForegroundColor Yellow
    }
}

# ── 清理暂存区嵌套产物（递归复制不过滤，逐层清理 __pycache__/.pyc/.env/.tmp_runner）──
Get-ChildItem -Path $Stage -Recurse -Directory -Filter '__pycache__' -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path $Stage -Recurse -Directory -Filter '.tmp_runner' -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path $Stage -Recurse -File -Include '*.pyc', '*.tmp' -ErrorAction SilentlyContinue |
    Remove-Item -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path $Stage -Recurse -File -Filter '.env' -ErrorAction SilentlyContinue |
    Remove-Item -Force -ErrorAction SilentlyContinue

# ── 根文件 ──
foreach ($f in @('README.md', 'LICENSE', 'requirements.txt', 'run.ps1', '.gitignore')) {
    $src = Join-Path $Root $f
    if (Test-Path $src) { Copy-Item -Force $src (Join-Path $Stage $f) }
}

# ── 打 zip（v8.5.0: 改用 .NET ZipFile——PS5.1 的 Compress-Archive 对 2GB+
#    模型文件 Optimal 压缩极慢/失败；模型权重高熵压缩率低，整体 NoCompression
#    秒级完成，体积差别可忽略）──
New-Item -ItemType Directory -Force -Path $Dist | Out-Null
if (Test-Path $ZipPath) { Remove-Item $ZipPath -Force }
Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory(
    $Stage, $ZipPath,
    [System.IO.Compression.CompressionLevel]::NoCompression, $false)
$sizeMB = [Math]::Round((Get-Item $ZipPath).Length / 1MB, 1)
Remove-Item -Recurse -Force $Stage

Write-Host ""
Write-Host "✅ 打包完成: $ZipPath  ($sizeMB MB)" -ForegroundColor Green
Write-Host "   用户使用: 解压 → 运行 run.ps1 → 浏览器自动打开 → 页面填 API Key"
